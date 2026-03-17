from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.strategy import Signal, BUY, SELL, EXIT_LONG, EXIT_SHORT


@dataclass(frozen=True)
class Decision:
    approve: bool
    action: str  # "enter" | "skip" | "close" | "hold"
    reason: str
    meta: Optional[dict[str, Any]] = None


class TradeGuardian:
    """
    Confirms or vetoes trades before orders are sent.

    Modes:
      - off:       always approves entry/exit
      - heuristic: simple risk/sanity checks (spread, SL/TP presence)
      - llm:       local LLM decides (fails closed on error)
    """

    def __init__(
        self,
        mode: str = "off",
        max_spread_points: Optional[float] = None,
        llm_model: Optional[str] = None,
        llm_max_new_tokens: int = 200,
    ) -> None:
        self.mode = (mode or "off").strip().lower()
        self.max_spread_points = float(max_spread_points) if max_spread_points is not None else None
        self.llm_model = llm_model
        self.llm_max_new_tokens = int(llm_max_new_tokens)

    def decide(self, mt5, signal: Signal, rates, positions) -> Decision:
        if self.mode == "off":
            return self._allow(signal, positions)
        if self.mode == "heuristic":
            return self._heuristic(mt5, signal, rates, positions)
        if self.mode == "llm":
            return self._llm(mt5, signal, rates, positions)
        return Decision(approve=False, action="skip", reason=f"Unknown AI mode '{self.mode}'")

    def _allow(self, signal: Signal, positions) -> Decision:
        if signal.direction in (BUY, SELL) and positions:
            return Decision(approve=False, action="skip", reason="Position already open")
        if signal.direction in (EXIT_LONG, EXIT_SHORT) and not positions:
            return Decision(approve=False, action="hold", reason="No position to close")
        if signal.direction in (EXIT_LONG, EXIT_SHORT):
            return Decision(approve=True, action="close", reason="Exit signal")
        if signal.direction in (BUY, SELL):
            return Decision(approve=True, action="enter", reason="Entry signal")
        return Decision(approve=True, action="hold", reason="No action")

    def _heuristic(self, mt5, signal: Signal, rates, positions) -> Decision:
        # Block re-entry while a position exists.
        if signal.direction in (BUY, SELL) and positions:
            return Decision(approve=False, action="skip", reason="Position already open")

        if signal.direction in (BUY, SELL):
            if not signal.sl_price or not signal.tp_price:
                return Decision(approve=False, action="skip", reason="Missing SL/TP")

            tick = mt5.symbol_info_tick(signal.symbol)
            info = mt5.symbol_info(signal.symbol)
            if not tick or not info:
                return Decision(approve=False, action="skip", reason="Missing tick/symbol info")

            point = float(getattr(info, "point", 0.0) or 0.0)
            if point > 0:
                spread_points = (float(tick.ask) - float(tick.bid)) / point
            else:
                spread_points = float(tick.ask) - float(tick.bid)

            if self.max_spread_points is not None and spread_points > self.max_spread_points:
                return Decision(
                    approve=False,
                    action="skip",
                    reason=f"Spread too high ({spread_points:.1f} points > {self.max_spread_points:.1f})",
                    meta={"spread_points": spread_points},
                )

            return Decision(approve=True, action="enter", reason="Heuristic checks passed", meta={"spread_points": spread_points})

        if signal.direction in (EXIT_LONG, EXIT_SHORT):
            if not positions:
                return Decision(approve=False, action="hold", reason="No position to close")
            return Decision(approve=True, action="close", reason="Exit signal")

        return Decision(approve=True, action="hold", reason="No action")

    def _llm(self, mt5, signal: Signal, rates, positions) -> Decision:
        """
        Local LLM gating.
        - Fails closed (skip/hold) if the model errors or returns invalid output.
        """
        try:
            from core.local_llm import LocalLLM

            llm = LocalLLM(model_name=self.llm_model)
            prompt = self._build_prompt(mt5, signal, rates, positions)
            text = llm.generate(prompt, max_new_tokens=self.llm_max_new_tokens)
            parsed = _extract_json(text)
            if not isinstance(parsed, dict):
                return Decision(approve=False, action="skip", reason="LLM output not JSON")

            approve = bool(parsed.get("approve", False))
            action = str(parsed.get("action", "skip")).strip().lower()
            reason = str(parsed.get("reason", ""))

            if action not in ("enter", "skip", "close", "hold"):
                action = "skip"
            if not reason:
                reason = "LLM decision"

            # Map action → approval safety
            if action in ("enter", "close"):
                return Decision(approve=approve, action=action, reason=reason, meta={"raw": text})
            return Decision(approve=False, action=action, reason=reason, meta={"raw": text})

        except Exception as exc:
            return Decision(approve=False, action="skip", reason=f"LLM error: {exc}")

    @staticmethod
    def _build_prompt(mt5, signal: Signal, rates, positions) -> str:
        # Keep it small: last 30 bars summary (close) + current spread.
        closes = np.asarray(rates["close"], dtype=np.float64)
        highs = np.asarray(rates["high"], dtype=np.float64)
        lows = np.asarray(rates["low"], dtype=np.float64)

        tail = 30
        closes_tail = closes[-tail:].round(6).tolist()
        highs_tail = highs[-tail:].round(6).tolist()
        lows_tail = lows[-tail:].round(6).tolist()

        tick = mt5.symbol_info_tick(signal.symbol)
        info = mt5.symbol_info(signal.symbol)
        spread = None
        spread_points = None
        if tick and info and getattr(info, "point", 0):
            spread = float(tick.ask) - float(tick.bid)
            spread_points = spread / float(info.point)

        pos_summary = []
        for p in positions or []:
            pos_summary.append(
                {
                    "ticket": int(getattr(p, "ticket", 0) or 0),
                    "type": int(getattr(p, "type", -1)),
                    "volume": float(getattr(p, "volume", 0.0) or 0.0),
                    "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
                    "profit": float(getattr(p, "profit", 0.0) or 0.0),
                }
            )

        return (
            "You are a risk gate for an automated MetaTrader5 trading bot.\n"
            "Return ONLY valid JSON.\n"
            'Schema: {"approve":true|false,"action":"enter|skip|close|hold","reason":"..."}\n'
            "Rules:\n"
            "- Prefer SKIP over ENTER unless the setup is clearly valid.\n"
            "- If there is already an open position, never ENTER.\n"
            "- If the bot produced an EXIT_* signal and a matching position exists, CLOSE unless spread is extreme.\n"
            "\n"
            f"signal={{symbol:{signal.symbol!r},direction:{signal.direction!r},volume:{signal.volume},sl:{signal.sl_price},tp:{signal.tp_price},comment:{signal.comment!r}}}\n"
            f"spread_points={spread_points}\n"
            f"closes_tail={closes_tail}\n"
            f"highs_tail={highs_tail}\n"
            f"lows_tail={lows_tail}\n"
            f"positions={pos_summary}\n"
        )


def _extract_json(text: str):
    import json
    import re

    if not text:
        return None

    # Try direct parse first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract the first {...} block.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

