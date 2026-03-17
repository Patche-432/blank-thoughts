# core/bot.py
# Main trading loop.
# Connects to MT5, polls the strategy on every tick, and places market orders.
#
# Run standalone:   python -m core.bot
# Or from web.py:   from core.bot import Bot; bot.run()

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script: `python core/bot.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
import os
import threading
from typing import Optional

from core.connection import MT5Connection, MT5ConnectionError
from core.guardian import Decision, TradeGuardian
from core.strategy import Strategy, Signal, BUY, SELL, HOLD, EXIT_LONG, EXIT_SHORT

log = logging.getLogger(__name__)


class BotError(RuntimeError):
    """Raised when the bot hits a condition it cannot recover from."""


class Bot:
    """
    Trading bot - fetches M5 bars, evaluates the strategy, and places orders.

    Args:
        symbol:     Forex pair, e.g. "EURUSD"
        volume:     Lot size, e.g. 0.01
        timeframe:  MT5 timeframe constant (default: TIMEFRAME_M5 = 5)
        bar_count:  Bars to fetch per tick (should exceed indicator lookbacks)
        poll_secs:  Seconds between strategy evaluations
        dry_run:    If True, never sends orders (default: True)
        max_pos:    Max open positions per symbol (default: 1)
        ai_mode:    Trade confirmation mode: off|heuristic|llm (default: heuristic)
        max_spread_points: Optional spread cap (points) for entries
        strategy:   Optional pre-built Strategy (useful for testing)
    """

    # MT5 timeframe constant - defined here so callers don't need to import mt5
    TIMEFRAME_M5 = 5

    def __init__(
        self,
        symbol: str = "EURUSD",
        volume: float = 0.01,
        timeframe: int = TIMEFRAME_M5,
        bar_count: int = 250,
        poll_secs: float = 10.0,
        dry_run: Optional[bool] = None,
        max_pos: int = 1,
        ai_mode: Optional[str] = None,
        max_spread_points: Optional[float] = None,
        ai_model: Optional[str] = None,
        strategy: Optional[Strategy] = None,
    ) -> None:
        if volume <= 0:
            raise ValueError(f"volume must be > 0, got {volume}")
        if bar_count < 1:
            raise ValueError(f"bar_count must be >= 1, got {bar_count}")
        if poll_secs < 1:
            raise ValueError(f"poll_secs must be >= 1, got {poll_secs}")
        if max_pos < 0:
            raise ValueError(f"max_pos must be >= 0, got {max_pos}")

        self.symbol = symbol.strip().upper()
        self.volume = volume
        self.timeframe = timeframe
        self.bar_count = bar_count
        self.poll_secs = poll_secs
        self.max_pos = max_pos
        self.strategy = strategy or Strategy(symbol=self.symbol, volume=volume)

        if ai_mode is None:
            ai_mode = (os.getenv("BOT_AI_MODE", "heuristic") or "heuristic").strip().lower()
        self.guardian = TradeGuardian(
            mode=ai_mode,
            max_spread_points=max_spread_points,
            llm_model=ai_model,
        )

        if dry_run is None:
            dry_run_env = (os.getenv("BOT_DRY_RUN", "1") or "1").strip().lower()
            dry_run = dry_run_env not in ("0", "false", "no", "off")
        self.dry_run = bool(dry_run)

        self._running = False
        self._stop_event = threading.Event()
        self._last_signal: Optional[Signal] = None
        self._last_decision: Optional[Decision] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Connect to MT5 and enter the signal loop. Blocks until stopped.

        Raises:
            MT5ConnectionError: If the initial MT5 connection fails.
            BotError:           If the symbol is invalid or unavailable.
        """
        log.info(
            "Bot starting - symbol=%s  volume=%s  bars=%d  poll=%.1fs  dry_run=%s",
            self.symbol,
            self.volume,
            self.bar_count,
            self.poll_secs,
            self.dry_run,
        )
        try:
            with MT5Connection() as conn:
                self._validate_symbol_or_raise(conn)
                self._running = True
                log.info("Signal loop active - call stop() or press Ctrl+C to exit.")
                self._loop(conn)

        except KeyboardInterrupt:
            log.info("Stopped by keyboard interrupt.")
        except (MT5ConnectionError, BotError):
            raise  # let the caller (web.py thread) log and handle these
        except Exception as exc:
            # Wrap unexpected errors so the caller gets a consistent type
            raise BotError(f"Unexpected fatal error: {exc}") from exc
        finally:
            self._running = False
            log.info("Bot shut down.")

    def stop(self) -> None:
        """Thread-safe stop. Wakes the poll loop immediately."""
        log.info("Stop requested.")
        self._running = False
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._running

    def last_signal(self) -> Optional[Signal]:
        return self._last_signal

    def last_decision(self) -> Optional[Decision]:
        return self._last_decision

    # ── Tick loop ─────────────────────────────────────────────────────────────

    def _loop(self, conn: MT5Connection) -> None:
        """Poll the strategy every poll_secs seconds until stopped."""
        while self._running:
            try:
                self._tick(conn)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Log tick errors but keep the loop alive —
                # transient MT5 hiccups shouldn't kill the bot.
                log.error("Tick error (will retry next poll): %s", exc)

            # wait() respects stop() waking us early via the event
            self._stop_event.wait(timeout=self.poll_secs)
            self._stop_event.clear()

    def _tick(self, conn: MT5Connection) -> None:
        """Fetch bars → evaluate signal → place order if actionable."""
        import MetaTrader5 as mt5

        rates = self._fetch_rates(mt5)
        if rates is None or len(rates) == 0:
            log.warning("No rate data returned - skipping tick.")
            return

        signal = self.strategy.get_signal(rates)
        self._last_signal = signal

        positions = self._get_positions(mt5, self.symbol)

        decision = self.guardian.decide(mt5, signal, rates, positions)
        self._last_decision = decision
        log.info("AI decision: approve=%s action=%s reason=%s", decision.approve, decision.action, decision.reason)

        if decision.action == "close" and decision.approve:
            self._handle_exit_signal(mt5, signal, positions)
            return

        if positions:
            log.debug("Position open (%d) - no new entries.", len(positions))
            return

        if decision.action == "enter" and decision.approve:
            if signal.direction == BUY:
                log.info("► BUY   TP=%.5f  SL=%.5f", signal.tp_price or 0.0, signal.sl_price or 0.0)
                self._place_order(mt5, signal, mt5.ORDER_TYPE_BUY)
                return
            if signal.direction == SELL:
                log.info("► SELL  TP=%.5f  SL=%.5f", signal.tp_price or 0.0, signal.sl_price or 0.0)
                self._place_order(mt5, signal, mt5.ORDER_TYPE_SELL)
                return

        log.debug("HOLD/skip.")

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_rates(self, mt5):
        """Pull bar_count bars from MT5. Returns a numpy array or None."""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.bar_count)
            if rates is None:
                code, msg = mt5.last_error()
                log.error(
                    "copy_rates_from_pos failed [%d] %s - check that %s is in Market Watch.",
                    code,
                    msg,
                    self.symbol,
                )
            return rates
        except Exception as exc:
            log.error("_fetch_rates raised: %s", exc)
            return None

    # ── Order placement ───────────────────────────────────────────────────────

    def _place_order(self, mt5, signal: Signal, order_type: int) -> Optional[int]:
        """
        Send a market order to MT5.

        Returns the ticket number on success, or None on any failure.
        Invalid SL/TP levels are detected before sending and cancel the order.
        """
        if self.dry_run:
            log.warning("DRY RUN: would send %s %s vol=%.2f", signal.direction, signal.symbol, signal.volume)
            return None

        if self.max_pos > 0 and not self._can_open_new_position(mt5, signal.symbol, self.max_pos):
            return None

        try:
            tick = mt5.symbol_info_tick(signal.symbol)
            sym_info = mt5.symbol_info(signal.symbol)

            if not tick:
                log.error("No tick data for %s - order cancelled.", signal.symbol)
                return None
            if not sym_info:
                log.error("No symbol info for %s - order cancelled.", signal.symbol)
                return None

            is_buy = order_type == mt5.ORDER_TYPE_BUY
            price = tick.ask if is_buy else tick.bid
            digits = sym_info.digits
            sl = round(signal.sl_price, digits) if signal.sl_price else 0.0
            tp = round(signal.tp_price, digits) if signal.tp_price else 0.0
            side = "BUY" if is_buy else "SELL"

            if not self._levels_are_valid(is_buy, price, sl, tp, side):
                return None

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": float(signal.volume),
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 20240101,
                "comment": signal.comment or "BB_bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            log.info(
                "Sending %s | %s | vol=%.2f | price=%.5f | sl=%.5f | tp=%.5f",
                side,
                signal.symbol,
                signal.volume,
                price,
                sl,
                tp,
            )

            result = mt5.order_send(request)
            return self._handle_order_result(mt5, result)

        except Exception as exc:
            log.error("_place_order raised: %s", exc)
            return None

    @staticmethod
    def _get_positions(mt5, symbol: str):
        try:
            return list(mt5.positions_get(symbol=symbol) or [])
        except Exception as exc:
            log.error("positions_get failed: %s", exc)
            return []

    def _handle_exit_signal(self, mt5, signal: Signal, positions) -> None:
        if not positions:
            log.debug("Exit signal but no positions - ignoring.")
            return

        if signal.direction == EXIT_LONG:
            wanted = mt5.POSITION_TYPE_BUY
            side = "LONG"
        else:
            wanted = mt5.POSITION_TYPE_SELL
            side = "SHORT"

        to_close = [p for p in positions if getattr(p, "type", None) == wanted]
        if not to_close:
            log.debug("Exit %s signal but no matching positions - ignoring.", side)
            return

        if self.dry_run:
            log.warning("DRY RUN: would close %d %s position(s) for %s", len(to_close), side, signal.symbol)
            return

        for p in to_close:
            self._close_position(mt5, p)

    def _close_position(self, mt5, position) -> Optional[int]:
        try:
            symbol = position.symbol
            volume = float(position.volume)
            ticket = int(position.ticket)

            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                log.error("No tick data for %s - cannot close position %d.", symbol, ticket)
                return None

            is_long = position.type == mt5.POSITION_TYPE_BUY
            order_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
            price = tick.bid if is_long else tick.ask
            side = "CLOSE_LONG" if is_long else "CLOSE_SHORT"

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": ticket,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 10,
                "magic": 20240101,
                "comment": "BB_bot_exit",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            log.info("Sending %s | %s | vol=%.2f | price=%.5f | pos=%d", side, symbol, volume, price, ticket)
            result = mt5.order_send(request)
            return self._handle_order_result(mt5, result)

        except Exception as exc:
            log.error("_close_position raised: %s", exc)
            return None

    @staticmethod
    def _can_open_new_position(mt5, symbol: str, max_pos: int) -> bool:
        try:
            positions = mt5.positions_get(symbol=symbol) or []
        except Exception as exc:
            log.error("positions_get failed: %s", exc)
            return False

        if len(positions) >= max_pos:
            log.info("Position limit reached for %s (%d/%d) - skipping order.", symbol, len(positions), max_pos)
            return False
        return True

    def _levels_are_valid(self, is_buy: bool, price: float, sl: float, tp: float, side: str) -> bool:
        """
        Sanity-check that SL and TP are on the correct side of the entry price.
        Returns False (and logs the problem) if either level is invalid.
        """
        if sl:
            sl_wrong = (is_buy and sl >= price) or (not is_buy and sl <= price)
            if sl_wrong:
                log.error(
                    "SL %.5f is on the wrong side of entry %.5f for %s - order cancelled.",
                    sl,
                    price,
                    side,
                )
                return False

        if tp:
            tp_wrong = (is_buy and tp <= price) or (not is_buy and tp >= price)
            if tp_wrong:
                log.error(
                    "TP %.5f is on the wrong side of entry %.5f for %s - order cancelled.",
                    tp,
                    price,
                    side,
                )
                return False

        return True

    def _handle_order_result(self, mt5, result) -> Optional[int]:
        """Return the ticket number on success, or log the reason for failure."""
        if result is None:
            code, msg = mt5.last_error()
            log.error("order_send returned None [%d] %s", code, msg)
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order rejected - retcode=%d  comment=%s", result.retcode, result.comment)
            return None
        log.info("Order placed - ticket=%d", result.order)
        return result.order

    # ── Startup validation ────────────────────────────────────────────────────

    def _validate_symbol_or_raise(self, conn: MT5Connection) -> None:
        """
        Ensure the symbol exists and is visible in MT5's Market Watch.

        Raises:
            BotError: If the symbol is unknown or cannot be added.
        """
        import MetaTrader5 as mt5

        try:
            info = mt5.symbol_info(self.symbol)
            if info is None:
                raise BotError(
                    f"Symbol '{self.symbol}' not found in MT5. "
                    "Check the symbol name and that your broker offers it."
                )
            if not info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    raise BotError(
                        f"Could not add '{self.symbol}' to Market Watch. "
                        "Add it manually in MT5 and restart."
                    )
            log.info("Symbol %s OK", self.symbol)
        except BotError:
            raise
        except Exception as exc:
            raise BotError(f"Symbol validation failed for '{self.symbol}': {exc}") from exc


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Bot(
        symbol="EURUSD",
        volume=0.01,
        timeframe=Bot.TIMEFRAME_M5,
        bar_count=250,
        poll_secs=10.0,
    ).run()
