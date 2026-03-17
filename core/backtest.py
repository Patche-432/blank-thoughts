from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script: `python core/backtest.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.strategy import BUY, SELL


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str  # BUY | SELL
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "tp"|"sl"|"exit"|"eod"

    @property
    def pnl_price(self) -> float:
        mult = 1.0 if self.side == BUY else -1.0
        return (self.exit_price - self.entry_price) * mult


def _infer_pip_size(symbol: str) -> float:
    s = (symbol or "").upper()
    return 0.01 if "JPY" in s else 0.0001


def _dt_from_rate_time(t) -> datetime:
    # MT5 rates: "time" is usually epoch seconds (int).
    try:
        return datetime.fromtimestamp(int(t), tz=timezone.utc)
    except Exception:
        return datetime.now(tz=timezone.utc)


def _load_rates_from_csv(path: str):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        raise ValueError("CSV contains no rows")

    def parse_time(x: str) -> int:
        x = (x or "").strip()
        if not x:
            raise ValueError("Missing time")
        if x.isdigit():
            v = int(x)
            # Many exports use epoch milliseconds.
            if len(x) > 10:
                v = v // 1000
            return v
        # ISO-ish
        dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    dtype = np.dtype(
        [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
        ]
    )
    arr = np.empty(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        arr[i]["time"] = parse_time(
            row.get("time")
            or row.get("timestamp")
            or row.get("datetime")
            or row.get("date")
            or ""
        )
        arr[i]["open"] = float(row["open"])
        arr[i]["high"] = float(row["high"])
        arr[i]["low"] = float(row["low"])
        arr[i]["close"] = float(row["close"])
    return arr


def _align_bid_ask(bid_rates, ask_rates):
    """
    Align bid/ask arrays by timestamp (seconds).
    Returns (bid_aligned, ask_aligned) with matching times.
    """
    i = 0
    j = 0
    bi: list[int] = []
    ai: list[int] = []

    while i < len(bid_rates) and j < len(ask_rates):
        tb = int(bid_rates[i]["time"])
        ta = int(ask_rates[j]["time"])
        if tb == ta:
            bi.append(i)
            ai.append(j)
            i += 1
            j += 1
        elif tb < ta:
            i += 1
        else:
            j += 1

    return bid_rates[bi], ask_rates[ai]


def _mid_from_bid_ask(bid_rates, ask_rates):
    mid = np.empty(len(bid_rates), dtype=bid_rates.dtype)
    mid["time"] = bid_rates["time"]
    for f in ("open", "high", "low", "close"):
        mid[f] = (bid_rates[f] + ask_rates[f]) / 2.0
    return mid


def _load_rates_from_mt5(symbol: str, timeframe: str, start: datetime, end: datetime):
    import MetaTrader5 as mt5

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M10": mt5.TIMEFRAME_M10,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    tf = tf_map.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Try M5, M15, H1, D1.")

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        code, msg = mt5.last_error()
        raise RuntimeError(f"copy_rates_range returned no data [{code}] {msg}")
    return rates


def backtest(
    rates,
    symbol: str,
    *,
    pip_size: float,
    bid_rates=None,
    ask_rates=None,
    spread: float = 0.0,
    sl_tp_order: str = "worst",
    strategy_kwargs: Optional[dict] = None,
) -> list[Trade]:
    strategy_kwargs = strategy_kwargs or {}
    bb_period = int(strategy_kwargs.get("bb_period", 24))
    bb_dev = float(strategy_kwargs.get("bb_dev", 1.7))
    adx_period = int(strategy_kwargs.get("adx_period", 14))
    adx_threshold = float(strategy_kwargs.get("adx_threshold", 25.0))
    swing_lookback = int(strategy_kwargs.get("swing_lookback", 8))
    take_profit = float(strategy_kwargs.get("take_profit", 0.2414))
    require_confirmation = bool(strategy_kwargs.get("require_confirmation", True))

    trades: list[Trade] = []
    position = None  # dict

    closes = np.asarray(rates["close"], dtype=np.float64)
    highs = np.asarray(rates["high"], dtype=np.float64)
    lows = np.asarray(rates["low"], dtype=np.float64)

    mid, upper, lower = _bollinger(closes, period=bb_period, dev=bb_dev)
    adx = _adx(highs, lows, closes, period=adx_period)
    swing_low = _rolling_min(lows, window=swing_lookback)
    swing_high = _rolling_max(highs, window=swing_lookback)

    # Need enough bars for indicators (ADX needs ~2*period, BB needs period).
    warmup = max(bb_period + 2, (adx_period * 2) + 5, swing_lookback + 2, 60)
    for i in range(warmup, len(rates)):
        bar = rates[i]
        t = _dt_from_rate_time(bar["time"])

        bid_bar = bid_rates[i] if bid_rates is not None else None
        ask_bar = ask_rates[i] if ask_rates is not None else None

        # Manage open position first (SL/TP intrabar).
        if position is not None:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]

            if bid_bar is not None and ask_bar is not None:
                # Longs are executed on bid; shorts on ask.
                high_bid = float(bid_bar["high"])
                low_bid = float(bid_bar["low"])
                high_ask = float(ask_bar["high"])
                low_ask = float(ask_bar["low"])
            else:
                high_bid = float(bar["high"])
                low_bid = float(bar["low"])
                high_ask = float(bar["high"])
                low_ask = float(bar["low"])

            hit_sl = False
            hit_tp = False
            if side == BUY:
                if sl is not None and low_bid <= sl:
                    hit_sl = True
                if tp is not None and high_bid >= tp:
                    hit_tp = True
            else:
                if sl is not None and high_ask >= sl:
                    hit_sl = True
                if tp is not None and low_ask <= tp:
                    hit_tp = True

            exit_reason = None
            exit_price = None

            if hit_sl and hit_tp:
                if sl_tp_order in ("sl-first", "worst"):
                    exit_reason = "sl"
                elif sl_tp_order in ("tp-first", "best"):
                    exit_reason = "tp"
                else:
                    exit_reason = "sl"
            elif hit_sl:
                exit_reason = "sl"
            elif hit_tp:
                exit_reason = "tp"

            if exit_reason == "sl":
                exit_price = float(sl)
            elif exit_reason == "tp":
                exit_price = float(tp)

            if exit_reason and exit_price is not None:
                if bid_bar is None or ask_bar is None:
                    # Apply fixed spread on exit (paying the spread).
                    exit_price = exit_price - spread if side == BUY else exit_price + spread
                trades.append(
                    Trade(
                        symbol=symbol,
                        side=side,
                        entry_time=position["entry_time"],
                        entry_price=position["entry_price"],
                        exit_time=t,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                    )
                )
                position = None
                continue

        # Strategy signals for this bar.
        if not np.isfinite(mid[i]) or not np.isfinite(lower[i]) or not np.isfinite(upper[i]) or not np.isfinite(adx[i]):
            continue

        exit_long = closes[i] > mid[i] and closes[i - 1] > mid[i - 1]
        exit_short = closes[i] < mid[i] and closes[i - 1] < mid[i - 1]

        # Strategy-driven exit (close at close).
        if position is not None and (exit_long or exit_short):
            if bid_bar is not None and ask_bar is not None:
                close_price = float(bid_bar["close"]) if position["side"] == BUY else float(ask_bar["close"])
            else:
                close_price = float(bar["close"])
            side = position["side"]
            if bid_bar is None or ask_bar is None:
                close_price = close_price - spread if side == BUY else close_price + spread
            trades.append(
                Trade(
                    symbol=symbol,
                    side=side,
                    entry_time=position["entry_time"],
                    entry_price=position["entry_price"],
                    exit_time=t,
                    exit_price=close_price,
                    exit_reason="exit",
                )
            )
            position = None
            continue

        if adx[i] < adx_threshold:
            continue

        if require_confirmation:
            buy_signal = closes[i - 1] < lower[i - 1] and closes[i] > lower[i]
            sell_signal = closes[i - 1] > upper[i - 1] and closes[i] < upper[i]
        else:
            buy_signal = closes[i] < lower[i]
            sell_signal = closes[i] > upper[i]

        # Entry (enter at close).
        if position is None and (buy_signal or sell_signal):
            side = BUY if buy_signal else SELL
            if bid_bar is not None and ask_bar is not None:
                entry = float(ask_bar["close"]) if side == BUY else float(bid_bar["close"])
            else:
                entry = float(bar["close"])
            if bid_bar is None or ask_bar is None:
                # Apply fixed spread on entry.
                entry = entry + spread if side == BUY else entry - spread

            sl = float(swing_low[i]) if side == BUY else float(swing_high[i])
            tp = float(closes[i]) * (1.0 + take_profit) if side == BUY else float(closes[i]) * (1.0 - take_profit)
            position = {
                "side": side,
                "entry_time": t,
                "entry_price": entry,
                "sl": sl,
                "tp": tp,
            }

    # Close any open position at end-of-data.
    if position is not None:
        last = rates[-1]
        t = _dt_from_rate_time(last["time"])
        if bid_rates is not None and ask_rates is not None:
            price = float(bid_rates[-1]["close"]) if position["side"] == BUY else float(ask_rates[-1]["close"])
        else:
            price = float(last["close"])
        side = position["side"]
        if bid_rates is None or ask_rates is None:
            price = price - spread if side == BUY else price + spread
        trades.append(
            Trade(
                symbol=symbol,
                side=side,
                entry_time=position["entry_time"],
                entry_price=position["entry_price"],
                exit_time=t,
                exit_price=price,
                exit_reason="eod",
            )
        )

    return trades


def _bollinger(closes: np.ndarray, period: int, dev: float):
    closes = np.asarray(closes, dtype=np.float64)
    n = closes.size
    mid = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return mid, upper, lower

    c1 = np.cumsum(closes, dtype=np.float64)
    c2 = np.cumsum(closes * closes, dtype=np.float64)

    for i in range(period - 1, n):
        j = i - period
        sum1 = c1[i] - (c1[j] if j >= 0 else 0.0)
        sum2 = c2[i] - (c2[j] if j >= 0 else 0.0)
        mean = sum1 / period
        var = (sum2 / period) - (mean * mean)
        var = max(var, 0.0)
        std = var ** 0.5
        mid[i] = mean
        upper[i] = mean + dev * std
        lower[i] = mean - dev * std

    return mid, upper, lower


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < (period * 2) + 2:
        return out

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    def wilder(values: np.ndarray, p: int) -> np.ndarray:
        s = np.full(values.shape[0], np.nan, dtype=np.float64)
        s[p - 1] = values[:p].sum()
        for k in range(p, values.shape[0]):
            s[k] = s[k - 1] - (s[k - 1] / p) + values[k]
        return s

    tr_s = wilder(tr, period)
    plus_s = wilder(plus_dm, period)
    minus_s = wilder(minus_dm, period)

    tr_ok = np.isfinite(tr_s) & (tr_s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(tr_ok, 100.0 * (plus_s / tr_s), np.nan)
        minus_di = np.where(tr_ok, 100.0 * (minus_s / tr_s), np.nan)
        denom = plus_di + minus_di
        denom_ok = np.isfinite(denom) & (denom > 0)
        dx = np.where(denom_ok, 100.0 * (np.abs(plus_di - minus_di) / denom), np.nan)
    dx = np.where(tr_ok & ~denom_ok, 0.0, dx)

    adx = np.full(dx.shape[0], np.nan, dtype=np.float64)
    start = (period - 1) + (period - 1)
    seed = dx[(period - 1) : (period - 1) + period]
    adx[start] = float(np.nanmean(seed)) if np.isfinite(seed).any() else 0.0
    for k in range(start + 1, dx.shape[0]):
        dx_k = dx[k] if np.isfinite(dx[k]) else 0.0
        adx[k] = ((adx[k - 1] * (period - 1)) + dx_k) / period

    # Align: adx array is for bars 1..n-1
    out[1:] = adx
    return out


def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.size, np.nan, dtype=np.float64)
    for i in range(window - 1, values.size):
        out[i] = float(np.min(values[i - window + 1 : i + 1]))
    return out


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.size, np.nan, dtype=np.float64)
    for i in range(window - 1, values.size):
        out[i] = float(np.max(values[i - window + 1 : i + 1]))
    return out


def _summarize(trades: list[Trade], pip_size: float) -> dict:
    if not trades:
        return {"trades": 0}

    pnls = np.array([t.pnl_price for t in trades], dtype=np.float64)
    wins = pnls > 0
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity

    return {
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()) if len(trades) else 0.0,
        "pnl_price": float(pnls.sum()),
        "pnl_pips": float((pnls / pip_size).sum()) if pip_size else None,
        "avg_pips": float((pnls / pip_size).mean()) if pip_size else None,
        "max_drawdown_price": float(dd.max()) if len(dd) else 0.0,
        "max_drawdown_pips": float((dd / pip_size).max()) if pip_size else None,
    }


def _write_trades(path: str, trades: list[Trade], pip_size: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "side", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "pnl_price", "pnl_pips"])
        for t in trades:
            pnl = t.pnl_price
            w.writerow(
                [
                    t.symbol,
                    t.side,
                    t.entry_time.isoformat(),
                    f"{t.entry_price:.6f}",
                    t.exit_time.isoformat(),
                    f"{t.exit_price:.6f}",
                    t.exit_reason,
                    f"{pnl:.6f}",
                    f"{(pnl / pip_size):.2f}" if pip_size else "",
                ]
            )


def main() -> int:
    if len(__import__("sys").argv) == 1:
        auto_rc = _auto_run_latest_combined()
        if auto_rc is not None:
            return auto_rc
        raise SystemExit(
            "No arguments provided and no combined CSV found.\n"
            "Run one of:\n"
            "  py -m core.combine_quotes --bid-csv <bid.csv> --ask-csv <ask.csv> --out .\\core\\data\\combined\\mid.csv\n"
            "  py -m core.backtest --csv .\\core\\data\\combined\\mid.csv --symbol EURUSD --out trades.csv\n"
        )

    p = argparse.ArgumentParser(description="Simple backtest runner for the MT5 bot strategy.")
    p.add_argument("--csv", help="CSV with columns time/timestamp,open,high,low,close (timestamp can be ms)")
    p.add_argument("--bid-csv", help="Bid CSV export (timestamp,open,high,low,close)")
    p.add_argument("--ask-csv", help="Ask CSV export (timestamp,open,high,low,close)")
    p.add_argument("--mt5", action="store_true", help="Load bars from MT5 terminal (requires MetaTrader5 + config)")

    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", default="M5", help="MT5 timeframe when using --mt5 (e.g., M5, M15, H1)")
    p.add_argument("--start", help="Start datetime (ISO), e.g. 2025-01-01T00:00:00Z")
    p.add_argument("--end", help="End datetime (ISO), e.g. 2025-02-01T00:00:00Z")

    p.add_argument("--pip-size", type=float, default=None, help="Pip size (default: inferred, JPY=0.01 else 0.0001)")
    p.add_argument("--spread", type=float, default=0.0, help="Fixed spread as absolute price (default: 0)")
    p.add_argument("--sl-tp-order", default="worst", choices=["worst", "best", "sl-first", "tp-first"])
    p.add_argument("--out", default="backtest_trades.csv", help="Output trades CSV path")

    # Strategy knobs (match our Strategy kwargs)
    p.add_argument("--bb-period", type=int, default=24)
    p.add_argument("--bb-dev", type=float, default=1.7)
    p.add_argument("--adx-period", type=int, default=14)
    p.add_argument("--adx-threshold", type=float, default=25.0)
    p.add_argument("--swing-lookback", type=int, default=8)
    p.add_argument("--take-profit", type=float, default=0.2414)
    p.add_argument("--require-confirmation", action="store_true", default=True)
    p.add_argument("--no-confirmation", dest="require_confirmation", action="store_false")

    args = p.parse_args()

    symbol = args.symbol.strip().upper()
    pip_size = float(args.pip_size) if args.pip_size else _infer_pip_size(symbol)

    strat_kwargs = {
        "bb_period": args.bb_period,
        "bb_dev": args.bb_dev,
        "adx_period": args.adx_period,
        "adx_threshold": args.adx_threshold,
        "swing_lookback": args.swing_lookback,
        "take_profit": args.take_profit,
        "require_confirmation": args.require_confirmation,
    }

    src_count = int(bool(args.csv)) + int(bool(args.mt5)) + int(bool(args.bid_csv or args.ask_csv))
    if src_count != 1:
        raise SystemExit(
            "Pick exactly one source:\n"
            "  - --csv <mid_ohlc.csv>\n"
            "  - --bid-csv <bid.csv> --ask-csv <ask.csv>\n"
            "  - --mt5 --start <iso> --end <iso>\n"
            "\n"
            "Tip: if you want a single file, run:\n"
            "  py -m core.combine_quotes --bid-csv <bid.csv> --ask-csv <ask.csv> --out mid.csv\n"
        )

    bid_rates = None
    ask_rates = None

    if args.csv:
        rates = _load_rates_from_csv(args.csv)
    elif args.bid_csv or args.ask_csv:
        if not (args.bid_csv and args.ask_csv):
            raise SystemExit("--bid-csv requires --ask-csv (and vice versa)")
        bid_rates = _load_rates_from_csv(args.bid_csv)
        ask_rates = _load_rates_from_csv(args.ask_csv)
        bid_rates, ask_rates = _align_bid_ask(bid_rates, ask_rates)
        rates = _mid_from_bid_ask(bid_rates, ask_rates)
    else:
        # MT5 source
        from core.connection import MT5Connection

        if not args.start or not args.end:
            raise SystemExit("--mt5 requires --start and --end (ISO datetimes)")

        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        with MT5Connection() as _:
            rates = _load_rates_from_mt5(symbol, args.timeframe, start, end)

    trades = backtest(
        rates,
        symbol,
        pip_size=pip_size,
        bid_rates=bid_rates,
        ask_rates=ask_rates,
        spread=float(args.spread),
        sl_tp_order=args.sl_tp_order,
        strategy_kwargs=strat_kwargs,
    )
    summary = _summarize(trades, pip_size)
    _write_trades(args.out, trades, pip_size)

    print("Backtest summary")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"Trades written to: {args.out}")
    return 0


def _auto_run_latest_combined() -> Optional[int]:
    """
    Convenience behavior:
    If you run `python core/backtest.py` (no args), we look for the newest CSV under
    `core/data/combined/` and backtest it automatically.
    """
    import sys

    root = Path(__file__).resolve().parents[1]
    combined_dir = root / "core" / "data" / "combined"
    if not combined_dir.exists():
        combined_dir = root / "data" / "combined"
    if not combined_dir.exists():
        return None

    csvs = [p for p in combined_dir.glob("*.csv") if not p.name.lower().startswith("trades_")]
    if not csvs:
        return None

    ohlc_csvs = [p for p in csvs if _looks_like_ohlc_csv(p)]
    if not ohlc_csvs:
        return None

    latest = max(ohlc_csvs, key=lambda p: p.stat().st_mtime)

    # Infer symbol from filename prefix (e.g. eurusd-m5-mid-2025.csv).
    name = latest.name
    sym = (name.split("-")[0] or "EURUSD").upper()
    results_dir = root / "core" / "data" / "results"
    out = str((results_dir / f"trades_{latest.stem}.csv").resolve())

    print(f"Auto backtest using: {latest}")
    sys.argv = [sys.argv[0], "--csv", str(latest), "--symbol", sym, "--out", out]
    return main()


def _looks_like_ohlc_csv(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = (f.readline() or "").strip().lower()
    except Exception:
        return False

    # Require OHLC columns; accept time or timestamp.
    cols = [c.strip() for c in header.split(",") if c.strip()]
    required = {"open", "high", "low", "close"}
    has_time = "time" in cols or "timestamp" in cols or "datetime" in cols or "date" in cols
    return has_time and required.issubset(set(cols))


if __name__ == "__main__":
    raise SystemExit(main())
