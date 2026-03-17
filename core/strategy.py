from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
EXIT_LONG = "EXIT_LONG"
EXIT_SHORT = "EXIT_SHORT"


@dataclass(frozen=True)
class Signal:
    symbol: str
    volume: float
    direction: str = HOLD
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    comment: str = ""


class BollingerBusterADXStrategy:
    """
    Bollinger Bands + ADX strategy (MT5 rates → Signal).

    Entry (with confirmation, default):
      - Long: previous close below lower BB, current close back above lower BB, ADX >= threshold
      - Short: previous close above upper BB, current close back below upper BB, ADX >= threshold

    Exit:
      - Long: last 2 closes above BB mid
      - Short: last 2 closes below BB mid

    SL/TP:
      - SL uses recent swing high/low over `swing_lookback`
      - TP is a % of entry close (take_profit)
    """

    def __init__(
        self,
        symbol: str,
        volume: float,
        bb_period: int = 24,
        bb_dev: float = 1.7,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        swing_lookback: int = 8,
        take_profit: float = 0.2414,
        require_confirmation: bool = True,
    ) -> None:
        self.symbol = symbol.strip().upper()
        self.volume = float(volume)
        self.bb_period = int(bb_period)
        self.bb_dev = float(bb_dev)
        self.adx_period = int(adx_period)
        self.adx_threshold = float(adx_threshold)
        self.swing_lookback = int(swing_lookback)
        self.take_profit = float(take_profit)
        self.require_confirmation = bool(require_confirmation)

        if self.volume <= 0:
            raise ValueError("volume must be > 0")
        if self.bb_period < 2:
            raise ValueError("bb_period must be >= 2")
        if self.bb_dev <= 0:
            raise ValueError("bb_dev must be > 0")
        if self.adx_period < 2:
            raise ValueError("adx_period must be >= 2")
        if self.adx_threshold <= 0:
            raise ValueError("adx_threshold must be > 0")
        if self.swing_lookback < 2:
            raise ValueError("swing_lookback must be >= 2")
        if self.take_profit <= 0:
            raise ValueError("take_profit must be > 0")

    def get_signal(self, rates) -> Signal:
        closes = self._extract_series(rates, "close")
        highs = self._extract_series(rates, "high")
        lows = self._extract_series(rates, "low")

        min_bars = max(self.bb_period + 1, self.swing_lookback + 1, (self.adx_period * 2) + 2)
        if closes.size < min_bars:
            return Signal(symbol=self.symbol, volume=self.volume, direction=HOLD, comment="Not enough bars")

        (mid_prev, upper_prev, lower_prev), (mid, upper, lower) = self._bb_last_two(
            closes, period=self.bb_period, dev=self.bb_dev
        )
        adx_last = self._adx_last(highs, lows, closes, period=self.adx_period)
        if not np.isfinite(adx_last):
            return Signal(symbol=self.symbol, volume=self.volume, direction=HOLD, comment="ADX unavailable")

        close_prev = float(closes[-2])
        close_now = float(closes[-1])

        exit_long = close_now > mid and close_prev > mid_prev
        exit_short = close_now < mid and close_prev < mid_prev
        if exit_long:
            return Signal(symbol=self.symbol, volume=self.volume, direction=EXIT_LONG, comment="Exit long at BB mid")
        if exit_short:
            return Signal(symbol=self.symbol, volume=self.volume, direction=EXIT_SHORT, comment="Exit short at BB mid")

        if adx_last < self.adx_threshold:
            return Signal(symbol=self.symbol, volume=self.volume, direction=HOLD, comment="ADX below threshold")

        if self.require_confirmation:
            buy_signal = close_prev < lower_prev and close_now > lower
            sell_signal = close_prev > upper_prev and close_now < upper
        else:
            buy_signal = close_now < lower
            sell_signal = close_now > upper

        if buy_signal:
            sl = float(np.min(lows[-self.swing_lookback :]))
            tp = close_now * (1.0 + self.take_profit)
            return Signal(
                symbol=self.symbol,
                volume=self.volume,
                direction=BUY,
                sl_price=sl,
                tp_price=tp,
                comment="BB+ADX buy",
            )

        if sell_signal:
            sl = float(np.max(highs[-self.swing_lookback :]))
            tp = close_now * (1.0 - self.take_profit)
            return Signal(
                symbol=self.symbol,
                volume=self.volume,
                direction=SELL,
                sl_price=sl,
                tp_price=tp,
                comment="BB+ADX sell",
            )

        return Signal(symbol=self.symbol, volume=self.volume, direction=HOLD, comment="No edge")

    @staticmethod
    def _extract_series(rates, field: str) -> np.ndarray:
        try:
            series = rates[field]
        except Exception:
            series = None
        if series is None:
            raise ValueError(f"rates must contain a '{field}' field")
        return np.asarray(series, dtype=np.float64)

    @staticmethod
    def _bb_last_two(closes: np.ndarray, period: int, dev: float):
        if closes.size < period + 1:
            raise ValueError("Not enough closes for Bollinger")

        prev_window = closes[-(period + 1) : -1]
        curr_window = closes[-period:]

        mid_prev = float(prev_window.mean())
        std_prev = float(prev_window.std(ddof=0))
        upper_prev = mid_prev + dev * std_prev
        lower_prev = mid_prev - dev * std_prev

        mid = float(curr_window.mean())
        std = float(curr_window.std(ddof=0))
        upper = mid + dev * std
        lower = mid - dev * std

        return (mid_prev, upper_prev, lower_prev), (mid, upper, lower)

    @staticmethod
    def _adx_last(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """
        Compute the last ADX value (Wilder's smoothing).
        Returns NaN if not enough data.
        """
        n = close.size
        if n < (period * 2) + 2:
            return float("nan")

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

        def wilder_smooth(values: np.ndarray, p: int) -> np.ndarray:
            out = np.full(values.shape[0], np.nan, dtype=np.float64)
            first = values[:p].sum()
            out[p - 1] = first
            for i in range(p, values.shape[0]):
                out[i] = out[i - 1] - (out[i - 1] / p) + values[i]
            return out

        tr_s = wilder_smooth(tr, period)
        plus_s = wilder_smooth(plus_dm, period)
        minus_s = wilder_smooth(minus_dm, period)

        tr_ok = np.isfinite(tr_s) & (tr_s > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = np.where(tr_ok, 100.0 * (plus_s / tr_s), np.nan)
            minus_di = np.where(tr_ok, 100.0 * (minus_s / tr_s), np.nan)

            denom = plus_di + minus_di
            denom_ok = np.isfinite(denom) & (denom > 0)
            dx = np.where(denom_ok, 100.0 * (np.abs(plus_di - minus_di) / denom), np.nan)
        # If denom collapses to 0 with valid TR (flat DM), treat DX as 0 instead of NaN.
        dx = np.where(tr_ok & ~denom_ok, 0.0, dx)

        # ADX is Wilder-smoothed DX
        adx = np.full(dx.shape[0], np.nan, dtype=np.float64)
        start = (period - 1) + (period - 1)
        if start >= dx.shape[0]:
            return float("nan")

        seed = dx[(period - 1) : (period - 1) + period]
        adx[start] = float(np.nanmean(seed)) if np.isfinite(seed).any() else 0.0
        for i in range(start + 1, dx.shape[0]):
            dx_i = dx[i] if np.isfinite(dx[i]) else 0.0
            adx[i] = ((adx[i - 1] * (period - 1)) + dx_i) / period

        return float(adx[-1])


class Strategy:
    """Default strategy used by the MT5 bot (alias for BollingerBusterADXStrategy)."""

    def __init__(self, symbol: str, volume: float, **kwargs) -> None:
        self._impl = BollingerBusterADXStrategy(symbol=symbol, volume=volume, **kwargs)

    def get_signal(self, rates) -> Signal:
        return self._impl.get_signal(rates)
