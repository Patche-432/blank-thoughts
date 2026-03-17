# blank-thoughts

Local web UI + optional MT5 trading bot.

## Run the web UI

- Start: `py web.py`
- Open: `http://localhost:5000/`

## MT5 bot (optional)

This project includes an MT5 polling bot in `core/bot.py`.

1) Install and open MetaTrader 5, log into your broker account, and ensure your symbol is visible in *Market Watch*.
2) Edit `config.py` and set `MT5_CONFIG` if you need explicit login/server/path.
3) Safety defaults:
   - Bot endpoints are disabled unless you set `BOT_ENABLE=1`
   - Orders are blocked by default unless you set `BOT_DRY_RUN=0` **and** `BOT_ALLOW_LIVE=1`

### Control via HTTP

- Status: `GET  http://localhost:5000/bot/status`
- Start:  `POST http://localhost:5000/bot/start`
- Stop:   `POST http://localhost:5000/bot/stop`

Example start body:

```json
{"symbol":"EURUSD","volume":0.01,"poll_secs":10,"bar_count":250,"dry_run":true,"max_pos":1}
```

### Strategy (BB + ADX)

Default strategy is implemented in `core/strategy.py` and supports a `strategy` object in `/bot/start`, for example:

```json
{
  "symbol": "EURUSD",
  "volume": 0.01,
  "dry_run": true,
  "strategy": { "bb_period": 24, "bb_dev": 1.7, "adx_period": 14, "adx_threshold": 25, "swing_lookback": 8, "take_profit": 0.2414, "require_confirmation": true }
}
```

Note: `take_profit` is a fraction (0.2414 = 24.14%).

### AI confirmation (recommended)

The bot runs every entry/exit signal through a "guardian" that can veto bad trades.

- `ai_mode: "heuristic"` uses spread + SL/TP sanity checks (fast, no model).
- `ai_mode: "llm"` uses a local Transformers model (slow/heavy; fails closed on errors).

You can pass these in `/bot/start`:

```json
{"ai_mode":"heuristic","max_spread_points":25}
```

## Backtesting

This repo includes a simple bar-by-bar backtest runner: `core/backtest.py`.

Tip: if you already combined quotes into `core/data/combined/*.csv`, you can run `py core/backtest.py` with no args and it will backtest the newest combined file automatically.

### Option A: Backtest from MT5 history

Requires MT5 installed + `MetaTrader5` Python package + a working `config.py`.

```powershell
py -m core.backtest --mt5 --symbol EURUSD --timeframe M5 --start 2025-01-01T00:00:00Z --end 2025-02-01T00:00:00Z --out trades.csv
```

### Option B: Backtest from CSV

CSV must have columns: `time,open,high,low,close` (time can be epoch seconds or ISO).

```powershell
py -m core.backtest --csv .\data\EURUSD_M5.csv --symbol EURUSD --out trades.csv
```

### Option C: Backtest from bid/ask CSV exports

If you have separate bid/ask downloads (like `core/data/download/eurusd-m5-bid-*.csv` and `core/data/download/eurusd-m5-ask-*.csv`), run:

```powershell
py -m core.backtest --bid-csv .\core\data\download\eurusd-m5-bid-2025-01-01-2025-12-31.csv --ask-csv .\core\data\download\eurusd-m5-ask-2025-01-01-2025-12-31.csv --symbol EURUSD --out trades.csv
```

### Option D: Combine bid/ask into one CSV (mid)

If you just want one file, combine bid+ask into a mid OHLC file:

```powershell
py -m core.combine_quotes --bid-csv .\core\data\download\eurusd-m5-bid-2025-01-01-2025-12-31.csv --ask-csv .\core\data\download\eurusd-m5-ask-2025-01-01-2025-12-31.csv --out .\core\data\combined\eurusd-m5-mid-2025.csv
```

Or (auto): `py .\core\combine_quotes.py` will pick the newest bid/ask pair in `core\data\download\` and write a combined CSV to `core\data\combined\`.

Then backtest with:

```powershell
py -m core.backtest --csv .\core\data\combined\eurusd-m5-mid-2025.csv --symbol EURUSD --out trades.csv
```

Notes:
- This is an approximation (no commissions/slippage by default). You can add a fixed `--spread` (absolute price).
- If SL and TP are both hit in the same candle, `--sl-tp-order` controls assumptions (`worst` by default).
