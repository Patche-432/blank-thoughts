from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    # Allow running as a script: `python core/combine_quotes.py`
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import re
from pathlib import Path

from core.backtest import _align_bid_ask, _load_rates_from_csv, _mid_from_bid_ask


def _write_mid_csv(path: str, mid_rates) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close"])
        for row in mid_rates:
            w.writerow(
                [
                    int(row["time"]),
                    f"{float(row['open']):.6f}",
                    f"{float(row['high']):.6f}",
                    f"{float(row['low']):.6f}",
                    f"{float(row['close']):.6f}",
                ]
            )

def _find_download_dirs() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        root / "core" / "data" / "download",
        root / "data" / "download",
    ]

def _auto_combine_latest() -> int | None:
    rx = re.compile(
        r"^(?P<symbol>[a-z0-9]+)-(?P<tf>[a-z0-9]+)-(?P<side>bid|ask)-(?P<start>\d{4}-\d{2}-\d{2})-(?P<end>\d{4}-\d{2}-\d{2})\.csv$",
        re.IGNORECASE,
    )

    by_key: dict[tuple[str, str, str, str], dict[str, Path]] = {}
    for d in _find_download_dirs():
        if not d.exists():
            continue
        for f in d.glob("*.csv"):
            m = rx.match(f.name)
            if not m:
                continue
            symbol = m.group("symbol").upper()
            tf = m.group("tf").upper()
            side = m.group("side").lower()
            start = m.group("start")
            end = m.group("end")
            key = (symbol, tf, start, end)
            by_key.setdefault(key, {})[side] = f

    pairs = []
    for (symbol, tf, start, end), sides in by_key.items():
        bid = sides.get("bid")
        ask = sides.get("ask")
        if bid and ask:
            pairs.append((symbol, tf, start, end, bid, ask))

    if not pairs:
        return None

    def score(p):
        _, _, _, _, bid, ask = p
        return max(bid.stat().st_mtime, ask.stat().st_mtime)

    symbol, tf, start, end, bid_path, ask_path = max(pairs, key=score)

    root = Path(__file__).resolve().parents[1]
    out_path = root / "core" / "data" / "combined" / f"{symbol.lower()}-{tf.lower()}-mid-{start}-{end}.csv"

    bid = _load_rates_from_csv(str(bid_path))
    ask = _load_rates_from_csv(str(ask_path))
    bid_a, ask_a = _align_bid_ask(bid, ask)
    mid = _mid_from_bid_ask(bid_a, ask_a)
    _write_mid_csv(str(out_path), mid)

    print(f"Auto-combined latest downloads:")
    print(f"- bid: {bid_path}")
    print(f"- ask: {ask_path}")
    print(f"- out: {out_path}")
    print(f"- bars: {len(mid)} (aligned from bid={len(bid)} ask={len(ask)})")
    return 0


def main() -> int:
    import sys

    if len(sys.argv) == 1:
        rc = _auto_combine_latest()
        if rc is not None:
            return rc
        raise SystemExit(
            "No arguments provided and no bid/ask downloads found.\n"
            "Put files under `core/data/download/` like:\n"
            "  eurusd-m5-bid-2025-01-01-2025-12-31.csv\n"
            "  eurusd-m5-ask-2025-01-01-2025-12-31.csv\n"
            "\n"
            "Or run with explicit paths:\n"
            "  py -m core.combine_quotes --bid-csv <bid.csv> --ask-csv <ask.csv> --out .\\core\\data\\combined\\mid.csv\n"
        )

    p = argparse.ArgumentParser(description="Combine bid+ask OHLC into a single mid-price OHLC CSV.")
    p.add_argument("--bid-csv", required=True, help="Bid CSV (timestamp/time + open/high/low/close)")
    p.add_argument("--ask-csv", required=True, help="Ask CSV (timestamp/time + open/high/low/close)")
    p.add_argument("--out", required=True, help="Output mid CSV path")
    args = p.parse_args()

    bid = _load_rates_from_csv(args.bid_csv)
    ask = _load_rates_from_csv(args.ask_csv)
    bid_a, ask_a = _align_bid_ask(bid, ask)
    mid = _mid_from_bid_ask(bid_a, ask_a)
    _write_mid_csv(args.out, mid)

    print(f"Wrote mid OHLC to: {args.out}")
    print(f"Bars: {len(mid)} (aligned from bid={len(bid)} ask={len(ask)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
