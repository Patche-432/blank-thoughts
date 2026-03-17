import logging
import os
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("web")

# ── Trading bot runtime (MT5) ────────────────────────────────────────────────

_bot_lock = threading.Lock()
_bot = None
_bot_thread = None


def _bot_enabled() -> bool:
    raw = (os.getenv("BOT_ENABLE", "0") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _live_trading_allowed() -> bool:
    raw = (os.getenv("BOT_ALLOW_LIVE", "0") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _bot_state():
    with _bot_lock:
        return _bot, _bot_thread


def _run_bot_in_thread(bot) -> None:
    global _bot_thread
    try:
        bot.run()
    except Exception as exc:
        log.exception("Bot crashed: %s", exc)
    finally:
        with _bot_lock:
            _bot_thread = None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/bot/status", methods=["GET"])
def bot_status():
    enabled = _bot_enabled()
    bot, thread = _bot_state()
    running = bool(thread and thread.is_alive()) if thread else False
    last_signal = None
    last_decision = None

    if bot is not None:
        sig = bot.last_signal()
        if sig is not None:
            last_signal = {
                "symbol": sig.symbol,
                "volume": sig.volume,
                "direction": sig.direction,
                "sl_price": sig.sl_price,
                "tp_price": sig.tp_price,
                "comment": sig.comment,
            }
        dec = bot.last_decision()
        if dec is not None:
            last_decision = {
                "approve": dec.approve,
                "action": dec.action,
                "reason": dec.reason,
                "meta": dec.meta,
            }

    payload = {
        "enabled": enabled,
        "running": running,
        "thread_alive": bool(thread.is_alive()) if thread else False,
        "bot": None,
        "last_signal": last_signal,
        "last_decision": last_decision,
        "safety": {
            "default_dry_run": True,
            "allow_live": _live_trading_allowed(),
        },
    }

    if bot is not None:
        payload["bot"] = {
            "symbol": bot.symbol,
            "volume": bot.volume,
            "timeframe": bot.timeframe,
            "bar_count": bot.bar_count,
            "poll_secs": bot.poll_secs,
            "dry_run": bot.dry_run,
            "max_pos": bot.max_pos,
        }

    return jsonify(payload)


@app.route("/bot/start", methods=["POST"])
def bot_start():
    global _bot, _bot_thread
    if not _bot_enabled():
        return jsonify({"error": "Bot is disabled. Set BOT_ENABLE=1 to enable start/stop endpoints."}), 403

    bot, thread = _bot_state()
    if thread and thread.is_alive():
        return jsonify({"error": "Bot is already running."}), 409

    data = request.get_json(silent=True) or {}
    symbol = str(data.get("symbol", "EURUSD"))
    volume = float(data.get("volume", 0.01))
    poll_secs = float(data.get("poll_secs", 10.0))
    bar_count = int(data.get("bar_count", 250))
    dry_run = data.get("dry_run", None)
    max_pos = int(data.get("max_pos", 1))
    strategy_params = data.get("strategy", {}) or {}
    ai_mode = str(data.get("ai_mode", "") or os.getenv("BOT_AI_MODE", "heuristic"))
    max_spread_points = data.get("max_spread_points", None)
    ai_model = data.get("ai_model", None)
    if not isinstance(strategy_params, dict):
        return jsonify({"error": "strategy must be an object/dict"}), 400

    if dry_run is False and not _live_trading_allowed():
        return (
            jsonify({"error": "Live trading is not allowed. Set BOT_ALLOW_LIVE=1 (and understand the risk) to enable."}),
            403,
        )

    from core.bot import Bot
    from core.strategy import Strategy

    try:
        strat = Strategy(symbol=symbol, volume=volume, **strategy_params)
        new_bot = Bot(
            symbol=symbol,
            volume=volume,
            timeframe=Bot.TIMEFRAME_M5,
            bar_count=bar_count,
            poll_secs=poll_secs,
            dry_run=dry_run,
            max_pos=max_pos,
            ai_mode=ai_mode,
            max_spread_points=max_spread_points,
            ai_model=ai_model,
            strategy=strat,
        )
    except Exception as exc:
        return jsonify({"error": f"Invalid bot/strategy settings: {exc}"}), 400

    t = threading.Thread(target=_run_bot_in_thread, args=(new_bot,), daemon=True)
    with _bot_lock:
        _bot = new_bot
        _bot_thread = t
    t.start()

    return jsonify({"ok": True, "running": True})


@app.route("/bot/stop", methods=["POST"])
def bot_stop():
    if not _bot_enabled():
        return jsonify({"error": "Bot is disabled. Set BOT_ENABLE=1 to enable start/stop endpoints."}), 403

    bot, thread = _bot_state()
    if bot is None or not (thread and thread.is_alive()):
        return jsonify({"ok": True, "running": False})

    bot.stop()
    return jsonify({"ok": True, "running": False})


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
