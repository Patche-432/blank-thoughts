"""Trading bot package (MT5)."""

from .bot import Bot, BotError
from .connection import MT5Connection, MT5ConnectionError
from .guardian import Decision, TradeGuardian
from .strategy import (
    Strategy,
    BollingerBusterADXStrategy,
    Signal,
    BUY,
    SELL,
    HOLD,
    EXIT_LONG,
    EXIT_SHORT,
)

__all__ = [
    "Bot",
    "BotError",
    "MT5Connection",
    "MT5ConnectionError",
    "Decision",
    "TradeGuardian",
    "Strategy",
    "BollingerBusterADXStrategy",
    "Signal",
    "BUY",
    "SELL",
    "HOLD",
    "EXIT_LONG",
    "EXIT_SHORT",
]
