"""
MT5 connection wrapper used by bot.py.
Provides a context manager and manual connect/disconnect interface.
Reads credentials from config.py → MT5_CONFIG.
"""

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)


class MT5ConfigError(ValueError):
    """Raised when config.py is missing or contains invalid values."""


class MT5ConnectionError(RuntimeError):
    """Raised when the MT5 terminal is unreachable or login fails."""


class MT5NotConnectedError(RuntimeError):
    """Raised when an MT5 data method is called before connecting."""


class MT5Connection:
    """
    Thin wrapper around the MetaTrader5 library for use inside the bot loop.
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        """
        Args:
            cfg: Optional config dict. Defaults to MT5_CONFIG from config.py.

        Raises:
            MT5ConfigError: If config.py is missing or MT5_CONFIG is not a dict.
            ImportError:    If the MetaTrader5 package is not installed.
        """
        # Defer MT5 import so the error message is clear
        try:
            import MetaTrader5  # noqa: F401 — just checking it exists
        except ImportError:
            raise ImportError(
                "MetaTrader5 package not found.\n"
                "  Windows: py -m pip install MetaTrader5\n"
                "  Mac/Wine: pip install MetaTrader5"
            )

        if cfg is not None:
            self._cfg = cfg
        else:
            try:
                from config import MT5_CONFIG

                self._cfg = MT5_CONFIG
            except ImportError:
                raise MT5ConfigError(
                    "config.py not found. Create it in the project root "
                    "and define MT5_CONFIG = { login, password, server, ... }"
                )

        if not isinstance(self._cfg, dict):
            raise MT5ConfigError(f"MT5_CONFIG must be a dict, got {type(self._cfg).__name__}.")

        self._connected = False
        self._stop_event = threading.Event()

    # ── Connection management ─────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialise the MT5 terminal and log in.

        Returns:
            True on success, False on failure (error is logged with details).
        """
        import MetaTrader5 as mt5

        log.info("Connecting to MT5 …")

        kwargs = self._build_init_kwargs()
        if kwargs is None:
            return False  # already logged in _build_init_kwargs

        try:
            if not mt5.initialize(**kwargs):
                code, msg = mt5.last_error()
                log.error("mt5.initialize() failed [%d] %s", code, msg)
                return False
        except Exception as exc:
            log.error("Unexpected error during mt5.initialize(): %s", exc)
            return False

        self._connected = True
        self._log_post_connect_info(mt5)
        return True

    def disconnect(self) -> None:
        """Shut down the MT5 connection. Safe to call even if not connected."""
        if not self._connected:
            return

        import MetaTrader5 as mt5

        try:
            mt5.shutdown()
            log.info("MT5 disconnected.")
        except Exception as exc:
            log.error("Error during mt5.shutdown(): %s", exc)
        finally:
            self._connected = False  # always clear, even if shutdown raised

    def stop(self) -> None:
        """Signal any waiting poll loop to wake up and exit (thread-safe)."""
        self._stop_event.set()

    def is_connected(self) -> bool:
        return self._connected

    # ── Data accessors ────────────────────────────────────────────────────────

    def account_info(self):
        """
        Return the MT5 account info namedtuple, or None on failure.

        Raises:
            MT5NotConnectedError: If called before connect().
        """
        self._require_connected()
        import MetaTrader5 as mt5

        try:
            result = mt5.account_info()
            if result is None:
                code, msg = mt5.last_error()
                log.error("account_info() returned None [%d] %s", code, msg)
            return result
        except Exception as exc:
            log.error("account_info() raised: %s", exc)
            return None

    def terminal_info(self):
        """
        Return the MT5 terminal info namedtuple, or None on failure.

        Raises:
            MT5NotConnectedError: If called before connect().
        """
        self._require_connected()
        import MetaTrader5 as mt5

        try:
            result = mt5.terminal_info()
            if result is None:
                code, msg = mt5.last_error()
                log.error("terminal_info() returned None [%d] %s", code, msg)
            return result
        except Exception as exc:
            log.error("terminal_info() raised: %s", exc)
            return None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MT5Connection":
        if not self.connect():
            raise MT5ConnectionError("MT5 failed to connect — check the logs above for details.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        # Log any exception that escaped the with-block, but don't suppress it
        if exc_type and exc_type is not KeyboardInterrupt:
            log.error("Exception inside MT5Connection context: [%s] %s", exc_type.__name__, exc_val)
        return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_init_kwargs(self) -> Optional[dict]:
        """
        Build the kwargs dict for mt5.initialize() from self._cfg.
        Returns None and logs an error if any value is the wrong type.
        """
        try:
            kwargs: dict = {
                "timeout": int(self._cfg.get("timeout", 10_000)),
                "portable": bool(self._cfg.get("portable", False)),
            }
            # Only include optional fields when they have a value
            if self._cfg.get("path"):
                kwargs["path"] = str(self._cfg["path"])
            if self._cfg.get("login"):
                kwargs["login"] = int(self._cfg["login"])
            if self._cfg.get("password"):
                kwargs["password"] = str(self._cfg["password"])
            if self._cfg.get("server"):
                kwargs["server"] = str(self._cfg["server"])
            return kwargs
        except (TypeError, ValueError) as exc:
            log.error("Bad value in MT5_CONFIG: %s — check config.py for typos.", exc)
            return None

    def _log_post_connect_info(self, mt5) -> None:
        """Log build version and account summary immediately after connecting."""
        try:
            log.info("Connected ✓  build=%s", mt5.version())
            acct = mt5.account_info()
            if acct:
                log.info("Account: %s | %s | %.2f %s", acct.login, acct.server, acct.balance, acct.currency)
            else:
                log.warning("Connected but could not retrieve account info.")
        except Exception as exc:
            log.warning("Post-connect info error (non-fatal): %s", exc)

    def _require_connected(self) -> None:
        """Raise MT5NotConnectedError if not yet connected."""
        if not self._connected:
            raise MT5NotConnectedError(
                "Not connected to MT5. Call connect() or use the context manager first."
            )
