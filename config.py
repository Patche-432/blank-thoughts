# Local configuration.
#
# NOTE: Keep real credentials out of source control. If you store secrets here,
# add `config.py` to your `.gitignore`.
#
# MetaTrader5 initialize() kwargs:
# - path: full path to terminal64.exe (optional)
# - login/password/server: optional (many setups auto-login via the terminal UI)

MT5_CONFIG = {
    "timeout": 10_000,
    "portable": False,
    "path": None,
    "login": None,
    "password": None,
    "server": None,
}

