import yfinance as yf
import pandas as pd

SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "US100":  "NQ=F",
    "US500":  "ES=F",
    "SP500":  "ES=F",
    "GOLD":   "GC=F",
}

def get_yf_symbol(symbol):
    return SYMBOL_MAP.get(symbol.upper(), symbol)

def get_historical_data_yfinance(symbol, period="7d", interval="1h"):
    yf_symbol = get_yf_symbol(symbol)
    ticker = yf.Ticker(yf_symbol)

    # Plafonds yfinance par intervalle
    MAX_PERIOD_DAYS = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
        "60m": 730, "1h": 730, "90m": 60,
    }

    def _parse_days(p: str) -> int | None:
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2]) * 30
        return None

    req_days = _parse_days(period)
    max_days = MAX_PERIOD_DAYS.get(interval)
    if req_days and max_days and req_days > max_days:
        period = f"{max_days}d"

    df = ticker.history(period=period, interval=interval)

    if df.empty:
        # Fallback daily si l'intervalle ne donne rien (marché fermé, etc.)
        df = ticker.history(period="60d", interval="1d")
    if df.empty:
        return None

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"])

    # Strip timezone
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    df = df.set_index("time")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0

    return df[["Open", "High", "Low", "Close", "Volume"]]

def get_macro_ema(symbol: str, span_fast: int = 50, span_slow: int = 200) -> pd.DataFrame | None:
    """Retourne EMA50/EMA200 calculées sur le 1h, indexées en UTC sans tz."""
    yf_symbol = get_yf_symbol(symbol)
    df = yf.Ticker(yf_symbol).history(period="60d", interval="1h")
    if df.empty:
        return None

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.set_index("time")

    df["EMA50_macro"]  = df["Close"].ewm(span=span_fast, adjust=False).mean()
    df["EMA200_macro"] = df["Close"].ewm(span=span_slow, adjust=False).mean()
    return df[["EMA50_macro", "EMA200_macro"]]


def get_current_price_yfinance(symbol):
    yf_symbol = get_yf_symbol(symbol)
    ticker = yf.Ticker(yf_symbol)

    # Essai 1 : dernière minute
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        return float(data["Close"].iloc[-1])

    # Essai 2 : dernier jour connu
    data = ticker.history(period="5d", interval="1d")
    if not data.empty:
        return float(data["Close"].iloc[-1])

    return None
