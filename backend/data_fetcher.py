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

    # Période minimale suffisante pour chaque intervalle
    period_map = {
        "5m":  "5d",
        "15m": "10d",
        "1h":  "60d",
        "4h":  "60d",
        "1d":  "180d",
    }
    safe_period = period_map.get(interval, "60d")

    df = ticker.history(period=safe_period, interval=interval)

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