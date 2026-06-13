import yfinance as yf

for sym, yf_sym in [("US100", "^NDX"), ("US500", "^GSPC"), ("XAUUSD", "GC=F")]:
    t = yf.Ticker(yf_sym)
    info = t.fast_info
    print(f"{sym} → last: {info.get('last_price')} | prev_close: {info.get('previous_close')}")