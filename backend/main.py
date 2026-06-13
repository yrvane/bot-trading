from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
from datetime import datetime
import asyncio
import threading
import time

from data_fetcher import get_historical_data_yfinance, get_current_price_yfinance
from strategies import calculate_vwap, detect_signal, calculate_sl_tp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

TRADES_FILE = os.path.join(BASE_DIR, "data", "bot_trades.csv")
os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)

open_positions = {}
last_signal_state = {}

# ==================== GESTION DES TRADES ====================
def load_trades():
    try:
        df = pd.read_csv(TRADES_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df.to_dict('records')
    except:
        return []

def save_trade(trade):
    try:
        df = pd.read_csv(TRADES_FILE)
    except:
        df = pd.DataFrame(columns=['date', 'symbol', 'type', 'entry_price', 'exit_price', 'pnl', 'status', 'sl', 'tp'])
    new_trade = pd.DataFrame([trade])
    df = pd.concat([df, new_trade], ignore_index=True)
    df.to_csv(TRADES_FILE, index=False)

def update_trade_exit(symbol, exit_price, pnl):
    trades = load_trades()
    for t in trades:
        if t.get('symbol') == symbol and t.get('status') == 'open':
            t['exit_price'] = exit_price
            t['pnl'] = pnl
            t['status'] = 'closed'
            break
    pd.DataFrame(trades).to_csv(TRADES_FILE, index=False)

# ==================== WEBSOCKET ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ==================== BOT AUTO-TRADING ====================
async def check_and_execute_trades(symbol):
    global open_positions, last_signal_state

    if symbol in open_positions:
        position = open_positions[symbol]
        current_price = get_current_price_yfinance(symbol)

        if current_price:
            exit_price = None
            pnl = 0
            reason = None

            if position['type'] == 'LONG':
                if current_price <= position['sl']:
                    exit_price, reason = position['sl'], "SL"
                    pnl = (exit_price - position['entry_price']) / position['entry_price'] * 100
                elif current_price >= position['tp']:
                    exit_price, reason = position['tp'], "TP"
                    pnl = (exit_price - position['entry_price']) / position['entry_price'] * 100
            else:
                if current_price >= position['sl']:
                    exit_price, reason = position['sl'], "SL"
                    pnl = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                elif current_price <= position['tp']:
                    exit_price, reason = position['tp'], "TP"
                    pnl = (position['entry_price'] - exit_price) / position['entry_price'] * 100

            if exit_price:
                update_trade_exit(symbol, exit_price, pnl)
                del open_positions[symbol]
                await manager.broadcast({
                    "type": "trade_closed",
                    "symbol": symbol,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": reason
                })
        return

    df = get_historical_data_yfinance(symbol, period="60d", interval="1h")
    if df is not None and len(df) > 10:
        df = calculate_vwap(df)
        signal, reason = detect_signal(df)

        last_signal = last_signal_state.get(symbol)

        if signal and signal != last_signal:
            last_signal_state[symbol] = signal
            current_price = get_current_price_yfinance(symbol)
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else 10
            sl, tp = calculate_sl_tp(current_price, signal, atr)

            trade = {
                'date': datetime.now(),
                'symbol': symbol,
                'type': signal,
                'entry_price': current_price,
                'exit_price': 0,
                'pnl': 0,
                'status': 'open',
                'sl': sl,
                'tp': tp
            }
            save_trade(trade)
            open_positions[symbol] = trade

            await manager.broadcast({"type": "new_trade", "trade": trade})

        elif not signal:
            last_signal_state[symbol] = None

# ==================== ROUTES API ====================
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/price/{symbol}")
async def get_price(symbol: str):
    price = get_current_price_yfinance(symbol)
    return {"symbol": symbol, "price": price, "timestamp": datetime.now().isoformat()}

@app.get("/api/history/{symbol}")
async def get_history(symbol: str, bars: int = 300, interval: str = "1h"):
    period_map = {"5m": "5d", "15m": "10d", "1h": "60d", "4h": "60d", "1d": "180d"}
    df = get_historical_data_yfinance(symbol, period=period_map.get(interval, "60d"), interval=interval)

    if df is None or df.empty:
        return {"error": f"No data for {symbol}"}

    df = calculate_vwap(df)
    df = df.tail(bars)

    return {
        "dates":        df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        "open":         df['Open'].tolist(),
        "high":         df['High'].tolist(),
        "low":          df['Low'].tolist(),
        "close":        df['Close'].tolist(),
        "vwap":         df['VWAP'].tolist(),
        "vwap_plus_1":  df['VWAP_plus_1'].tolist(),
        "vwap_minus_1": df['VWAP_minus_1'].tolist(),
    }

@app.get("/api/positions")
async def get_positions():
    return list(open_positions.values())

@app.get("/api/trades")
async def get_trades():
    return load_trades()

# ==================== WEBSOCKET ====================
@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket)
    try:
        while True:
            price = get_current_price_yfinance(symbol)
            if price:
                await websocket.send_json({
                    "type": "price",
                    "price": price,
                    "timestamp": datetime.now().isoformat()
                })

            await check_and_execute_trades(symbol)

            if symbol in open_positions:
                await websocket.send_json({
                    "type": "position_update",
                    "position": open_positions[symbol]
                })

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    import webbrowser

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")
        print("✅ Navigateur ouvert sur http://127.0.0.1:8000")

    print("🚀 Bot Auto-Trading démarré (yfinance uniquement)")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)