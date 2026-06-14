"""
main.py — Bot Auto-Trading VWAP
Intègre : filtres RSI/EMA/Volume, auto-critique, sentiment news, alertes Telegram
"""
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

from data_fetcher    import get_historical_data_yfinance, get_current_price_yfinance
from strategies      import calculate_vwap, detect_signal, calculate_sl_tp
from trade_analyzer  import load_trades as analyzer_load, analyze, suggest_adjustments, load_config, save_config
from News_sentiment  import get_sentiment, sentiment_allows_trade
from Telegram_alerts import (alert_new_trade, alert_trade_closed,
                              alert_signal_blocked, alert_config_updated,
                              alert_bot_started)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

TRADES_FILE = os.path.join(BASE_DIR, "data", "bot_trades.csv")
os.makedirs(os.path.dirname(TRADES_FILE), exist_ok=True)

open_positions    = {}
last_signal_state = {}

strategy_config = load_config()
print(f"⚙️  Config stratégie : RSI [{strategy_config['rsi_min']}-{strategy_config['rsi_max']}] "
      f"| SL {strategy_config['sl_mult']}x | TP {strategy_config['tp_mult']}x")


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTION DES TRADES
# ═══════════════════════════════════════════════════════════════════════════════

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
        df = pd.DataFrame(columns=[
            'date','symbol','type','entry_price','exit_price','pnl','status','sl','tp'
        ])
    df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    df.to_csv(TRADES_FILE, index=False)

def update_trade_exit(symbol, exit_price, pnl):
    trades = load_trades()
    for t in trades:
        if t.get('symbol') == symbol and t.get('status') == 'open':
            t['exit_price'] = exit_price
            t['pnl']        = pnl
            t['status']     = 'closed'
            break
    pd.DataFrame(trades).to_csv(TRADES_FILE, index=False)

def run_auto_critique():
    """Auto-critique après chaque trade fermé — met à jour la config et alerte Telegram."""
    global strategy_config
    try:
        df = analyzer_load(path=TRADES_FILE)
        if len(df) < 5:
            return
        analysis        = analyze(df)
        new_cfg, reasons = suggest_adjustments(analysis, strategy_config)
        if new_cfg != strategy_config:
            strategy_config = new_cfg
            save_config(new_cfg)
            print(f"🔄 Config mise à jour : {' | '.join(reasons)}")
            alert_config_updated(reasons, new_cfg)
    except Exception as e:
        print(f"⚠️  Auto-critique error : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, message: dict):
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except:
                pass

manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════════════════════
#  BOT AUTO-TRADING
# ═══════════════════════════════════════════════════════════════════════════════

async def check_and_execute_trades(symbol: str):
    global open_positions, last_signal_state, strategy_config

    # ── Position ouverte → surveiller SL/TP ──────────────────────────────────
    if symbol in open_positions:
        position      = open_positions[symbol]
        current_price = get_current_price_yfinance(symbol)
        if not current_price:
            return

        exit_price = reason = None
        pnl = 0

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
                "type": "trade_closed", "symbol": symbol,
                "exit_price": exit_price, "pnl": pnl, "reason": reason
            })

            # Telegram + auto-critique
            alert_trade_closed(symbol, exit_price, pnl, reason)
            threading.Thread(target=run_auto_critique, daemon=True).start()
        return

    # ── Pas de position → chercher un signal ─────────────────────────────────
    df = get_historical_data_yfinance(symbol, period="60d", interval="1h")
    if df is None or len(df) < 20:
        return

    df = calculate_vwap(df)
    df = df.dropna(subset=['VWAP', 'ATR', 'RSI', 'EMA200'])
    if len(df) < 20:
        return

    signal, reason = detect_signal(
        df,
        rsi_min=strategy_config.get("rsi_min", 40),
        rsi_max=strategy_config.get("rsi_max", 60),
        require_ema_trend=strategy_config.get("require_ema_trend", True),
        require_volume=strategy_config.get("require_volume", True),
    )

    last_signal = last_signal_state.get(symbol)
    if not signal or signal == last_signal:
        if not signal:
            last_signal_state[symbol] = None
        return

    # ── Filtre sentiment news ─────────────────────────────────────────────────
    sentiment = get_sentiment(symbol)
    if not sentiment_allows_trade(sentiment, signal):
        print(f"📰 Signal {signal} {symbol} bloqué — {sentiment['signal']} ({sentiment['confidence']}%)")
        alert_signal_blocked(
            symbol, signal,
            sentiment['signal'], sentiment['confidence'], sentiment['summary']
        )
        return

    # ── Ouvrir la position ────────────────────────────────────────────────────
    last_signal_state[symbol] = signal
    current_price = get_current_price_yfinance(symbol)
    if not current_price:
        return

    atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else 10
    sl, tp = calculate_sl_tp(
        current_price, signal, atr,
        sl_mult=strategy_config.get("sl_mult", 1.0),
        tp_mult=strategy_config.get("tp_mult", 2.0),
    )

    trade = {
        'date':        datetime.now(),
        'symbol':      symbol,
        'type':        signal,
        'entry_price': current_price,
        'exit_price':  0,
        'pnl':         0,
        'status':      'open',
        'sl':          sl,
        'tp':          tp,
    }
    save_trade(trade)
    open_positions[symbol] = trade

    await manager.broadcast({
        "type":      "new_trade",
        "trade":     trade,
        "sentiment": sentiment.get("summary", ""),
        "reason":    reason,
    })

    # Telegram
    alert_new_trade(trade, sentiment=sentiment.get("summary", ""), reason=reason)
    print(f"🟢 {signal} {symbol} @ {current_price:.2f} | SL {sl:.2f} | TP {tp:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES API
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/price/{symbol}")
async def get_price(symbol: str):
    price = get_current_price_yfinance(symbol)
    return {"symbol": symbol, "price": price, "timestamp": datetime.now().isoformat()}

@app.get("/api/history/{symbol}")
async def get_history(symbol: str, bars: int = 300, interval: str = "1h"):
    period_map = {"5m":"5d","15m":"10d","1h":"60d","4h":"60d","1d":"180d"}
    df = get_historical_data_yfinance(symbol, period=period_map.get(interval,"60d"), interval=interval)
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

@app.get("/api/trades")
async def get_trades():
    return load_trades()

@app.get("/api/positions")
async def get_positions():
    return list(open_positions.values())

@app.get("/api/sentiment/{symbol}")
async def get_sentiment_route(symbol: str):
    return get_sentiment(symbol)

@app.get("/api/config")
async def get_config():
    return strategy_config

@app.post("/api/analyze")
async def trigger_analysis():
    run_auto_critique()
    return {"status": "ok", "config": strategy_config}


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket)
    try:
        while True:
            price = get_current_price_yfinance(symbol)
            if price:
                await websocket.send_json({
                    "type":      "price",
                    "price":     price,
                    "timestamp": datetime.now().isoformat()
                })

            await check_and_execute_trades(symbol)

            if symbol in open_positions:
                await websocket.send_json({
                    "type":     "position_update",
                    "position": open_positions[symbol]
                })

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    import webbrowser

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    print("🚀 Bot démarré — VWAP + RSI/EMA/Volume + Auto-critique + Sentiment + Telegram")
    alert_bot_started()
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)