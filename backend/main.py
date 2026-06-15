"""
main.py — Bot Auto-Trading
Stratégie : VWAP + Fibo 0.5/0.618 + Rejet VWAP + DR/IDR + Sessions + M15
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone, timedelta
import asyncio
import threading
import time

from data_fetcher    import get_historical_data_yfinance, get_current_price_yfinance
from strategies      import (calculate_vwap, calculate_idr, detect_signal,
                              calculate_sl_tp, find_last_impulse)
from trade_analyzer  import (load_trades as analyzer_load, analyze,
                              suggest_adjustments, load_config, save_config)
from News_sentiment  import get_sentiment, get_news, sentiment_allows_trade
from Telegram_alerts import (alert_new_trade, alert_trade_closed,
                              alert_signal_blocked, alert_config_updated,
                              alert_daily_summary, alert_bot_started)

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

INITIAL_PORTFOLIO        = 50000.0
DAILY_SUMMARY_TIME_UTC   = os.getenv("DAILY_SUMMARY_TIME_UTC", "21:05")
DAILY_SUMMARY_STATE_FILE = os.path.join(BASE_DIR, "data", "daily_summary_state.json")
OPEN_POSITIONS_FILE      = os.path.join(BASE_DIR, "data", "open_positions.json")
LAST_SIGNAL_STATE_FILE   = os.path.join(BASE_DIR, "data", "last_signal_state.json")

def _save_open_positions():
    with open(OPEN_POSITIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(open_positions, f, indent=2, default=str)

def _load_open_positions() -> dict:
    try:
        with open(OPEN_POSITIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        pass
    # Reconstruire depuis le CSV si le JSON n'existe pas
    try:
        import pandas as pd
        df = pd.read_csv(TRADES_FILE)
        open_rows = df[df['status'] == 'open']
        result = {}
        for _, row in open_rows.iterrows():
            result[row['symbol']] = {
                'date':        str(row['date']),
                'symbol':      row['symbol'],
                'type':        row['type'],
                'entry_price': float(row['entry_price']),
                'exit_price':  float(row['exit_price']),
                'pnl':         float(row['pnl']),
                'status':      row['status'],
                'sl':          float(row['sl']),
                'tp':          float(row['tp']),
            }
        if result:
            print(f"📂 {len(result)} position(s) restaurée(s) depuis le CSV")
        return result
    except:
        return {}

def _save_last_signal_state():
    with open(LAST_SIGNAL_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(last_signal_state, f, indent=2)

def _load_last_signal_state() -> dict:
    try:
        with open(LAST_SIGNAL_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

open_positions    = _load_open_positions()
last_signal_state = _load_last_signal_state()
strategy_config   = load_config()

print(f"⚙️  Config : SL {strategy_config['sl_mult']}x | TP {strategy_config['tp_mult']}x")


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTION DES TRADES
# ═══════════════════════════════════════════════════════════════════════════════

def load_trades():
    try:
        df = pd.read_csv(TRADES_FILE)
        if df.empty:
            return []
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
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
    if not trades:
        return
    now_str = datetime.now().isoformat()
    for t in trades:
        # Préserver la date comme string, ou mettre now() si absente/NaT
        d = t.get('date')
        t['date'] = now_str if (d is None or str(d) in ('NaT', '', 'nan')) else str(d)
        if t.get('symbol') == symbol and t.get('status') == 'open':
            t['exit_price'] = exit_price
            t['pnl']        = pnl
            t['status']     = 'closed'
    pd.DataFrame(trades).to_csv(TRADES_FILE, index=False)

def run_auto_critique():
    global strategy_config
    try:
        df = analyzer_load(path=TRADES_FILE)
        if len(df) < 5:
            return
        analysis         = analyze(df)
        new_cfg, reasons = suggest_adjustments(analysis, strategy_config)
        if new_cfg != strategy_config:
            strategy_config = new_cfg
            save_config(new_cfg)
            print(f"🔄 Config mise à jour : {' | '.join(reasons)}")
            alert_config_updated(reasons, new_cfg)
    except Exception as e:
        print(f"⚠️  Auto-critique error : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DAILY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def load_daily_summary_state() -> dict:
    try:
        with open(DAILY_SUMMARY_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"last_sent_date": None}

def save_daily_summary_state(state: dict):
    os.makedirs(os.path.dirname(DAILY_SUMMARY_STATE_FILE), exist_ok=True)
    with open(DAILY_SUMMARY_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def build_daily_summary() -> dict:
    today_utc = datetime.now(timezone.utc).date()
    trades = load_trades()
    today_trades  = [t for t in trades if pd.to_datetime(t.get("date"), errors="coerce").date() == today_utc]
    closed_today  = [t for t in today_trades if t.get("status") == "closed"]
    open_today    = [t for t in today_trades if t.get("status") == "open"]
    df_today      = pd.DataFrame(closed_today)

    if closed_today:
        analysis = analyze(df_today)
        _, reasons = suggest_adjustments(analysis, strategy_config)
    else:
        analysis = {}
        reasons  = ["Aucun trade fermé aujourd'hui"]

    top_symbol = "—"
    if today_trades:
        df_t = pd.DataFrame(today_trades)
        sp   = df_t.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        top_symbol = str(sp.index[0]) if not sp.empty else "—"

    total_pnl = float(df_today["pnl"].sum()) if not df_today.empty else 0.0

    return {
        "date_label":    today_utc.strftime("%d/%m/%Y"),
        "closed_trades": len(closed_today),
        "open_trades":   len(open_today),
        "wins":          analysis.get("wins", 0),
        "losses":        analysis.get("losses", 0),
        "win_rate":      analysis.get("win_rate", 0.0),
        "profit_factor": analysis.get("profit_factor", 0),
        "avg_win":       analysis.get("avg_win", 0),
        "avg_loss":      analysis.get("avg_loss", 0),
        "max_streak":    analysis.get("max_streak", 0),
        "best_hour":     f"{analysis['best_hour']}h" if analysis.get("best_hour") is not None else "—",
        "worst_hour":    f"{analysis['worst_hour']}h" if analysis.get("worst_hour") is not None else "—",
        "top_symbol":    top_symbol,
        "total_pnl_pct": total_pnl,
        "improvements":  reasons,
    }

def should_send_daily_summary(now_utc=None) -> bool:
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        h, m = [int(x) for x in DAILY_SUMMARY_TIME_UTC.split(":", 1)]
    except:
        h, m = 21, 5
    state = load_daily_summary_state()
    if state.get("last_sent_date") == now_utc.date().isoformat():
        return False
    return now_utc.hour > h or (now_utc.hour == h and now_utc.minute >= m)

def daily_summary_worker():
    while True:
        try:
            if should_send_daily_summary():
                summary = build_daily_summary()
                if alert_daily_summary(summary):
                    save_daily_summary_state({"last_sent_date": datetime.now(timezone.utc).date().isoformat()})
        except Exception as e:
            print(f"⚠️  Daily summary error : {e}")
        time.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

def build_portfolio_summary(initial_capital: float = INITIAL_PORTFOLIO) -> dict:
    trades = load_trades()
    closed = []
    for t in trades:
        if t.get("status") != "closed":
            continue
        dt = pd.to_datetime(t.get("date"), errors="coerce")
        # Si la date est absente, utiliser la dernière date connue + 1 min (ou now())
        if pd.isna(dt):
            fallback = closed[-1][0] + timedelta(minutes=1) if closed else datetime.now()
            dt = pd.Timestamp(fallback)
        try:
            pnl = float(t.get("pnl", 0) or 0)
        except:
            pnl = 0.0
        closed.append((dt.to_pydatetime(), t, pnl))

    closed.sort(key=lambda x: x[0])
    equity = float(initial_capital)
    wins   = 0
    pnl_curve      = []
    win_rate_curve = []
    latest_updates = []

    for i, (dt, t, pnl) in enumerate(closed):
        if not pnl_curve:
            start = (dt - timedelta(minutes=1)).isoformat()
            pnl_curve.append({"time": start, "value": 0.0})
            win_rate_curve.append({"time": start, "value": 0.0})
        equity *= 1 + (pnl / 100.0)
        if pnl > 0:
            wins += 1
        cum_pnl = round(((equity / initial_capital) - 1) * 100, 2)
        wr      = round((wins / (i + 1)) * 100, 2)
        pnl_curve.append({"time": dt.isoformat(), "value": cum_pnl})
        win_rate_curve.append({"time": dt.isoformat(), "value": wr})
        latest_updates.append({
            "date":   dt.isoformat(),
            "symbol": t.get("symbol", ""),
            "type":   t.get("type", ""),
            "pnl":    round(pnl, 2),
            "equity": round(equity, 2),
            "reason": t.get("status", "closed"),
        })

    n = len(closed)
    return {
        "initial_capital":  round(initial_capital, 2),
        "current_capital":  round(equity, 2),
        "total_return_pct": round(((equity / initial_capital) - 1) * 100, 2) if initial_capital else 0,
        "closed_trades":    n,
        "win_rate":         round((wins / n) * 100, 2) if n else 0.0,
        "pnl_curve":        pnl_curve,
        "win_rate_curve":   win_rate_curve,
        "trade_updates":    latest_updates[-25:][::-1],
        "generated_at":     datetime.now().isoformat(),
    }


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

    async def broadcast(self, msg: dict):
        for conn in self.active_connections:
            try:
                await conn.send_json(msg)
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
            _save_open_positions()
            await manager.broadcast({
                "type": "trade_closed", "symbol": symbol,
                "exit_price": exit_price, "pnl": pnl, "reason": reason
            })
            alert_trade_closed(symbol, exit_price, pnl, reason)
            threading.Thread(target=run_auto_critique, daemon=True).start()
        return

    # ── Chercher un signal en M15 ─────────────────────────────────────────────
    df = get_historical_data_yfinance(symbol, period="5d", interval="15m")
    if df is None or len(df) < 50:
        return

    df = calculate_vwap(df)
    df = calculate_idr(df)
    df = df.dropna(subset=['VWAP', 'ATR'])
    if len(df) < 25:
        return

    signal, reason, _meta = detect_signal(df)

    last_signal = last_signal_state.get(symbol)
    if not signal or signal == last_signal:
        if not signal:
            last_signal_state[symbol] = None
        return

    # ── Filtre sentiment ──────────────────────────────────────────────────────
    sentiment = get_sentiment(symbol)
    if not sentiment_allows_trade(sentiment, signal):
        print(f"📰 {signal} {symbol} bloqué — {sentiment['signal']} ({sentiment['confidence']}%)")
        last_signal_state[symbol] = signal
        _save_last_signal_state()
        return

    # ── Ouvrir la position ────────────────────────────────────────────────────
    last_signal_state[symbol] = signal
    _save_last_signal_state()
    current_price = get_current_price_yfinance(symbol)
    if not current_price:
        return

    atr  = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else 10
    fib  = find_last_impulse(df, lookback=40)
    last = df.iloc[-1]

    def safe_val(v):
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    sl, tp = calculate_sl_tp(
        current_price, signal, atr,
        sl_mult=strategy_config.get("sl_mult", 1.0),
        tp_mult=strategy_config.get("tp_mult", 2.0),
        swing_high=fib['swing_high'] if fib else None,
        swing_low=fib['swing_low']   if fib else None,
        idr_high=safe_val(last.get('IDR_High')),
        idr_low=safe_val(last.get('IDR_Low')),
        tp_level=_meta.get("tp"),
        sl_level=_meta.get("sl"),
    )

    trade = {
        'date':        datetime.now().isoformat(),
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
    _save_open_positions()

    await manager.broadcast({
        "type": "new_trade", "trade": trade,
        "sentiment": sentiment.get("summary", ""),
        "reason": reason,
    })
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
    df = calculate_idr(df)
    df = df.tail(bars)

    def safe(col):
        if col not in df.columns:
            return []
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                for v in df[col].tolist()]

    return {
        "dates":        df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        "open":         df['Open'].tolist(),
        "high":         df['High'].tolist(),
        "low":          df['Low'].tolist(),
        "close":        df['Close'].tolist(),
        "vwap":         safe('VWAP'),
        "vwap_plus_1":  safe('VWAP_plus_1'),
        "vwap_minus_1": safe('VWAP_minus_1'),
        "idr_high":     safe('IDR_High'),
        "idr_low":      safe('IDR_Low'),
        "dr_high":      safe('DR_High'),
        "dr_low":       safe('DR_Low'),
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

@app.get("/api/news/{symbol}")
async def get_news_route(symbol: str, limit: int = 8):
    return {
        "symbol":       symbol,
        "sentiment":    get_sentiment(symbol),
        "articles":     get_news(symbol, max_articles=limit),
        "generated_at": datetime.now().isoformat(),
    }

@app.get("/api/portfolio")
async def get_portfolio_summary():
    return build_portfolio_summary()

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

    print("🚀 Bot — VWAP + Fibo 0.5/0.618 + Rejet VWAP + DR/IDR + Sessions + Telegram")
    alert_bot_started()
    threading.Thread(target=daily_summary_worker, daemon=True).start()
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)