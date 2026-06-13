import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration compte démo (à remplacer par tes identifiants)
MT5_ACCOUNT = int(os.getenv("MT5_ACCOUNT", 5051636226))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "NdC@Te04")
MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")

SYMBOL_MAP = {
    "SP500": "US500",
    "GOLD": "XAUUSD"
}

def connect_mt5():
    """Connecte à MT5"""
    if not mt5.initialize():
        return False
    if not mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER):
        return False
    return True

def disconnect_mt5():
    mt5.shutdown()

def get_current_price(symbol):
    """Récupère le prix actuel"""
    mt5_symbol = SYMBOL_MAP.get(symbol, symbol)
    if not connect_mt5():
        return None
    tick = mt5.symbol_info_tick(mt5_symbol)
    disconnect_mt5()
    if tick:
        return tick.ask if tick.ask else tick.bid
    return None

def get_historical_data(symbol, timeframe=mt5.TIMEFRAME_H1, n_bars=500):
    """Récupère les données historiques"""
    mt5_symbol = SYMBOL_MAP.get(symbol, symbol)
    if not connect_mt5():
        return None
    rates = mt5.copy_rates_from_pos(mt5_symbol, timeframe, 0, n_bars)
    disconnect_mt5()
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df

def place_order(symbol, order_type, volume, price, sl, tp):
    """Place un ordre sur MT5"""
    mt5_symbol = SYMBOL_MAP.get(symbol, symbol)
    if not connect_mt5():
        return False, "MT5 non connecté"
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": f"Bot_Trading_{datetime.now().strftime('%Y%m%d')}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    disconnect_mt5()
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"Erreur: {result.retcode}"
    return True, f"Ordre exécuté | Ticket: {result.order}"