import pandas as pd
import numpy as np

def calculate_vwap(df):
    """Calcule le VWAP et ses bandes"""
    df = df.copy()
    df['date'] = df.index.date
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['pv'] = typical_price * df['Volume']
    
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
    df['VWAP'] = df['cum_pv'] / df['cum_vol']
    
    df['variance'] = ((typical_price - df['VWAP']) ** 2) * df['Volume']
    df['cum_variance'] = df.groupby('date')['variance'].cumsum()
    df['std'] = np.sqrt(df['cum_variance'] / df['cum_vol'])
    
    df['VWAP_plus_1'] = df['VWAP'] + df['std']
    df['VWAP_plus_2'] = df['VWAP'] + (df['std'] * 2)
    df['VWAP_minus_1'] = df['VWAP'] - df['std']
    df['VWAP_minus_2'] = df['VWAP'] - (df['std'] * 2)
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    return df

def detect_signal(df):
    """Détecte un signal d'entrée"""
    if len(df) < 2:
        return None, ""
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Condition LONG
    if last_row['Close'] > last_row['VWAP'] and prev_row['Close'] <= prev_row['VWAP']:
        return "LONG", f"Prix ({last_row['Close']:.2f}) passe au-dessus du VWAP"
    
    # Condition SHORT
    elif last_row['Close'] < last_row['VWAP'] and prev_row['Close'] >= prev_row['VWAP']:
        return "SHORT", f"Prix ({last_row['Close']:.2f}) passe en dessous du VWAP"
    
    return None, ""

def calculate_sl_tp(price, signal, atr):
    """Calcule Stop Loss et Take Profit (RR 2/1)"""
    if signal == "LONG":
        sl = price - atr
        tp = price + (atr * 2)
    else:
        sl = price + atr
        tp = price - (atr * 2)
    return sl, tp