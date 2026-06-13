import pandas as pd
import numpy as np

# ==================== VWAP ====================
def calculate_vwap(df, session_start_hour=0, session_end_hour=24):
    df = df.copy()
    df['date'] = df.index.date
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['typical_price'] = typical_price
    df['pv'] = typical_price * df['Volume']
    
    df['cum_pv'] = df.groupby('date')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
    df['VWAP'] = df['cum_pv'] / df['cum_vol']
    
    df['variance'] = ((typical_price - df['VWAP']) ** 2) * df['Volume']
    df['cum_variance'] = df.groupby('date')['variance'].cumsum()
    df['std'] = np.sqrt(df['cum_variance'] / df['cum_vol'])
    
    df['VWAP_plus_1'] = df['VWAP'] + df['std']
    df['VWAP_plus_2'] = df['VWAP'] + (df['std'] * 2)
    df['VWAP_plus_3'] = df['VWAP'] + (df['std'] * 3)
    df['VWAP_minus_1'] = df['VWAP'] - df['std']
    df['VWAP_minus_2'] = df['VWAP'] - (df['std'] * 2)
    df['VWAP_minus_3'] = df['VWAP'] - (df['std'] * 3)
    
    return df

def detect_vwap_rejection(df, lookback=3):
    df = df.copy()
    df['vwap_rejection'] = None
    for i in range(lookback, len(df)):
        if (df['Low'].iloc[i] <= df['VWAP'].iloc[i] <= df['Close'].iloc[i] and
            df['Close'].iloc[i] > df['VWAP'].iloc[i]):
            df.loc[df.index[i], 'vwap_rejection'] = 'bullish'
        elif (df['High'].iloc[i] >= df['VWAP'].iloc[i] >= df['Close'].iloc[i] and
              df['Close'].iloc[i] < df['VWAP'].iloc[i]):
            df.loc[df.index[i], 'vwap_rejection'] = 'bearish'
    return df

def check_asian_session_end(df):
    df = df.copy()
    # Correction : utiliser .hour sur l'index directement
    hours = df.index.hour
    df['asian_session_end'] = (hours >= 7) & (hours <= 9)
    return df

# ==================== DR/IDR ====================
def calculate_dr_idr(df):
    df = df.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    
    def get_dr_levels(group):
        # Exclure les colonnes de groupement pour éviter le warning
        rdr_mask = (group['hour'] == 9) | ((group['hour'] == 10) & (group['minute'] <= 30))
        rdr_data = group[rdr_mask]
        if len(rdr_data) > 0:
            rdr_high = rdr_data['High'].max()
            rdr_low = rdr_data['Low'].min()
        else:
            rdr_high, rdr_low = np.nan, np.nan
        return pd.Series({
            'DR_high': rdr_high,
            'DR_low': rdr_low,
            'DR_mid': (rdr_high + rdr_low) / 2 if not np.isnan(rdr_high) else np.nan
        })
    
    # Correction du warning : utiliser include_groups=False
    dr_levels = df.groupby('date', group_keys=False).apply(get_dr_levels, include_groups=False)
    df = df.merge(dr_levels, left_on='date', right_index=True, how='left')
    return df

def detect_dr_rejection(df, lookback=2):
    df = df.copy()
    df['dr_rejection'] = None
    for i in range(1, len(df)):
        if not pd.isna(df['DR_low'].iloc[i]):
            if (df['Low'].iloc[i] <= df['DR_low'].iloc[i] <= df['Close'].iloc[i] and
                df['Close'].iloc[i] > df['DR_low'].iloc[i]):
                df.loc[df.index[i], 'dr_rejection'] = 'bullish'
        
        if not pd.isna(df['DR_high'].iloc[i]):
            if (df['High'].iloc[i] >= df['DR_high'].iloc[i] >= df['Close'].iloc[i] and
                df['Close'].iloc[i] < df['DR_high'].iloc[i]):
                df.loc[df.index[i], 'dr_rejection'] = 'bearish'
    return df

def detect_dr_breakout(df, lookback=2):
    df = df.copy()
    df['dr_breakout'] = None
    for i in range(1, len(df)):
        if not pd.isna(df['DR_high'].iloc[i]) and not pd.isna(df['DR_high'].iloc[i-1]):
            if (df['Close'].iloc[i] > df['DR_high'].iloc[i] and
                df['Close'].iloc[i-1] <= df['DR_high'].iloc[i-1]):
                df.loc[df.index[i], 'dr_breakout'] = 'bullish'
        
        if not pd.isna(df['DR_low'].iloc[i]) and not pd.isna(df['DR_low'].iloc[i-1]):
            if (df['Close'].iloc[i] < df['DR_low'].iloc[i] and
                df['Close'].iloc[i-1] >= df['DR_low'].iloc[i-1]):
                df.loc[df.index[i], 'dr_breakout'] = 'bearish'
    return df

# ==================== STRATÉGIE ====================
def apply_strategy(df):
    df = df.copy()
    print("📊 Calcul du VWAP...")
    df = calculate_vwap(df)
    print("📊 Calcul des niveaux DR...")
    df = calculate_dr_idr(df)
    print("📊 Détection des rejets...")
    df = detect_vwap_rejection(df)
    df = detect_dr_rejection(df)
    df = detect_dr_breakout(df)
    df = check_asian_session_end(df)
    
    df['signal'] = 0
    
    for i in range(1, len(df)):
        # CONDITIONS LONG (achat)
        if (df['Close'].iloc[i] > df['VWAP'].iloc[i] and
            df['asian_session_end'].iloc[i] and
            df['dr_rejection'].iloc[i] == 'bullish'):
            if df['vwap_rejection'].iloc[i] == 'bullish':
                df.loc[df.index[i], 'signal'] = 1
            elif df['vwap_rejection'].iloc[i] is None:
                df.loc[df.index[i], 'signal'] = 1
        
        # CONDITIONS SHORT (vente)
        elif (df['Close'].iloc[i] < df['VWAP'].iloc[i] and
              df['asian_session_end'].iloc[i] and
              df['dr_rejection'].iloc[i] == 'bearish'):
            if df['vwap_rejection'].iloc[i] == 'bearish':
                df.loc[df.index[i], 'signal'] = -1
            elif df['vwap_rejection'].iloc[i] is None:
                df.loc[df.index[i], 'signal'] = -1
    
    return df

def calculate_atr_based_sl_tp(df, atr_multiplier_sl=1, atr_multiplier_tp=2):
    df = df.copy()
    if 'ATR_14' not in df.columns:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
    df['sl_distance'] = df['ATR_14'] * atr_multiplier_sl
    df['tp_distance'] = df['ATR_14'] * atr_multiplier_tp
    return df

# ==================== BACKTEST ====================
def run_backtest(df):
    df = df.copy()
    trades = []
    position = None
    entry_price = 0
    entry_date = None
    stop_loss = 0
    take_profit = 0
    
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        signal = df['signal'].iloc[i]
        
        if position is None and signal != 0:
            if signal == 1:
                position = 'long'
            else:
                position = 'short'
            
            entry_price = current_price
            entry_date = current_date
            atr = df['ATR_14'].iloc[i]
            
            if position == 'long':
                stop_loss = entry_price - atr
                take_profit = entry_price + (atr * 2)
            else:
                stop_loss = entry_price + atr
                take_profit = entry_price - (atr * 2)
        
        elif position == 'long':
            if current_price <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * 100
                trades.append({'entry_date': entry_date, 'exit_date': current_date, 'direction': 'long', 'pnl_percent': pnl, 'type': 'stop_loss'})
                position = None
            elif current_price >= take_profit:
                pnl = (take_profit - entry_price) / entry_price * 100
                trades.append({'entry_date': entry_date, 'exit_date': current_date, 'direction': 'long', 'pnl_percent': pnl, 'type': 'take_profit'})
                position = None
        
        elif position == 'short':
            if current_price >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price * 100
                trades.append({'entry_date': entry_date, 'exit_date': current_date, 'direction': 'short', 'pnl_percent': pnl, 'type': 'stop_loss'})
                position = None
            elif current_price <= take_profit:
                pnl = (entry_price - take_profit) / entry_price * 100
                trades.append({'entry_date': entry_date, 'exit_date': current_date, 'direction': 'short', 'pnl_percent': pnl, 'type': 'take_profit'})
                position = None
    
    return pd.DataFrame(trades)

# ==================== MAIN ====================
if __name__ == "__main__":
    print("📥 Chargement des données SP500...")
    df = pd.read_csv('US500_H1.csv', index_col=0, parse_dates=True)
    
    print("🔧 Application de la stratégie...")
    df = apply_strategy(df)
    df = calculate_atr_based_sl_tp(df)
    
    print("🎯 Lancement du backtest (RR 2/1)...")
    trades = run_backtest(df)
    
    if len(trades) == 0:
        print("❌ Aucun trade effectué")
    else:
        winning = trades[trades['pnl_percent'] > 0]
        losing = trades[trades['pnl_percent'] < 0]
        
        print("\n" + "="*60)
        print("📊 RÉSULTATS STRATÉGIE VWAP + DR/IDR (RR 2/1)")
        print("="*60)
        print(f"Nombre total de trades : {len(trades)}")
        print(f"Trades gagnants : {len(winning)} ({len(winning)/len(trades)*100:.1f}%)")
        print(f"Trades perdants : {len(losing)} ({len(losing)/len(trades)*100:.1f}%)")
        print(f"Profit total : {trades['pnl_percent'].sum():.2f}%")
        print(f"Profit moyen : {trades['pnl_percent'].mean():.2f}%")
        print(f"Plus gros gain : {trades['pnl_percent'].max():.2f}%")
        print(f"Plus grosse perte : {trades['pnl_percent'].min():.2f}%")
        
        # Afficher les trades
        print("\n📋 Derniers trades :")
        print(trades.tail(10))