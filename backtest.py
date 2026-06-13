import pandas as pd
import numpy as np
from strategy_complete import add_indicators, generate_signals

def run_backtest(df, rr_ratio=1, risk_percent=1):
    """
    Exécute un backtest avec take profit et stop loss
    rr_ratio = ratio risque/récompense (1 = RR 1/1)
    """
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
        
        # Entrée en position
        if position is None and signal == 1:
            position = 'long'
            entry_price = current_price
            entry_date = current_date
            atr = df['ATR_14'].iloc[i]
            
            # Stop Loss (1 ATR) et Take Profit (RR 1/1)
            stop_loss = entry_price - atr
            take_profit = entry_price + atr
            
        # Sortie de position
        elif position == 'long':
            # Stop hit
            if current_price <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': stop_loss,
                    'pnl_percent': pnl,
                    'type': 'stop_loss'
                })
                position = None
            # Take profit hit
            elif current_price >= take_profit:
                pnl = (take_profit - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': take_profit,
                    'pnl_percent': pnl,
                    'type': 'take_profit'
                })
                position = None
    
    # Convertir en DataFrame
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        print("❌ Aucun trade effectué")
        return trades_df
    
    # Statistiques
    winning_trades = trades_df[trades_df['pnl_percent'] > 0]
    losing_trades = trades_df[trades_df['pnl_percent'] < 0]
    
    print("\n" + "="*50)
    print("📊 RÉSULTATS DU BACKTEST (RR 1/1)")
    print("="*50)
    print(f"Nombre total de trades : {len(trades_df)}")
    print(f"Trades gagnants : {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
    print(f"Trades perdants : {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
    print(f"Profit moyen par trade : {trades_df['pnl_percent'].mean():.2f}%")
    print(f"Profit total : {trades_df['pnl_percent'].sum():.2f}%")
    print(f"Plus gros gain : {trades_df['pnl_percent'].max():.2f}%")
    print(f"Plus grosse perte : {trades_df['pnl_percent'].min():.2f}%")
    
    # Drawdown maximum
    cumulative = trades_df['pnl_percent'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()
    print(f"Drawdown maximum : {max_drawdown:.2f}%")
    
    # Win rate et expectancy
    win_rate = len(winning_trades) / len(trades_df) * 100
    avg_win = winning_trades['pnl_percent'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_percent'].mean() if len(losing_trades) > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
    print(f"Espérance de gain par trade : {expectancy:.2f}%")
    
    return trades_df

def plot_results(df, trades_df):
    """Trace les résultats"""
    import matplotlib.pyplot as plt
    
    if len(trades_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Graphique 1 : Prix et signaux
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], label='Prix SP500', linewidth=1, alpha=0.7)
    
    # Ajouter les points d'entrée
    for _, trade in trades_df.iterrows():
        if trade['type'] == 'take_profit':
            ax1.scatter(trade['entry_date'], trade['entry_price'], 
                       color='green', marker='^', s=100, zorder=5)
        else:
            ax1.scatter(trade['entry_date'], trade['entry_price'], 
                       color='red', marker='v', s=100, zorder=5)
    
    ax1.set_title('Prix SP500 avec signaux d\'entrée')
    ax1.set_ylabel('Prix')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2 : Courbe de capital
    ax2 = axes[1]
    trades_df['cumulative_pnl'] = trades_df['pnl_percent'].cumsum()
    ax2.plot(trades_df['exit_date'], trades_df['cumulative_pnl'], 
             linewidth=2, color='blue')
    ax2.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                     alpha=0.3, color='green')
    ax2.set_title('Courbe de capital cumulative')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Profit cumulé (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Charger les données
    print("📥 Chargement des données SP500...")
    df = pd.read_csv('US500_H1.csv', index_col=0, parse_dates=True)
    
    print("🔧 Calcul des indicateurs et signaux...")
    df = add_indicators(df)
    df = generate_signals(df)
    
    print(f"📈 Période : {df.index[0]} → {df.index[-1]}")
    print(f"💹 Nombre de bougies : {len(df)}")
    
    # Lancer le backtest
    trades = run_backtest(df)
    
    # Afficher les résultats graphiques
    if len(trades) > 0:
        print("\n📈 Génération des graphiques...")
        plot_results(df, trades)
        
        # Afficher les 5 derniers trades
        print("\n📋 Derniers trades :")
        print(trades[['entry_date', 'exit_date', 'type', 'pnl_percent']].tail())
    else:
        print("\n💡 Aucun trade généré. Vérifie les conditions de la stratégie.")