"""
strategies.py — VWAP Crossover + filtres RSI / EMA / Volume
"""
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  INDICATEURS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df.index.date
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['pv'] = tp * df['Volume']

    df['cum_pv']  = df.groupby('date')['pv'].cumsum()
    df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
    df['VWAP']    = df['cum_pv'] / df['cum_vol']

    df['variance']     = ((tp - df['VWAP']) ** 2) * df['Volume']
    df['cum_variance'] = df.groupby('date')['variance'].cumsum()
    df['std']          = np.sqrt(df['cum_variance'] / df['cum_vol'])

    df['VWAP_plus_1']  = df['VWAP'] + df['std']
    df['VWAP_plus_2']  = df['VWAP'] + df['std'] * 2
    df['VWAP_minus_1'] = df['VWAP'] - df['std']
    df['VWAP_minus_2'] = df['VWAP'] - df['std'] * 2

    # ATR (14)
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # EMA 50 / 200
    df['EMA50']  = df['Close'].ewm(span=50,  adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Volume moyen 20 bougies
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  DÉTECTION DE SIGNAL — avec filtres
# ═══════════════════════════════════════════════════════════════════════════════

def detect_signal(df: pd.DataFrame,
                  rsi_min: float = 40.0,
                  rsi_max: float = 60.0,
                  require_ema_trend: bool = True,
                  require_volume: bool = True):
    required = ['VWAP', 'RSI', 'EMA200', 'Vol_MA20', 'ATR']
    for col in required:
        if col not in df.columns:
            return None, f"Colonne manquante : {col}"
    """
    Signal VWAP crossover confirmé par RSI, EMA200 et volume.

    Paramètres ajustables (l'auto-critique peut les modifier) :
      rsi_min / rsi_max   : zone neutre RSI (évite les extrêmes)
      require_ema_trend   : impose que le prix soit du bon côté de l'EMA200
      require_volume      : impose un volume > moyenne 20 bougies
    """
    if len(df) < 3:
        return None, ""

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ── Croisement VWAP (signal brut) ────────────────────────────────────────
    cross_long  = last['Close'] > last['VWAP'] and prev['Close'] <= prev['VWAP']
    cross_short = last['Close'] < last['VWAP'] and prev['Close'] >= prev['VWAP']

    if not (cross_long or cross_short):
        return None, ""

    direction = "LONG" if cross_long else "SHORT"
    reasons_ok  = [f"Croisement VWAP {direction} @ {last['Close']:.2f}"]
    reasons_ko  = []

    # ── Filtre RSI ────────────────────────────────────────────────────────────
    rsi = last.get('RSI', np.nan)
    if not np.isnan(rsi):
        if rsi_min <= rsi <= rsi_max:
            reasons_ok.append(f"RSI neutre ({rsi:.1f})")
        else:
            reasons_ko.append(f"RSI hors zone ({rsi:.1f} — zone [{rsi_min},{rsi_max}])")
    
    # ── Filtre EMA200 (tendance) ──────────────────────────────────────────────
    ema200 = last.get('EMA200', np.nan)
    if require_ema_trend and not np.isnan(ema200):
        if direction == "LONG"  and last['Close'] > ema200:
            reasons_ok.append(f"Prix > EMA200 ({ema200:.2f}) ✓ tendance haussière")
        elif direction == "SHORT" and last['Close'] < ema200:
            reasons_ok.append(f"Prix < EMA200 ({ema200:.2f}) ✓ tendance baissière")
        else:
            reasons_ko.append(f"Prix à contre-tendance EMA200 ({ema200:.2f})")

    # ── Filtre Volume ─────────────────────────────────────────────────────────
    vol_ma = last.get('Vol_MA20', np.nan)
    if require_volume and not np.isnan(vol_ma) and vol_ma > 0:
        if last['Volume'] >= vol_ma:
            reasons_ok.append(f"Volume élevé ({last['Volume']:.0f} >= moy {vol_ma:.0f})")
        else:
            reasons_ko.append(f"Volume faible ({last['Volume']:.0f} < moy {vol_ma:.0f})")

    # ── Décision finale ───────────────────────────────────────────────────────
    if reasons_ko:
        reason_str = " | ".join(reasons_ok + [f"❌ {r}" for r in reasons_ko])
        return None, f"Signal rejeté ({direction}) — {reason_str}"

    return direction, " | ".join(reasons_ok)


# ═══════════════════════════════════════════════════════════════════════════════
#  SL / TP
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(price: float, signal: str, atr: float,
                    sl_mult: float = 1.0, tp_mult: float = 2.0):
    """
    SL/TP basés sur l'ATR. Les multiplicateurs sont ajustables
    par l'auto-critique (ex: élargir le SL si trop de SL prématurés).
    """
    if signal == "LONG":
        sl = price - atr * sl_mult
        tp = price + atr * tp_mult
    else:
        sl = price + atr * sl_mult
        tp = price - atr * tp_mult
    return round(sl, 4), round(tp, 4)