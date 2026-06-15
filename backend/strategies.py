"""
strategies.py — VWAP Daily + DR/IDR + Rejets sur niveaux clés
              Setup A : post-breakout retest → follow orderflow
              Setup B : range → rejet bande VWAP / DR/IDR → target opposé
              Sessions : 9h-12h et 16h-18h UTC
"""
import pandas as pd
import numpy as np
from datetime import time as dtime


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSIONS (UTC) — fenêtres de trading
# ═══════════════════════════════════════════════════════════════════════════════

SESSIONS = {
    "morning":   (dtime(8, 0),  dtime(12, 0)),
    "afternoon": (dtime(15, 0), dtime(18, 0)),
}

def get_active_session(dt) -> str:
    t = dt.time() if hasattr(dt, 'time') else dt
    for name, (start, end) in SESSIONS.items():
        if start <= t <= end:
            return name
    return "off"

def is_session_active(dt) -> bool:
    return get_active_session(dt) != "off"


# ═══════════════════════════════════════════════════════════════════════════════
#  VWAP DAILY + ÉCARTS-TYPES
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
    df['VWAP_plus_3']  = df['VWAP'] + df['std'] * 3
    df['VWAP_minus_1'] = df['VWAP'] - df['std']
    df['VWAP_minus_2'] = df['VWAP'] - df['std'] * 2
    df['VWAP_minus_3'] = df['VWAP'] - df['std'] * 3

    # ATR 14
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

    # Volume moyen 20
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # Garder RSI/EMA pour compatibilité auto-critique
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI']    = 100 - (100 / (1 + rs))
    df['EMA50']  = df['Close'].ewm(span=50,  adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  DR / IDR
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_idr(df: pd.DataFrame,
                  idr_start: dtime = dtime(7, 0),
                  idr_end:   dtime = dtime(9, 0)) -> pd.DataFrame:
    """IDR = range 07h-09h UTC (ouverture London)"""
    df = df.copy()
    for col in ['IDR_High', 'IDR_Low', 'IDR_Mid', 'DR_High', 'DR_Low']:
        df[col] = np.nan

    for date in df.index.normalize().unique():
        day_mask = df.index.normalize() == date
        idr_mask = day_mask & (df.index.time >= idr_start) & (df.index.time <= idr_end)
        if idr_mask.sum() < 1:
            continue

        idr_high = df.loc[idr_mask, 'High'].max()
        idr_low  = df.loc[idr_mask, 'Low'].min()
        dr_high  = df.loc[day_mask, 'High'].max()
        dr_low   = df.loc[day_mask, 'Low'].min()

        df.loc[day_mask, 'IDR_High'] = idr_high
        df.loc[day_mask, 'IDR_Low']  = idr_low
        df.loc[day_mask, 'IDR_Mid']  = (idr_high + idr_low) / 2
        df.loc[day_mask, 'DR_High']  = dr_high
        df.loc[day_mask, 'DR_Low']   = dr_low

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  FIBONACCI SUR DERNIÈRE IMPULSION
# ═══════════════════════════════════════════════════════════════════════════════

def find_last_impulse(df: pd.DataFrame, lookback: int = 40) -> dict | None:
    """Identifie la dernière impulsion et calcule les niveaux Fibo 0.5 / 0.618"""
    window = df.tail(lookback)
    if len(window) < 5:
        return None

    highs    = window['High'].values
    lows     = window['Low'].values
    idx_high = int(np.argmax(highs))
    idx_low  = int(np.argmin(lows))

    swing_high = highs[idx_high]
    swing_low  = lows[idx_low]
    fib_range  = swing_high - swing_low

    if fib_range <= 0:
        return None

    if idx_high > idx_low:          # impulsion haussière
        direction = "UP"
        fib_50    = swing_high - fib_range * 0.50
        fib_618   = swing_high - fib_range * 0.618
    else:                           # impulsion baissière
        direction = "DOWN"
        fib_50    = swing_low + fib_range * 0.50
        fib_618   = swing_low + fib_range * 0.618

    return {
        "direction":  direction,
        "swing_high": round(swing_high, 4),
        "swing_low":  round(swing_low,  4),
        "fib_50":     round(fib_50,  4),
        "fib_618":    round(fib_618, 4),
        "range":      round(fib_range, 4),
    }


def price_near_fibo(price: float, fib: dict, tolerance_pct: float = 0.20) -> tuple[bool, str]:
    if not fib:
        return False, "Pas d'impulsion détectée"

    tol = fib['range'] * tolerance_pct

    if abs(price - fib['fib_618']) <= tol:
        return True, f"Fibo 0.618 @ {fib['fib_618']:.2f}"
    if abs(price - fib['fib_50']) <= tol:
        return True, f"Fibo 0.50 @ {fib['fib_50']:.2f}"

    return False, f"Hors Fibo (50: {fib['fib_50']:.2f} | 61.8: {fib['fib_618']:.2f})"


# ═══════════════════════════════════════════════════════════════════════════════
#  CASSURE DU RANGE M15
# ═══════════════════════════════════════════════════════════════════════════════

def detect_range_breakout(df: pd.DataFrame, lookback: int = 20) -> tuple[str | None, float, float]:
    """Détecte la cassure du dernier high/low sur N bougies"""
    if len(df) < lookback + 2:
        return None, 0, 0

    range_window = df.iloc[-(lookback + 1):-1]
    current      = df.iloc[-1]
    range_high   = range_window['High'].max()
    range_low    = range_window['Low'].min()

    if current['Close'] > range_high:
        return "LONG", range_high, range_low
    if current['Close'] < range_low:
        return "SHORT", range_high, range_low

    return None, range_high, range_low


# ═══════════════════════════════════════════════════════════════════════════════
#  REJET SUR VWAP
# ═══════════════════════════════════════════════════════════════════════════════

def detect_vwap_rejection(df: pd.DataFrame, direction: str) -> tuple[bool, str]:
    """
    LONG  : mèche basse touche la VWAP, close au-dessus
    SHORT : mèche haute touche la VWAP, close en-dessous
    """
    if len(df) < 2:
        return False, ""

    last = df.iloc[-1]
    vwap = last.get('VWAP', np.nan)
    if pd.isna(vwap):
        return False, "VWAP non disponible"

    atr = last.get('ATR', np.nan)
    tol = atr * 0.5 if not pd.isna(atr) else 0

    if direction == "LONG":
        if last['Low'] <= vwap + tol and last['Close'] > vwap:
            return True, f"Rejet haussier VWAP @ {vwap:.2f}"
        return False, f"Pas de rejet haussier VWAP @ {vwap:.2f}"
    else:
        if last['High'] >= vwap - tol and last['Close'] < vwap:
            return True, f"Rejet baissier VWAP @ {vwap:.2f}"
        return False, f"Pas de rejet baissier VWAP @ {vwap:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFLUENCE DR/IDR
# ═══════════════════════════════════════════════════════════════════════════════

def check_dr_idr_confluence(row: pd.Series, direction: str) -> tuple[bool, str]:
    price = row['Close']
    atr   = row.get('ATR', price * 0.001)
    tol   = atr * 1.5

    levels = {
        'IDR_High': row.get('IDR_High', np.nan),
        'IDR_Low':  row.get('IDR_Low',  np.nan),
        'IDR_Mid':  row.get('IDR_Mid',  np.nan),
        'DR_High':  row.get('DR_High',  np.nan),
        'DR_Low':   row.get('DR_Low',   np.nan),
    }

    for name, level in levels.items():
        if not np.isnan(level) and abs(price - level) <= tol:
            return True, f"Confluence {name} @ {level:.2f}"

    return False, "Pas de niveau DR/IDR proche"


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def _get_key_levels(row: pd.Series) -> dict[str, float]:
    """Extrait tous les niveaux clés disponibles depuis la bougie courante."""
    keys = ['VWAP', 'VWAP_plus_1', 'VWAP_plus_2', 'VWAP_plus_3',
            'VWAP_minus_1', 'VWAP_minus_2', 'VWAP_minus_3',
            'DR_High', 'DR_Low', 'IDR_High', 'IDR_Low', 'IDR_Mid']
    return {k: float(row[k]) for k in keys if k in row.index and not pd.isna(row[k])}


def _next_level(levels: dict, price: float, direction: str,
                min_dist: float = 0.0) -> tuple[float | None, str | None]:
    """Retourne le niveau clé le plus proche dans la direction du trade."""
    candidates = [
        (val, name) for name, val in levels.items()
        if (direction == "LONG"  and val > price + min_dist) or
           (direction == "SHORT" and val < price - min_dist)
    ]
    if not candidates:
        return None, None
    return (min(candidates) if direction == "LONG" else max(candidates))


def _is_rejection(bar: pd.Series, level: float, direction: str, tol: float) -> bool:
    """
    Bougie de rejet sur un niveau :
      LONG  : mèche basse touche le niveau, close au-dessus, corps haussier ou neutre.
      SHORT : mèche haute touche le niveau, close en-dessous, corps baissier ou neutre.
    """
    if direction == "LONG":
        return (bar['Low']  <= level + tol and
                bar['Close'] > level and
                bar['Close'] >= bar['Open'])
    else:
        return (bar['High'] >= level - tol and
                bar['Close'] < level and
                bar['Close'] <= bar['Open'])


def detect_signal(df: pd.DataFrame,
                  require_macro_trend: bool = True,
                  # Legacy params conservés pour compatibilité main.py
                  min_score:              float = 4.0,
                  require_session:        bool  = True,
                  require_fibo:           bool  = True,
                  require_vwap_rejection: bool  = True,
                  require_dr_idr:         bool  = True,
                  require_breakout:       bool  = True,
                  rsi_min:           float = 35.0,
                  rsi_max:           float = 65.0,
                  require_ema_trend: bool  = False,
                  require_volume:    bool  = False,
                  **kwargs) -> tuple[str | None, str, dict]:
    """
    Deux setups de rejet sur niveaux VWAP / DR / IDR :

    Setup A — Post-breakout retest :
      Prix a cassé un niveau dans les 25 dernières bougies, est revenu le
      retester et forme une bougie de rejet → on suit l'orderflow du breakout.

    Setup B — Range bounce :
      Prix rejette une bande VWAP ou un niveau DR/IDR avec un RR ≥ 1.5 vers
      le niveau opposé le plus proche.

    Retourne (direction | None, reason, {"tp": float, "sl": float}).
    """
    if 'VWAP' not in df.columns or len(df) < 25:
        return None, "Données insuffisantes", {}

    last  = df.iloc[-1]
    price = last['Close']
    atr   = last.get('ATR', price * 0.001)
    tol   = atr * 0.5

    if pd.isna(last.get('VWAP', np.nan)):
        return None, "VWAP NaN", {}

    # ── 1. Session (bloquant) ─────────────────────────────────────────────────
    session = get_active_session(df.index[-1])
    if session == "off":
        return None, f"Hors session ({df.index[-1].strftime('%H:%M')} UTC)", {}

    # ── 2. Filtre macro EMA (bloquant) ────────────────────────────────────────
    ema50  = last.get('EMA50_macro',  last.get('EMA50',  np.nan))
    ema200 = last.get('EMA200_macro', last.get('EMA200', np.nan))
    macro_src = "1h" if 'EMA50_macro' in last.index else "native"
    macro_ok = {"LONG": True, "SHORT": True}
    if require_macro_trend and not pd.isna(ema50) and not pd.isna(ema200):
        macro_ok["LONG"]  = ema50 >= ema200
        macro_ok["SHORT"] = ema50 <= ema200

    # ── Niveaux clés disponibles ──────────────────────────────────────────────
    levels  = _get_key_levels(last)
    history = df.iloc[max(0, len(df) - 26):-1]   # 25 bougies hors courante

    # ══════════════════════════════════════════════════════════════════════════
    #  SETUP A — Post-breakout retest
    #  Prix a clairement cassé un niveau récemment, reteste et rejette.
    # ══════════════════════════════════════════════════════════════════════════
    for level_name, level_val in sorted(levels.items(), key=lambda x: abs(x[1] - price)):

        # LONG : breakout haussier récent + retest du niveau par le bas
        if (macro_ok["LONG"] and
                history['Close'].max() > level_val + atr * 0.8 and   # fut au-dessus
                price <= level_val + atr * 1.2 and                   # est revenu near
                _is_rejection(last, level_val, "LONG", tol)):

            tp, tp_name = _next_level(levels, price, "LONG", min_dist=atr * 0.8)
            if tp is None:
                continue
            sl  = round(level_val - atr * 0.5, 4)
            rr  = (tp - price) / max(price - sl, 1e-9)
            if rr < 1.0:
                continue
            return ("LONG",
                    f"Setup A LONG | Retest {level_name}@{level_val:.2f} "
                    f"→ {tp_name}@{tp:.2f} (RR {rr:.1f}) | {session} [{macro_src}]",
                    {"tp": tp, "sl": sl})

        # SHORT : breakout baissier récent + retest du niveau par le haut
        if (macro_ok["SHORT"] and
                history['Close'].min() < level_val - atr * 0.8 and   # fut en-dessous
                price >= level_val - atr * 1.2 and                   # est revenu near
                _is_rejection(last, level_val, "SHORT", tol)):

            tp, tp_name = _next_level(levels, price, "SHORT", min_dist=atr * 0.8)
            if tp is None:
                continue
            sl  = round(level_val + atr * 0.5, 4)
            rr  = (price - tp) / max(sl - price, 1e-9)
            if rr < 1.0:
                continue
            return ("SHORT",
                    f"Setup A SHORT | Retest {level_name}@{level_val:.2f} "
                    f"→ {tp_name}@{tp:.2f} (RR {rr:.1f}) | {session} [{macro_src}]",
                    {"tp": tp, "sl": sl})

    # ══════════════════════════════════════════════════════════════════════════
    #  SETUP B — Range bounce (rejet bande VWAP / DR/IDR → target opposé)
    # ══════════════════════════════════════════════════════════════════════════
    for level_name, level_val in sorted(levels.items(), key=lambda x: abs(x[1] - price)):

        # LONG depuis support
        if (macro_ok["LONG"] and
                _is_rejection(last, level_val, "LONG", tol)):

            tp, tp_name = _next_level(levels, price, "LONG", min_dist=atr * 1.0)
            if tp is None:
                continue
            sl  = round(level_val - atr * 0.5, 4)
            rr  = (tp - price) / max(price - sl, 1e-9)
            if rr < 1.5:
                continue
            return ("LONG",
                    f"Setup B LONG | Rejet {level_name}@{level_val:.2f} "
                    f"→ {tp_name}@{tp:.2f} (RR {rr:.1f}) | {session} [{macro_src}]",
                    {"tp": tp, "sl": sl})

        # SHORT depuis résistance
        if (macro_ok["SHORT"] and
                _is_rejection(last, level_val, "SHORT", tol)):

            tp, tp_name = _next_level(levels, price, "SHORT", min_dist=atr * 1.0)
            if tp is None:
                continue
            sl  = round(level_val + atr * 0.5, 4)
            rr  = (price - tp) / max(sl - price, 1e-9)
            if rr < 1.5:
                continue
            return ("SHORT",
                    f"Setup B SHORT | Rejet {level_name}@{level_val:.2f} "
                    f"→ {tp_name}@{tp:.2f} (RR {rr:.1f}) | {session} [{macro_src}]",
                    {"tp": tp, "sl": sl})

    return None, f"Aucun setup A/B | {session}", {}


# ═══════════════════════════════════════════════════════════════════════════════
#  SL / TP
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sl_tp(price: float, signal: str, atr: float,
                    sl_mult: float = 1.0, tp_mult: float = 2.0,
                    swing_high: float = None, swing_low: float = None,
                    idr_high: float = None, idr_low: float = None,
                    tp_level: float = None, sl_level: float = None):
    """
    Si tp_level / sl_level sont fournis (depuis detect_signal), on les utilise.
    Sinon fallback : SL sur swing, TP à tp_mult × ATR.
    """
    if signal == "LONG":
        sl = sl_level if sl_level else (swing_low  - atr * 0.1 if swing_low  else price - atr * sl_mult)
        tp = tp_level if tp_level else price + atr * tp_mult
    else:
        sl = sl_level if sl_level else (swing_high + atr * 0.1 if swing_high else price + atr * sl_mult)
        tp = tp_level if tp_level else price - atr * tp_mult

    return round(sl, 4), round(tp, 4)