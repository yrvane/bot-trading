"""
strategies.py — VWAP Daily + DR/IDR/Asian + Rejets sur niveaux clés
              Setup A : post-breakout retest → follow orderflow
              Setup B : range → rejet bande VWAP / DR/IDR/Asian → target opposé
              Setup C : cassure / momentum, sans exigence de mèche de rejet
              Sessions : London 08h-13h & NY 16h-19h heure de Paris
"""
import pandas as pd
import numpy as np
from datetime import time as dtime
from zoneinfo import ZoneInfo


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSIONS — horaires exprimés en heure de Paris (gère automatiquement le
#  passage CET/CEST été-hiver, comme time(..., "Europe/Paris") en Pine)
#  Asian  : 01:30-08:00 Paris
#  London : 08:00-13:00 Paris
#  NY     : 16:00-19:00 Paris
# ═══════════════════════════════════════════════════════════════════════════════

PARIS_TZ = ZoneInfo("Europe/Paris")

SESSIONS = {
    "london": (dtime(8, 0),  dtime(13, 0)),
    "ny":     (dtime(16, 0), dtime(19, 0)),
}
ASIAN_SESSION = (dtime(1, 30), dtime(8, 0))
IDR_SESSION   = (dtime(1, 30), dtime(11, 0))


def _paris_time(dt) -> dtime:
    """Convertit un datetime (naïf = UTC, comme partout ailleurs dans ce module)
    en heure locale Paris."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(PARIS_TZ).time()


def get_active_session(dt) -> str:
    t = dt if isinstance(dt, dtime) else _paris_time(dt)
    for name, (start, end) in SESSIONS.items():
        if start <= t <= end:
            return name
    return "off"

def is_session_active(dt) -> bool:
    return get_active_session(dt) != "off"


# ═══════════════════════════════════════════════════════════════════════════════
#  VWAP DAILY + ÉCARTS-TYPES
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_vwap(df: pd.DataFrame, atr_len: int = 14,
                    range_bars: int = 8, range_mult: float = 1.5,
                    vol_len: int = 20) -> pd.DataFrame:
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

    # ATR (Wilder RMA du True Range, comme ta.atr() en Pine)
    prev_close = df['Close'].shift(1)
    true_range = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['ATR'] = true_range.ewm(alpha=1 / atr_len, adjust=False, min_periods=atr_len).mean()

    # Filtre range local (anti-chop) : range doit dépasser ATR * range_mult
    df['Local_Range']  = df['High'].rolling(range_bars).max() - df['Low'].rolling(range_bars).min()
    df['Not_In_Range']  = df['Local_Range'] > df['ATR'] * range_mult

    # Volume moyen — confirmation Setup C (breakout)
    df['Vol_MA20'] = df['Volume'].rolling(vol_len).mean()

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
                  idr_start:   dtime = IDR_SESSION[0],
                  idr_end:     dtime = IDR_SESSION[1],
                  asian_start: dtime = ASIAN_SESSION[0],
                  asian_end:   dtime = ASIAN_SESSION[1]) -> pd.DataFrame:
    """IDR = range 01:30-11:00 Paris, Asian = range 01:30-08:00 Paris,
    DR = range du jour complet (jour calendaire UTC, comme new_day en Pine)."""
    df = df.copy()
    for col in ['IDR_High', 'IDR_Low', 'IDR_Mid', 'DR_High', 'DR_Low',
                'Asian_High', 'Asian_Low', 'Asian_Mid']:
        df[col] = np.nan

    paris_time = pd.Index([_paris_time(ts) for ts in df.index])

    for date in df.index.normalize().unique():
        day_mask = df.index.normalize() == date

        dr_high = df.loc[day_mask, 'High'].max()
        dr_low  = df.loc[day_mask, 'Low'].min()
        df.loc[day_mask, 'DR_High'] = dr_high
        df.loc[day_mask, 'DR_Low']  = dr_low

        idr_mask = day_mask & (paris_time >= idr_start) & (paris_time <= idr_end)
        if idr_mask.sum() >= 1:
            idr_high = df.loc[idr_mask, 'High'].max()
            idr_low  = df.loc[idr_mask, 'Low'].min()
            df.loc[day_mask, 'IDR_High'] = idr_high
            df.loc[day_mask, 'IDR_Low']  = idr_low
            df.loc[day_mask, 'IDR_Mid']  = (idr_high + idr_low) / 2

        asian_mask = day_mask & (paris_time >= asian_start) & (paris_time <= asian_end)
        if asian_mask.sum() >= 1:
            asian_high = df.loc[asian_mask, 'High'].max()
            asian_low  = df.loc[asian_mask, 'Low'].min()
            df.loc[day_mask, 'Asian_High'] = asian_high
            df.loc[day_mask, 'Asian_Low']  = asian_low
            df.loc[day_mask, 'Asian_Mid']  = (asian_high + asian_low) / 2

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

def _rejection_ok(df: pd.DataFrame, level: float, direction: str, atr: float,
                  atr_tol: float, wick_ratio: float, body_ratio: float,
                  struct_bars: int) -> bool:
    """
    Bougie de rejet sur un niveau (équivalent rej_l/rej_s en Pine) :
      • mèche touche le niveau (± atr_tol), close de l'autre côté, corps
        dirigé dans le sens du rejet et suffisamment large (body_ratio) ;
      • mèche de rejet suffisamment marquée par rapport au corps (wick_ratio) ;
      • struct_ok : le niveau doit avoir tenu sur les struct_bars bougies
        précédentes (support/résistance pas déjà cassé juste avant).
    """
    last  = df.iloc[-1]
    body  = max(abs(last['Close'] - last['Open']), atr * 0.05)
    body_ok = abs(last['Close'] - last['Open']) >= atr * body_ratio
    prior_closes = df['Close'].iloc[-(struct_bars + 1):-1]

    if direction == "LONG":
        wick = min(last['Close'], last['Open']) - last['Low']
        struct_ok = bool((prior_closes >= level).all()) if len(prior_closes) else True
        return bool(last['Low'] <= level + atr * atr_tol and
                    last['Close'] > level and
                    last['Close'] >= last['Open'] and
                    wick >= body * wick_ratio and
                    struct_ok and body_ok)
    else:
        wick = last['High'] - max(last['Close'], last['Open'])
        struct_ok = bool((prior_closes <= level).all()) if len(prior_closes) else True
        return bool(last['High'] >= level - atr * atr_tol and
                    last['Close'] < level and
                    last['Close'] <= last['Open'] and
                    wick >= body * wick_ratio and
                    struct_ok and body_ok)


def _breakout_ok(df: pd.DataFrame, level: float, direction: str, atr: float,
                 c_body_mult: float, c_break_atr: float, struct_bars: int,
                 vol_ok: bool) -> bool:
    """
    Cassure / momentum net (équivalent brk_l/brk_s en Pine) : clôture nette
    au-delà du niveau, corps large, volume supérieur à la moyenne, et cassure
    fraîche (le niveau n'était pas déjà franchi sur les struct_bars précédents).
    """
    last = df.iloc[-1]
    body_ok = abs(last['Close'] - last['Open']) >= atr * c_body_mult
    prior_closes = df['Close'].iloc[-(struct_bars + 1):-1]

    if direction == "LONG":
        struct_ok = bool((prior_closes <= level).all()) if len(prior_closes) else True
        return bool(last['Close'] > level + atr * c_break_atr and
                    last['Close'] >= last['Open'] and
                    struct_ok and body_ok and vol_ok)
    else:
        struct_ok = bool((prior_closes >= level).all()) if len(prior_closes) else True
        return bool(last['Close'] < level - atr * c_break_atr and
                    last['Close'] <= last['Open'] and
                    struct_ok and body_ok and vol_ok)


def _build_volume_profile(day_df: pd.DataFrame, buckets: int) -> tuple | None:
    """Construit une seule fois l'histogramme de volume par bucket du jour EN
    COURS (aucun lookahead : day_df doit être limité aux bougies jusqu'à la
    bougie courante incluse) — équivalent de l'accumulation vp_vol en Pine.
    Retourne (day_l, bucket_size, vol_by_bucket, max_vol) ou None si non calculable."""
    if day_df.empty:
        return None
    day_h = day_df['High'].max()
    day_l = day_df['Low'].min()
    if pd.isna(day_h) or pd.isna(day_l) or day_h <= day_l:
        return None

    bucket_size = (day_h - day_l) / buckets
    tp  = (day_df['High'] + day_df['Low'] + day_df['Close']) / 3
    idx = ((tp - day_l) / bucket_size).clip(0, buckets - 1).astype(int)
    vol_by_bucket = day_df['Volume'].groupby(idx).sum()
    max_vol = vol_by_bucket.max()
    if max_vol <= 0:
        return None

    return day_l, bucket_size, vol_by_bucket, max_vol


def _vp_pct(profile: tuple | None, level: float, buckets: int) -> float:
    """% de volume (vs bucket max) du bucket contenant `level` — équivalent
    f_vp_pct en Pine. `profile` vient de _build_volume_profile (calculé une
    seule fois par bougie, réutilisé pour chaque niveau testé)."""
    if profile is None or pd.isna(level):
        return 50.0
    day_l, bucket_size, vol_by_bucket, max_vol = profile
    lvl_idx = int(min(max(0, (level - day_l) / bucket_size), buckets - 1))
    return float(vol_by_bucket.get(lvl_idx, 0.0) / max_vol * 100)


def _evaluate_levels(df: pd.DataFrame, cfg: dict) -> tuple[str | None, str, dict]:
    """Détection brute (sans cooldown) sur la dernière bougie de `df`."""
    last  = df.iloc[-1]
    price = last['Close']
    atr   = last.get('ATR', np.nan)

    if pd.isna(atr) or pd.isna(last.get('VWAP', np.nan)):
        return None, "ATR/VWAP NaN", {}

    session = get_active_session(df.index[-1])
    if session == "off":
        return None, f"Hors session ({df.index[-1].strftime('%H:%M')} UTC)", {}

    recent      = df.iloc[-cfg['range_bars']:]
    local_range = recent['High'].max() - recent['Low'].min()
    if local_range <= atr * cfg['range_mult']:
        return None, f"Range local trop faible ({session})", {}

    ema50  = last.get('EMA50_macro',  last.get('EMA50',  np.nan))
    ema200 = last.get('EMA200_macro', last.get('EMA200', np.nan))
    vwap   = last['VWAP']
    macro_src = "1h+VWAP" if 'EMA50_macro' in last.index else "native"

    macro_bull_a = macro_bear_a = True
    if cfg['require_macro_trend']:
        ema_ok = not pd.isna(ema50) and not pd.isna(ema200)
        macro_bull_a = (ema50 >= ema200 if ema_ok else True) and price > vwap
        macro_bear_a = (ema50 <= ema200 if ema_ok else True) and price < vwap
    macro_bull_b = price > vwap
    macro_bear_b = price < vwap

    levels = [
        ("VWAP",     last.get('VWAP')),
        ("VWAP+1σ",  last.get('VWAP_plus_1')),
        ("VWAP+2σ",  last.get('VWAP_plus_2')),
        ("VWAP-1σ",  last.get('VWAP_minus_1')),
        ("VWAP-2σ",  last.get('VWAP_minus_2')),
        ("IDR_H",    last.get('IDR_High')),
        ("IDR_L",    last.get('IDR_Low')),
        ("IDR_M",    last.get('IDR_Mid')),
        ("DR_H",     last.get('DR_High')),
        ("DR_L",     last.get('DR_Low')),
        ("Asian_H",  last.get('Asian_High')),
        ("Asian_L",  last.get('Asian_Low')),
        ("Asian_M",  last.get('Asian_Mid')),
    ]
    extreme_support    = {last.get('VWAP_minus_2'), last.get('DR_Low'),
                          last.get('IDR_Low'), last.get('Asian_Low')}
    extreme_resistance = {last.get('VWAP_plus_2'), last.get('DR_High'),
                          last.get('IDR_High'), last.get('Asian_High')}

    lookback = cfg['lookback']
    hist_h = df['Close'].iloc[-lookback:].max()
    hist_l = df['Close'].iloc[-lookback:].min()

    day_df = df.loc[df.index.normalize() == df.index[-1].normalize()]

    def tp_long(min_dist):
        candidates = [v for _, v in levels if not pd.isna(v) and v > price + min_dist]
        return min(candidates) if candidates else None

    def tp_short(min_dist):
        candidates = [v for _, v in levels if not pd.isna(v) and v < price - min_dist]
        return max(candidates) if candidates else None

    tp_al, tp_bl, tp_cl = tp_long(atr * 0.8),  tp_long(atr * 1.0),  tp_long(atr * 1.2)
    tp_as, tp_bs, tp_cs = tp_short(atr * 0.8), tp_short(atr * 1.0), tp_short(atr * 1.2)

    vol_avg  = last.get('Vol_MA20', np.nan)
    vol_ok_c = bool(pd.isna(vol_avg) or last['Volume'] >= vol_avg * cfg['c_vol_mult'])

    vp_profile = _build_volume_profile(day_df, cfg['vp_buckets'])

    for level_name, level_val in levels:
        if pd.isna(level_val):
            continue

        vp_pct   = _vp_pct(vp_profile, level_val, cfg['vp_buckets'])
        is_lvn   = vp_pct <= cfg['vp_lvn_pct']
        is_hvn   = vp_pct >= cfg['vp_hvn_pct']
        rr_bonus = cfg['vp_rr_bonus'] if is_hvn else 0.0
        vp_tag   = " ★HVN" if is_hvn else ""

        rej_l_args = (df, level_val, "LONG",  atr, cfg['atr_tol'], cfg['wick_ratio'], cfg['body_ratio'], cfg['struct_bars'])
        rej_s_args = (df, level_val, "SHORT", atr, cfg['atr_tol'], cfg['wick_ratio'], cfg['body_ratio'], cfg['struct_bars'])

        # ── Setup A LONG ──────────────────────────────────────────────────
        sl_al = level_val - atr * 0.5
        rr_al = (tp_al - price) / max(price - sl_al, 1e-9) if tp_al is not None else 0.0
        if (not is_lvn and macro_bull_a and _rejection_ok(*rej_l_args) and
                hist_h > level_val + atr * 0.5 and price <= level_val + atr * 1.2 and
                tp_al is not None and rr_al >= cfg['rr_a'] - rr_bonus):
            return ("LONG",
                    f"Setup A ↑ | {level_name}@{level_val:.2f}{vp_tag} → TP {tp_al:.2f} "
                    f"(RR {rr_al:.1f}) | {session} [{macro_src}]",
                    {"tp": tp_al, "sl": sl_al})

        # ── Setup A SHORT ─────────────────────────────────────────────────
        sl_as = level_val + atr * 0.5
        rr_as = (price - tp_as) / max(sl_as - price, 1e-9) if tp_as is not None else 0.0
        if (not is_lvn and macro_bear_a and _rejection_ok(*rej_s_args) and
                hist_l < level_val - atr * 0.5 and price >= level_val - atr * 1.2 and
                tp_as is not None and rr_as >= cfg['rr_a'] - rr_bonus):
            return ("SHORT",
                    f"Setup A ↓ | {level_name}@{level_val:.2f}{vp_tag} → TP {tp_as:.2f} "
                    f"(RR {rr_as:.1f}) | {session} [{macro_src}]",
                    {"tp": tp_as, "sl": sl_as})

        # ── Setup B LONG ──────────────────────────────────────────────────
        macro_ok_bl = True if level_val in extreme_support else macro_bull_b
        sl_bl = level_val - atr * 0.5
        rr_bl = (tp_bl - price) / max(price - sl_bl, 1e-9) if tp_bl is not None else 0.0
        if (not is_lvn and macro_ok_bl and _rejection_ok(*rej_l_args) and
                tp_bl is not None and rr_bl >= cfg['rr_b'] - rr_bonus):
            return ("LONG",
                    f"Setup B ↑ | {level_name}@{level_val:.2f}{vp_tag} → TP {tp_bl:.2f} "
                    f"(RR {rr_bl:.1f}) | {session} [{macro_src}]",
                    {"tp": tp_bl, "sl": sl_bl})

        # ── Setup B SHORT ─────────────────────────────────────────────────
        macro_ok_bs = True if level_val in extreme_resistance else macro_bear_b
        sl_bs = level_val + atr * 0.5
        rr_bs = (price - tp_bs) / max(sl_bs - price, 1e-9) if tp_bs is not None else 0.0
        if (not is_lvn and macro_ok_bs and _rejection_ok(*rej_s_args) and
                tp_bs is not None and rr_bs >= cfg['rr_b'] - rr_bonus):
            return ("SHORT",
                    f"Setup B ↓ | {level_name}@{level_val:.2f}{vp_tag} → TP {tp_bs:.2f} "
                    f"(RR {rr_bs:.1f}) | {session} [{macro_src}]",
                    {"tp": tp_bs, "sl": sl_bs})

        # ── Setup C — cassure / momentum, sans filtre VP ni EMA H1 ─────────
        if cfg['enable_setup_c']:
            sl_cl = level_val - atr * 0.5
            rr_cl = (tp_cl - price) / max(price - sl_cl, 1e-9) if tp_cl is not None else 0.0
            if (macro_bull_b and
                    _breakout_ok(df, level_val, "LONG", atr, cfg['c_body_mult'], cfg['c_break_atr'],
                                cfg['struct_bars'], vol_ok_c) and
                    tp_cl is not None and rr_cl >= cfg['rr_c']):
                return ("LONG",
                        f"Setup C ↑ | Breakout {level_name}@{level_val:.2f} → TP {tp_cl:.2f} "
                        f"(RR {rr_cl:.1f}) | {session} [{macro_src}]",
                        {"tp": tp_cl, "sl": sl_cl})

            sl_cs = level_val + atr * 0.5
            rr_cs = (price - tp_cs) / max(sl_cs - price, 1e-9) if tp_cs is not None else 0.0
            if (macro_bear_b and
                    _breakout_ok(df, level_val, "SHORT", atr, cfg['c_body_mult'], cfg['c_break_atr'],
                                cfg['struct_bars'], vol_ok_c) and
                    tp_cs is not None and rr_cs >= cfg['rr_c']):
                return ("SHORT",
                        f"Setup C ↓ | Breakout {level_name}@{level_val:.2f} → TP {tp_cs:.2f} "
                        f"(RR {rr_cs:.1f}) | {session} [{macro_src}]",
                        {"tp": tp_cs, "sl": sl_cs})

    return None, f"Aucun setup A/B/C | {session}", {}


def detect_signal(df: pd.DataFrame,
                  require_macro_trend: bool = True,
                  rr_a: float = 1.0, rr_b: float = 1.5, rr_c: float = 1.2,
                  wick_ratio: float = 0.6, atr_tol: float = 0.3,
                  struct_bars: int = 3, body_ratio: float = 0.3,
                  range_bars: int = 8, range_mult: float = 1.5,
                  lookback: int = 10, cooldown_bars: int = 5,
                  vp_buckets: int = 20, vp_hvn_pct: float = 70.0,
                  vp_lvn_pct: float = 30.0, vp_rr_bonus: float = 0.3,
                  enable_setup_c: bool = True, c_body_mult: float = 0.8,
                  c_break_atr: float = 0.3, c_vol_mult: float = 1.3,
                  # Legacy params conservés pour compatibilité main.py / Backtest.py
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
    Trois setups sur niveaux VWAP / DR / IDR / Asian (portage du Pine
    "VWAP Strategy — Setup A/B/C v6") :

    Setup A — Post-breakout retest : le prix a cassé un niveau récemment,
      reteste et rejette dans le sens du breakout.
    Setup B — Range bounce : rejet sur bande VWAP / DR / IDR / Asian avec
      RR ≥ rr_b vers le niveau opposé le plus proche (macro ignorée sur les
      niveaux extrêmes : VWAP-2σ, DR_Low, IDR_Low, Asian_Low pour le support,
      symétrique pour la résistance).
    Setup C — Cassure / momentum net, sans exigence de mèche de rejet, filtré
      par le volume et un corps de bougie large.

    Filtres transverses : session (London/NY heure de Paris), range local
    anti-chop, Volume Profile journalier (HVN bonifie le RR minimum requis,
    LVN bloque le signal), cooldown en nombre de bougies entre deux signaux.

    Retourne (direction | None, reason, {"tp": float, "sl": float}).
    """
    min_len = max(25, struct_bars + 2, lookback, range_bars)
    if 'VWAP' not in df.columns or len(df) < min_len:
        return None, "Données insuffisantes", {}

    cfg = dict(require_macro_trend=require_macro_trend, rr_a=rr_a, rr_b=rr_b, rr_c=rr_c,
              wick_ratio=wick_ratio, atr_tol=atr_tol, struct_bars=struct_bars,
              body_ratio=body_ratio, range_bars=range_bars, range_mult=range_mult,
              lookback=lookback, vp_buckets=vp_buckets, vp_hvn_pct=vp_hvn_pct,
              vp_lvn_pct=vp_lvn_pct, vp_rr_bonus=vp_rr_bonus, enable_setup_c=enable_setup_c,
              c_body_mult=c_body_mult, c_break_atr=c_break_atr, c_vol_mult=c_vol_mult)

    signal, reason, meta = _evaluate_levels(df, cfg)
    if signal is None:
        return signal, reason, meta

    # ── Cooldown : bloque si un signal (toute direction/setup) est déjà
    #    survenu dans les cooldown_bars bougies précédentes ────────────────
    for k in range(1, cooldown_bars + 1):
        if len(df) - k < min_len:
            break
        prev_signal, _, _ = _evaluate_levels(df.iloc[:-k], cfg)
        if prev_signal is not None:
            return None, f"Cooldown actif ({k} bougie(s) depuis dernier signal)", {}

    return signal, reason, meta


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