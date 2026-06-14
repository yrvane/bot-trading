"""
analyzer.py — Auto-critique des trades perdus
Analyse les conditions de marché au moment des trades perdants
et ajuste automatiquement les paramètres de la stratégie.

Usage :
    python analyzer.py                        # analyse + rapport terminal
    python analyzer.py --apply                # analyse + applique les ajustements
    python analyzer.py --symbol XAUUSD        # filtre par symbole
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Fichier de config dynamique ───────────────────────────────────────────────
CONFIG_FILE = "strategy_config.json"
TRADES_FILE_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "bot_trades.csv"
)

DEFAULT_CONFIG = {
    "rsi_min":           40.0,
    "rsi_max":           60.0,
    "require_ema_trend": True,
    "require_volume":    True,
    "sl_mult":           1.0,
    "tp_mult":           2.0,
    "last_updated":      None,
    "update_reason":     "Paramètres par défaut",
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    config["last_updated"] = datetime.now().isoformat()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config sauvegardée → {CONFIG_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES TRADES
# ═══════════════════════════════════════════════════════════════════════════════

def load_trades(path: str = None, symbol: str = None) -> pd.DataFrame:
    path = path or TRADES_FILE_DEFAULT
    if not os.path.exists(path):
        print(f"❌ Fichier trades introuvable : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    # Garder uniquement les trades fermés
    df = df[df['status'] == 'closed'].copy()

    if symbol:
        df = df[df['symbol'] == symbol]

    if df.empty:
        print("⚠️  Aucun trade fermé trouvé.")
    else:
        print(f"✅ {len(df)} trades fermés chargés")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(df: pd.DataFrame) -> dict:
    """Analyse statistique complète des trades."""
    if df.empty:
        return {}

    df = df.copy()
    df['win']    = df['pnl'] > 0
    df['pnl_abs'] = df['pnl'].abs()

    wins   = df[df['win']]
    losses = df[~df['win']]

    # ── Stats de base ─────────────────────────────────────────────────────────
    win_rate    = len(wins) / len(df) * 100
    avg_win     = wins['pnl'].mean()   if len(wins)   > 0 else 0
    avg_loss    = losses['pnl'].mean() if len(losses) > 0 else 0
    pf          = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')

    # ── Pire série de pertes ──────────────────────────────────────────────────
    streak = max_streak = 0
    for w in df['win']:
        streak = 0 if w else streak + 1
        max_streak = max(max_streak, streak)

    # ── Analyse par type (LONG/SHORT) ─────────────────────────────────────────
    by_type = {}
    for t in ['LONG', 'SHORT']:
        sub = df[df['type'] == t]
        if len(sub) > 0:
            by_type[t] = {
                'count':    len(sub),
                'win_rate': round(sub['win'].mean() * 100, 1),
                'avg_pnl':  round(sub['pnl'].mean(), 2),
            }

    # ── Analyse des raisons de sortie ─────────────────────────────────────────
    by_reason = {}
    if 'exit_price' in df.columns and 'sl' in df.columns and 'tp' in df.columns:
        df['exit_reason'] = df.apply(
            lambda r: 'SL' if abs(r['exit_price'] - r['sl']) < abs(r['exit_price'] - r['tp']) else 'TP',
            axis=1
        )
        by_reason = df.groupby('exit_reason')['win'].agg(['count','mean']).to_dict()

    # ── Patterns temporels ────────────────────────────────────────────────────
    df['hour'] = df['date'].dt.hour
    df['dow']  = df['date'].dt.dayofweek   # 0=Lundi

    hour_perf = df.groupby('hour')['pnl'].agg(['mean','count']).round(3)
    dow_perf  = df.groupby('dow')['pnl'].agg(['mean','count']).round(3)
    dow_names = {0:'Lundi',1:'Mardi',2:'Mercredi',3:'Jeudi',4:'Vendredi'}
    dow_perf.index = dow_perf.index.map(lambda x: dow_names.get(x, str(x)))

    # ── Meilleure / pire heure ─────────────────────────────────────────────────
    best_hour = int(hour_perf['mean'].idxmax()) if not hour_perf.empty else None
    worst_hour = int(hour_perf['mean'].idxmin()) if not hour_perf.empty else None

    return {
        'total':        len(df),
        'wins':         len(wins),
        'losses':       len(losses),
        'win_rate':     round(win_rate, 1),
        'avg_win':      round(avg_win, 2),
        'avg_loss':     round(avg_loss, 2),
        'profit_factor': round(pf, 2),
        'max_streak':   max_streak,
        'by_type':      by_type,
        'by_reason':    by_reason,
        'best_hour':    best_hour,
        'worst_hour':   worst_hour,
        'hour_perf':    hour_perf.to_dict(),
        'dow_perf':     dow_perf.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-CRITIQUE → AJUSTEMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def suggest_adjustments(analysis: dict, current_config: dict) -> tuple[dict, list]:
    """
    Lit les stats et propose des ajustements aux paramètres de la stratégie.
    Retourne (nouvelle_config, liste_de_raisons).
    """
    config  = current_config.copy()
    reasons = []

    if not analysis:
        return config, ["Pas assez de données pour analyser"]

    wr = analysis.get('win_rate', 50)
    pf = analysis.get('profit_factor', 1)
    ms = analysis.get('max_streak', 0)

    # ── Win rate trop bas → resserrer les filtres RSI ─────────────────────────
    if wr < 40:
        old = (config['rsi_min'], config['rsi_max'])
        config['rsi_min'] = min(config['rsi_min'] + 3, 48)
        config['rsi_max'] = max(config['rsi_max'] - 3, 52)
        reasons.append(
            f"Win rate faible ({wr}%) → RSI resserré {old} → ({config['rsi_min']},{config['rsi_max']})"
        )

    # ── Profit factor < 1 → augmenter le TP ──────────────────────────────────
    if pf < 1.0:
        old = config['tp_mult']
        config['tp_mult'] = round(min(config['tp_mult'] + 0.25, 3.0), 2)
        reasons.append(
            f"Profit factor < 1 ({pf}) → TP multiplié {old}x → {config['tp_mult']}x"
        )

    # ── Trop de SL touchés → élargir le SL ───────────────────────────────────
    by_reason = analysis.get('by_reason', {})
    sl_count  = by_reason.get('count', {}).get('SL', 0)
    tp_count  = by_reason.get('count', {}).get('TP', 0)
    if sl_count > 0 and tp_count > 0 and sl_count / (sl_count + tp_count) > 0.7:
        old = config['sl_mult']
        config['sl_mult'] = round(min(config['sl_mult'] + 0.1, 2.0), 2)
        reasons.append(
            f"Trop de SL ({sl_count} vs {tp_count} TP) → SL élargi {old}x → {config['sl_mult']}x"
        )

    # ── Pire série de pertes élevée → activer filtre volume ──────────────────
    if ms >= 5 and not config['require_volume']:
        config['require_volume'] = True
        reasons.append(f"Série de {ms} pertes → filtre volume activé")

    # ── Performance mauvaise sur SHORT → désactiver SHORT ────────────────────
    by_type = analysis.get('by_type', {})
    if 'SHORT' in by_type and by_type['SHORT']['win_rate'] < 35:
        # On ne désactive pas directement mais on le signale
        reasons.append(
            f"SHORT peu performant (win rate {by_type['SHORT']['win_rate']}%) "
            f"— envisager de désactiver les signaux SHORT"
        )

    # ── Aucun ajustement nécessaire ───────────────────────────────────────────
    if not reasons:
        reasons.append(
            f"Stratégie stable (WR {wr}%, PF {pf}) — aucun ajustement nécessaire"
        )

    config['update_reason'] = " | ".join(reasons)
    return config, reasons


# ═══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(analysis: dict, new_config: dict, reasons: list):
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  AUTO-CRITIQUE — Analyse des trades")
    print(f"{'═'*60}")
    print(f"  Total trades   : {analysis['total']}")
    print(f"  Win rate       : {analysis['win_rate']}%  ({analysis['wins']}W / {analysis['losses']}L)")
    print(f"  Avg gain       : +{analysis['avg_win']}$")
    print(f"  Avg loss       :  {analysis['avg_loss']}$")
    print(f"  Profit factor  : {analysis['profit_factor']}")
    print(f"  Pire série     : {analysis['max_streak']} pertes consécutives")

    if analysis.get('by_type'):
        print(f"\n  {sep}")
        print(f"  PAR DIRECTION")
        for t, v in analysis['by_type'].items():
            print(f"  {t:6s} : {v['count']} trades | WR {v['win_rate']}% | PnL moy {v['avg_pnl']:+.2f}%")

    if analysis.get('best_hour') is not None:
        print(f"\n  {sep}")
        print(f"  HORAIRES")
        print(f"  Meilleure heure : {analysis['best_hour']}h00")
        print(f"  Pire heure      : {analysis['worst_hour']}h00")

    print(f"\n{'═'*60}")
    print(f"  AJUSTEMENTS SUGGÉRÉS")
    print(f"{'═'*60}")
    for r in reasons:
        print(f"  → {r}")

    print(f"\n  Nouvelle config :")
    print(f"  RSI zone         : [{new_config['rsi_min']} – {new_config['rsi_max']}]")
    print(f"  SL multiplicateur: {new_config['sl_mult']}x ATR")
    print(f"  TP multiplicateur: {new_config['tp_mult']}x ATR")
    print(f"  Filtre EMA200    : {'Oui' if new_config['require_ema_trend'] else 'Non'}")
    print(f"  Filtre Volume    : {'Oui' if new_config['require_volume'] else 'Non'}")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-critique des trades")
    parser.add_argument("--symbol", default=None,  help="Filtrer par symbole")
    parser.add_argument("--apply",  action="store_true", help="Appliquer les ajustements")
    parser.add_argument("--trades", default=None,  help="Chemin alternatif vers bot_trades.csv")
    args = parser.parse_args()

    df      = load_trades(path=args.trades, symbol=args.symbol)
    if df.empty:
        exit(0)

    analysis    = analyze(df)
    cur_config  = load_config()
    new_config, reasons = suggest_adjustments(analysis, cur_config)

    print_report(analysis, new_config, reasons)

    if args.apply:
        save_config(new_config)
        print("✅ Paramètres mis à jour — le bot utilisera la nouvelle config au prochain démarrage.")
    else:
        print("💡 Ajoute --apply pour appliquer ces ajustements automatiquement.")