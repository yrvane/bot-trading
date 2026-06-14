"""
telegram_alerts.py — Alertes Telegram pour le bot de trading

Setup (5 min) :
  1. Ouvre Telegram → cherche @BotFather → /newbot → copie le TOKEN
  2. Cherche @userinfobot → envoie n'importe quel message → copie ton CHAT_ID
  3. Ajoute dans .env :
       TELEGRAM_TOKEN=123456789:ABCdef...
       TELEGRAM_CHAT_ID=987654321

Test :
  python telegram_alerts.py --test
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime

# ── Chargement .env ───────────────────────────────────────────────────────────
def _load_env():
    candidates = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"✅ .env trouvé : {p}")
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip()
                        os.environ[key] = val  # ← force l'écrasement
                        print(f"   → {key} chargé")
            return
    print("❌ Aucun .env trouvé dans :")
    for p in candidates:
        print(f"   {p}")

_load_env()

print(f"DEBUG TOKEN = '{os.getenv('TELEGRAM_BOT_TOKEN')}'")
print(f"DEBUG CHAT  = '{os.getenv('TELEGRAM_CHAT_ID')}'")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVOI DE MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════

def _send(text: str, parse_mode: str = "HTML") -> bool:
    """Envoie un message Telegram. Retourne True si succès."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram non configuré (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID manquants)")
        return False

    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                print(f"📲 Telegram envoyé : {text[:60]}...")
                return True
            else:
                print(f"⚠️  Telegram error : {result}")
                return False
    except Exception as e:
        print(f"⚠️  Telegram exception : {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MESSAGES TYPÉS
# ═══════════════════════════════════════════════════════════════════════════════

def alert_new_trade(trade: dict, sentiment: str = "", reason: str = "") -> bool:
    """Alerte ouverture de position."""
    is_long    = trade["type"] == "LONG"
    entry      = float(trade["entry_price"])
    sl         = float(trade["sl"])
    tp         = float(trade["tp"])
    risk       = abs(entry - sl)
    reward     = abs(tp - entry)
    rr         = f"1 : {reward/risk:.1f}" if risk > 0 else "—"
    now        = datetime.now().strftime("%H:%M")
    emoji_dir  = "🟢" if is_long else "🔴"
    emoji_type = "📈 LONG" if is_long else "📉 SHORT"

    text = (
        f"{emoji_dir} <b>SIGNAL {emoji_type} — {trade['symbol']}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 Entrée  : <b>{entry:.2f}</b>\n"
        f"🛑 SL      : <b>{sl:.2f}</b>\n"
        f"🎯 TP      : <b>{tp:.2f}</b>\n"
        f"⚖️  R/R    : <b>{rr}</b>\n"
    )
    if sentiment:
        text += f"📰 News    : {sentiment}\n"
    if reason:
        text += f"📊 Signal  : {reason[:80]}\n"
    text += f"⏰ <i>{now}</i>"

    return _send(text)


def alert_trade_closed(symbol: str, exit_price: float,
                       pnl: float, reason: str) -> bool:
    """Alerte fermeture de position."""
    is_win   = pnl >= 0
    emoji    = "✅" if is_win else "❌"
    pnl_str  = f"{'+' if is_win else ''}{pnl:.2f}%"
    now      = datetime.now().strftime("%H:%M")

    text = (
        f"{emoji} <b>TRADE FERMÉ — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 Sortie  : <b>{exit_price:.2f}</b>\n"
        f"💹 PnL     : <b>{pnl_str}</b>\n"
        f"🏷️  Raison  : <b>{reason}</b>\n"
        f"⏰ <i>{now}</i>"
    )

    return _send(text)


def alert_signal_blocked(symbol: str, direction: str,
                         sentiment_signal: str, confidence: int,
                         summary: str) -> bool:
    """Alerte quand un signal est bloqué par le sentiment news."""
    text = (
        f"⛔ <b>SIGNAL BLOQUÉ — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 Direction : {direction}\n"
        f"📰 Sentiment : {sentiment_signal} ({confidence}%)\n"
        f"💬 <i>{summary}</i>"
    )
    return _send(text)


def alert_config_updated(reasons: list, config: dict) -> bool:
    """Alerte quand l'auto-critique met à jour les paramètres."""
    reasons_str = "\n".join(f"  • {r}" for r in reasons)
    text = (
        f"⚙️ <b>CONFIG MISE À JOUR (auto-critique)</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"{reasons_str}\n\n"
        f"📐 RSI       : [{config['rsi_min']} – {config['rsi_max']}]\n"
        f"🛑 SL mult   : {config['sl_mult']}x ATR\n"
        f"🎯 TP mult   : {config['tp_mult']}x ATR"
    )
    return _send(text)


def alert_daily_summary(summary: dict) -> bool:
    """Envoie un résumé quotidien du bot avec les axes d'amélioration."""
    improvements = summary.get("improvements", [])
    improvement_text = "\n".join(f"  • {item}" for item in improvements[:5]) or "  • Aucun axe prioritaire identifié"

    top_symbol = summary.get("top_symbol", "—")
    best_hour = summary.get("best_hour", "—")
    worst_hour = summary.get("worst_hour", "—")
    pnl = summary.get("total_pnl_pct", 0)
    pnl_str = f"{pnl:+.2f}%"

    text = (
        f"📅 <b>RÉSUMÉ QUOTIDIEN — {summary.get('date_label', 'Aujourd\'hui')}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📈 Trades fermés : <b>{summary.get('closed_trades', 0)}</b>\n"
        f"🟡 Trades ouverts : <b>{summary.get('open_trades', 0)}</b>\n"
        f"✅ Win rate       : <b>{summary.get('win_rate', 0):.1f}%</b>\n"
        f"💹 PnL du jour    : <b>{pnl_str}</b>\n"
        f"⚖️  Profit factor  : <b>{summary.get('profit_factor', '—')}</b>\n"
        f"🧠 Pire série     : <b>{summary.get('max_streak', 0)}</b> pertes\n"
        f"🏷️  Meilleur symbole : <b>{top_symbol}</b>\n"
        f"⏱️  Meilleure heure   : <b>{best_hour}</b>\n"
        f"⏱️  Pire heure        : <b>{worst_hour}</b>\n\n"
        f"🛠️ <b>Axes d'amélioration</b>\n"
        f"{improvement_text}"
    )
    return _send(text)


def alert_bot_started() -> bool:
    """Message de démarrage du bot."""
    now  = datetime.now().strftime("%d/%m/%Y %H:%M")
    text = (
        f"🚀 <b>Bot Trading démarré</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📊 Marchés  : XAUUSD · US100 · US500\n"
        f"🧠 Stratégie : VWAP + RSI + EMA200\n"
        f"📰 Sentiment : actif\n"
        f"⏰ <i>{now}</i>"
    )
    return _send(text)


def alert_test() -> bool:
    """Message de test pour vérifier la configuration."""
    text = (
        "✅ <b>Test Telegram — Bot Trading</b>\n"
        "━━━━━━━━━━━━━━━━\n"
        "La connexion fonctionne correctement.\n"
        "Tu recevras ici :\n"
        "  📈 Les signaux d'entrée\n"
        "  📉 Les clôtures de trades\n"
        "  ⚙️  Les mises à jour de config\n"
        "  ⛔ Les signaux bloqués par les news"
    )
    return _send(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Envoyer un message de test")
    args = parser.parse_args()

    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN manquant dans .env")
        print("   1. Ouvre Telegram → @BotFather → /newbot")
        print("   2. Copie le token dans .env : TELEGRAM_TOKEN=xxx")
    elif not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_CHAT_ID manquant dans .env")
        print("   1. Ouvre Telegram → @userinfobot → envoie un message")
        print("   2. Copie ton ID dans .env : TELEGRAM_CHAT_ID=xxx")
    elif args.test:
        print("📲 Envoi du message de test...")
        ok = alert_test()
        print("✅ Succès !" if ok else "❌ Échec — vérifie token et chat_id")
    else:
        print("Usage : python telegram_alerts.py --test")