"""
news_sentiment.py — Sentiment de marché via RSS + Claude API
"""

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional
import urllib.request
import os

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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    print("⚠️  ANTHROPIC_API_KEY non définie — fallback sur analyse par mots-clés")


# ═══════════════════════════════════════════════════════════════════════════════
#  SOURCES RSS PAR SYMBOLE
# ═══════════════════════════════════════════════════════════════════════════════

RSS_SOURCES = {
    "XAUUSD": [
        "https://feeds.kitco.com/MarketNuggets.xml",
        "https://www.investing.com/rss/news_357.rss",
        "https://www.fxstreet.com/rss/news",
    ],
    "US100": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
        "https://www.investing.com/rss/news_25.rss",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ],
    "US500": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
        "https://www.investing.com/rss/news_1.rss",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ],
    "_default": [
        "https://feeds.marketwatch.com/marketwatch/topstories/",
        "https://www.fxstreet.com/rss/news",
    ],
}

KEYWORDS = {
    "XAUUSD": ["gold", "xau", "or", "bullion", "inflation", "fed", "dollar", "safe haven"],
    "US100":  ["nasdaq", "tech", "apple", "nvidia", "microsoft", "ai", "interest rate"],
    "US500":  ["s&p", "sp500", "stock market", "wall street", "dow", "equity"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  FETCH RSS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_rss(url: str, timeout: int = 8) -> list:
    articles = []
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        root = ET.fromstring(content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            desc  = item.findtext("description", "").strip()
            if title:
                articles.append({"title": title, "desc": desc[:300]})

        for entry in root.findall(".//atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip()
            desc  = entry.findtext("atom:summary", "", ns).strip()
            if title:
                articles.append({"title": title, "desc": desc[:300]})

    except Exception:
        pass

    return articles[:10]


def get_news(symbol: str, max_articles: int = 15) -> list:
    sources  = RSS_SOURCES.get(symbol, RSS_SOURCES["_default"])
    keywords = KEYWORDS.get(symbol, [])
    all_articles = []

    for url in sources:
        for art in fetch_rss(url):
            text = (art["title"] + " " + art["desc"]).lower()
            if not keywords or any(kw in text for kw in keywords):
                all_articles.append(art)

    seen, unique = set(), []
    for art in all_articles:
        if art["title"] not in seen:
            seen.add(art["title"])
            unique.append(art)

    return unique[:max_articles]


# ═══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT VIA CLAUDE API
# ═══════════════════════════════════════════════════════════════════════════════

def call_claude(prompt: str) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        print(f"⚠️  Claude API error: {e}")
        return None


def analyze_sentiment_with_claude(symbol: str, articles: list) -> dict:
    headlines = "\n".join(
        f"- {a['title']}" + (f" | {a['desc'][:100]}" if a['desc'] else "")
        for a in articles[:12]
    )

    prompt = f"""Tu es un analyste financier expert. Analyse ces headlines concernant {symbol} et détermine le sentiment de marché à court terme.

HEADLINES :
{headlines}

Réponds UNIQUEMENT avec ce JSON (sans markdown, sans explication) :
{{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0-100,
  "summary": "Résumé en 1 phrase",
  "key_factors": ["facteur1", "facteur2"],
  "risk_events": ["événement à risque si applicable"]
}}"""

    response = call_claude(prompt)
    if not response:
        return _fallback_sentiment(articles)

    try:
        clean  = re.sub(r"```json|```", "", response).strip()
        result = json.loads(clean)
        result["source"]         = "claude"
        result["articles_count"] = len(articles)
        result["timestamp"]      = datetime.now(timezone.utc).isoformat()
        return result
    except Exception:
        return _fallback_sentiment(articles)


def _fallback_sentiment(articles: list) -> dict:
    POSITIVE = ["surge","rally","gain","rise","jump","bull","high","record",
                "strong","boost","growth","beat","optimism","hausse","monte"]
    NEGATIVE = ["fall","drop","decline","crash","fear","loss","weak","cut",
                "risk","warn","sell","bear","low","recession","baisse","chute"]

    pos = neg = 0
    for art in articles:
        text = (art["title"] + " " + art["desc"]).lower()
        pos += sum(1 for w in POSITIVE if w in text)
        neg += sum(1 for w in NEGATIVE if w in text)

    total = pos + neg
    if total == 0:
        signal, confidence = "NEUTRAL", 50
    elif pos / total > 0.6:
        signal, confidence = "BULLISH", int(pos / total * 100)
    elif neg / total > 0.6:
        signal, confidence = "BEARISH", int(neg / total * 100)
    else:
        signal, confidence = "NEUTRAL", 50

    return {
        "signal":         signal,
        "confidence":     confidence,
        "summary":        f"Analyse mots-clés ({pos} positifs / {neg} négatifs)",
        "key_factors":    [],
        "risk_events":    [],
        "source":         "keywords",
        "articles_count": len(articles),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE PUBLIC
# ═══════════════════════════════════════════════════════════════════════════════

_cache: dict = {}
CACHE_TTL = 900  # 15 minutes


def get_sentiment(symbol: str, verbose: bool = False) -> dict:
    now = time.time()
    if symbol in _cache:
        cached, ts = _cache[symbol]
        if now - ts < CACHE_TTL:
            if verbose:
                print(f"📰 Sentiment {symbol} (cache) : {cached['signal']} ({cached['confidence']}%)")
            return cached

    articles = get_news(symbol)
    result   = analyze_sentiment_with_claude(symbol, articles) if articles else {
        "signal": "NEUTRAL", "confidence": 50,
        "summary": "Aucune news récupérée", "key_factors": [], "risk_events": [],
        "source": "none", "articles_count": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _cache[symbol] = (result, now)

    if verbose:
        print(f"\n📰 SENTIMENT {symbol}")
        print(f"   Signal     : {result['signal']} ({result['confidence']}%)")
        print(f"   Résumé     : {result['summary']}")
        print(f"   Source     : {result['source']} ({result['articles_count']} articles)")
        if result.get('key_factors'):
            print(f"   Facteurs   : {', '.join(result['key_factors'])}")
        if result.get('risk_events'):
            print(f"   ⚠️  Risques : {', '.join(result['risk_events'])}")

    return result


def sentiment_allows_trade(sentiment: dict, direction: str) -> bool:
    sig  = sentiment.get("signal", "NEUTRAL")
    conf = sentiment.get("confidence", 50)
    if conf < 55:       return True
    if sig == "NEUTRAL": return True
    if direction == "LONG"  and sig == "BULLISH": return True
    if direction == "SHORT" and sig == "BEARISH":  return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="XAUUSD")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"🔍 Récupération des news pour {args.symbol}...")
    result = get_sentiment(args.symbol, verbose=True)
    print(f"\n✅ LONG autorisé  : {sentiment_allows_trade(result, 'LONG')}")
    print(f"✅ SHORT autorisé : {sentiment_allows_trade(result, 'SHORT')}")