"""
=============================================================
  BACKTEST — Stratégie VWAP Crossover
  Compatible avec data_fetcher.py et strategies.py existants
=============================================================
Usage :
    python backtest.py
    python backtest.py --symbol US100 --interval 1h --period 180d
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# ── Import de tes modules existants ──────────────────────────────────────────
try:
    from data_fetcher import get_historical_data_yfinance
    from strategies import calculate_vwap, detect_signal, calculate_sl_tp
    print("✅ Modules importés depuis ton projet")
except ImportError as e:
    print(f"⚠️  Impossible d'importer les modules ({e})")
    print("   Lance ce script depuis le dossier de ton projet.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTEUR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(symbol: str, interval: str = "1h", period: str = "180d",
                 initial_capital: float = 10_000, risk_pct: float = 1.0):
    """
    Rejoue la stratégie bougie par bougie sur données historiques.

    Paramètres
    ----------
    symbol          : XAUUSD | US100 | US500
    interval        : 5m | 15m | 1h | 4h | 1d
    period          : période yfinance (5d, 30d, 60d, 180d…)
    initial_capital : capital de départ en $
    risk_pct        : % du capital risqué par trade (ex: 1.0 = 1%)
    """

    print(f"\n{'═'*60}")
    print(f"  BACKTEST  {symbol}  |  {interval}  |  {period}")
    print(f"{'═'*60}")

    # ── 1. Chargement des données ─────────────────────────────────────────────
    print(f"📥 Chargement des données {symbol}...")
    df_raw = get_historical_data_yfinance(symbol, period=period, interval=interval)

    if df_raw is None or df_raw.empty:
        print("❌ Pas de données disponibles. Vérifiez symbol/period/interval.")
        return None

    df_raw = calculate_vwap(df_raw)
    df_raw = df_raw.dropna(subset=["VWAP", "ATR"])
    print(f"✅ {len(df_raw)} bougies chargées ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    # ── 2. Simulation bougie par bougie ───────────────────────────────────────
    capital     = initial_capital
    equity      = [initial_capital]
    trades      = []
    position    = None          # trade en cours
    last_signal = None

    # On simule à partir de la bougie 20 (assez d'historique pour VWAP/ATR)
    for i in range(20, len(df_raw)):
        window   = df_raw.iloc[:i+1]
        cur_bar  = df_raw.iloc[i]
        cur_time = df_raw.index[i]

        # ── Gestion de la position ouverte ────────────────────────────────────
        if position is not None:
            hit_sl = hit_tp = False

            if position["type"] == "LONG":
                if cur_bar["Low"]  <= position["sl"]: hit_sl = True
                if cur_bar["High"] >= position["tp"]: hit_tp = True
            else:  # SHORT
                if cur_bar["High"] >= position["sl"]: hit_sl = True
                if cur_bar["Low"]  <= position["tp"]: hit_tp = True

            # On ferme au premier événement (SL prioritaire si les deux)
            if hit_sl or hit_tp:
                exit_price = position["sl"] if hit_sl else position["tp"]
                reason     = "SL" if hit_sl else "TP"

                if position["type"] == "LONG":
                    pnl_pct = (exit_price - position["entry"]) / position["entry"] * 100
                else:
                    pnl_pct = (position["entry"] - exit_price) / position["entry"] * 100

                pnl_dollar = capital * (risk_pct / 100) * (pnl_pct / abs(
                    (position["sl"] - position["entry"]) / position["entry"] * 100
                )) if position["sl"] != position["entry"] else 0

                capital += pnl_dollar

                trades.append({
                    "open_time":   position["open_time"],
                    "close_time":  cur_time,
                    "symbol":      symbol,
                    "type":        position["type"],
                    "entry":       position["entry"],
                    "sl":          position["sl"],
                    "tp":          position["tp"],
                    "exit":        exit_price,
                    "reason":      reason,
                    "pnl_pct":     round(pnl_pct, 4),
                    "pnl_dollar":  round(pnl_dollar, 2),
                    "capital":     round(capital, 2),
                    "win":         pnl_dollar > 0,
                })
                position    = None
                last_signal = None

        equity.append(round(capital, 2))

        # ── Recherche de signal (seulement si pas en position) ─────────────────
        if position is None:
            signal, reason_sig = detect_signal(window)

            if signal and signal != last_signal:
                last_signal  = signal
                entry_price  = cur_bar["Close"]
                atr          = cur_bar["ATR"]
                sl, tp       = calculate_sl_tp(entry_price, signal, atr)

                position = {
                    "type":      signal,
                    "entry":     entry_price,
                    "sl":        sl,
                    "tp":        tp,
                    "open_time": cur_time,
                }
            elif not signal:
                last_signal = None

    # ── 3. Stats ──────────────────────────────────────────────────────────────
    if not trades:
        print("⚠️  Aucun trade généré sur cette période.")
        return None

    df_trades = pd.DataFrame(trades)

    wins        = df_trades["win"].sum()
    losses      = len(df_trades) - wins
    win_rate    = wins / len(df_trades) * 100
    total_pnl   = df_trades["pnl_dollar"].sum()
    avg_win     = df_trades.loc[df_trades["win"],  "pnl_dollar"].mean() if wins  > 0 else 0
    avg_loss    = df_trades.loc[~df_trades["win"], "pnl_dollar"].mean() if losses > 0 else 0
    profit_factor = abs(df_trades.loc[df_trades["win"],"pnl_dollar"].sum() /
                        df_trades.loc[~df_trades["win"],"pnl_dollar"].sum()) if losses > 0 else float("inf")

    # Drawdown maximum
    eq_series   = pd.Series(equity)
    rolling_max = eq_series.cummax()
    drawdown    = (eq_series - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    # Ratio de Sharpe simplifié (rendements par trade)
    returns     = df_trades["pnl_dollar"] / initial_capital
    sharpe      = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Série de pertes consécutives
    streak      = 0
    max_streak  = 0
    for w in df_trades["win"]:
        streak = 0 if w else streak + 1
        max_streak = max(max_streak, streak)

    stats = {
        "symbol":         symbol,
        "interval":       interval,
        "period":         period,
        "start":          str(df_raw.index[0].date()),
        "end":            str(df_raw.index[-1].date()),
        "initial_capital": initial_capital,
        "final_capital":  round(capital, 2),
        "total_trades":   len(df_trades),
        "wins":           int(wins),
        "losses":         int(losses),
        "win_rate":       round(win_rate, 1),
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round(total_pnl / initial_capital * 100, 2),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "max_drawdown":   round(max_dd, 2),
        "sharpe_ratio":   round(sharpe, 2),
        "max_loss_streak": int(max_streak),
    }

    # ── 4. Affichage terminal ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  RÉSULTATS — {symbol} {interval}")
    print(f"{'─'*60}")
    print(f"  Période          : {stats['start']} → {stats['end']}")
    print(f"  Capital initial  : ${initial_capital:,.0f}")
    print(f"  Capital final    : ${capital:,.2f}  ({stats['total_pnl_pct']:+.2f}%)")
    print(f"  Total trades     : {stats['total_trades']}")
    print(f"  Win rate         : {stats['win_rate']}%  ({wins}W / {losses}L)")
    print(f"  Avg gain         : ${avg_win:+.2f}")
    print(f"  Avg loss         : ${avg_loss:+.2f}")
    print(f"  Profit factor    : {profit_factor:.2f}")
    print(f"  Max drawdown     : {max_dd:.2f}%")
    print(f"  Sharpe ratio     : {sharpe:.2f}")
    print(f"  Pires pertes consécutives : {max_streak}")
    print(f"{'─'*60}\n")

    return {"stats": stats, "trades": df_trades, "equity": equity, "df": df_raw}


# ═══════════════════════════════════════════════════════════════════════════════
#  RAPPORT HTML
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(results_list: list, output_path: str = "backtest_report.html"):
    """Génère un rapport HTML interactif avec Chart.js"""

    cards_html = ""
    charts_data = []

    for res in results_list:
        s = res["stats"]
        df_t = res["trades"]
        equity = res["equity"]

        pnl_color  = "#00ffaa" if s["total_pnl"] >= 0 else "#ff3a5c"
        wr_color   = "#00ffaa" if s["win_rate"] >= 50 else "#ff3a5c"
        dd_color   = "#ff3a5c" if s["max_drawdown"] < -10 else "#ffc23a"
        pf_color   = "#00ffaa" if s["profit_factor"] >= 1.5 else "#ffc23a"

        chart_id = f"equity_{s['symbol']}_{s['interval']}".replace("/","")

        # Trade rows
        trade_rows = ""
        for _, t in df_t.tail(30).iloc[::-1].iterrows():
            w = t["win"]
            trade_rows += f"""
            <tr>
              <td>{pd.Timestamp(t['open_time']).strftime('%d/%m %H:%M')}</td>
              <td><span class="badge {'long' if t['type']=='LONG' else 'short'}">{t['type']}</span></td>
              <td>{t['entry']:.2f}</td>
              <td class="red">{t['sl']:.2f}</td>
              <td class="green">{t['tp']:.2f}</td>
              <td>{t['exit']:.2f}</td>
              <td class="{'green' if w else 'red'}">{'+' if w else ''}{t['pnl_pct']:.2f}%</td>
              <td class="{'green' if w else 'red'}">{'+' if t['pnl_dollar']>=0 else ''}{t['pnl_dollar']:.2f}$</td>
              <td><span class="reason {'tp-badge' if t['reason']=='TP' else 'sl-badge'}">{t['reason']}</span></td>
            </tr>"""

        charts_data.append({
            "id": chart_id,
            "label": f"{s['symbol']} {s['interval']}",
            "equity": equity,
            "color": "#00ffaa" if s["total_pnl"] >= 0 else "#ff3a5c",
        })

        cards_html += f"""
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">{s['symbol']}</div>
              <div class="card-sub">{s['interval']} · {s['start']} → {s['end']}</div>
            </div>
            <div class="pnl-big" style="color:{pnl_color}">{s['total_pnl_pct']:+.2f}%</div>
          </div>

          <div class="stat-row">
            <div class="stat-box">
              <div class="sl">Trades</div>
              <div class="sv">{s['total_trades']}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Win Rate</div>
              <div class="sv" style="color:{wr_color}">{s['win_rate']}%</div>
            </div>
            <div class="stat-box">
              <div class="sl">Profit Factor</div>
              <div class="sv" style="color:{pf_color}">{s['profit_factor']}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Max Drawdown</div>
              <div class="sv" style="color:{dd_color}">{s['max_drawdown']:.1f}%</div>
            </div>
            <div class="stat-box">
              <div class="sl">Sharpe</div>
              <div class="sv">{s['sharpe_ratio']}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Pire série</div>
              <div class="sv" style="color:#ff3a5c">{s['max_loss_streak']} pertes</div>
            </div>
          </div>

          <div class="stat-row" style="margin-top:8px">
            <div class="stat-box">
              <div class="sl">Capital initial</div>
              <div class="sv">${s['initial_capital']:,.0f}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Capital final</div>
              <div class="sv" style="color:{pnl_color}">${s['final_capital']:,.0f}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Avg Win</div>
              <div class="sv green">+${s['avg_win']:.2f}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Avg Loss</div>
              <div class="sv red">{s['avg_loss']:.2f}$</div>
            </div>
            <div class="stat-box">
              <div class="sl">Wins</div>
              <div class="sv green">{s['wins']}</div>
            </div>
            <div class="stat-box">
              <div class="sl">Losses</div>
              <div class="sv red">{s['losses']}</div>
            </div>
          </div>

          <div class="chart-wrap">
            <canvas id="{chart_id}"></canvas>
          </div>

          <div class="section-title">30 derniers trades</div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th><th>Dir</th><th>Entrée</th>
                  <th>SL</th><th>TP</th><th>Sortie</th>
                  <th>PnL%</th><th>PnL$</th><th>Raison</th>
                </tr>
              </thead>
              <tbody>{trade_rows}</tbody>
            </table>
          </div>
        </div>
        """

    # Build chart JS
    chart_js = ""
    for c in charts_data:
        labels = list(range(len(c["equity"])))
        data_json = json.dumps(c["equity"])
        chart_js += f"""
        new Chart(document.getElementById('{c["id"]}'), {{
          type: 'line',
          data: {{
            labels: {json.dumps(labels)},
            datasets: [{{
              label: 'Équité',
              data: {data_json},
              borderColor: '{c["color"]}',
              backgroundColor: '{c["color"]}18',
              borderWidth: 1.5,
              pointRadius: 0,
              fill: true,
              tension: 0.3,
            }}]
          }},
          options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
              x: {{ display: false }},
              y: {{
                grid: {{ color: '#1a2240' }},
                ticks: {{ color: '#4a5a80', callback: v => '$' + v.toLocaleString() }}
              }}
            }}
          }}
        }});
        """

    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Backtest Report — VWAP Strategy</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
    :root {{
      --bg:#060912; --panel:#0c1020; --border:#1a2240;
      --text:#c8d8ff; --muted:#4a5a80;
      --green:#00ffaa; --red:#ff3a5c; --gold:#ffc23a;
    }}
    body {{ font-family:-apple-system,sans-serif; background:var(--bg); color:var(--text);
            min-height:100vh; padding:30px 20px; }}
    .page-header {{ text-align:center; margin-bottom:40px; }}
    .page-title {{
      font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700;
      letter-spacing:4px; text-transform:uppercase;
      background:linear-gradient(90deg,#00e5ff,#3a8fff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }}
    .page-sub {{ font-family:'Share Tech Mono',monospace; font-size:.7rem; color:var(--muted); margin-top:6px; letter-spacing:.1em; }}
    .cards {{ display:flex; flex-direction:column; gap:30px; max-width:1100px; margin:0 auto; }}
    .card {{
      background:var(--panel); border:1px solid var(--border); border-radius:12px;
      padding:24px; position:relative; overflow:hidden;
    }}
    .card::before {{
      content:''; position:absolute; top:0; left:0; right:0; height:1px;
      background:linear-gradient(90deg,transparent,#00e5ff,transparent); opacity:.4;
    }}
    .card-header {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px; }}
    .card-title {{ font-family:'Rajdhani',sans-serif; font-size:1.4rem; font-weight:700; letter-spacing:2px; }}
    .card-sub {{ font-family:'Share Tech Mono',monospace; font-size:.65rem; color:var(--muted); margin-top:3px; }}
    .pnl-big {{ font-family:'Rajdhani',sans-serif; font-size:1.8rem; font-weight:700; }}
    .stat-row {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; }}
    .stat-box {{ background:#080d18; border:1px solid var(--border); border-radius:6px; padding:10px 12px; }}
    .sl {{ font-family:'Share Tech Mono',monospace; font-size:.58rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
    .sv {{ font-family:'Rajdhani',sans-serif; font-size:1.1rem; font-weight:700; margin-top:3px; }}
    .green {{ color:var(--green); }} .red {{ color:var(--red); }}
    .chart-wrap {{ height:180px; margin:20px 0; }}
    .section-title {{ font-family:'Share Tech Mono',monospace; font-size:.65rem; color:var(--muted);
                      text-transform:uppercase; letter-spacing:.1em; margin-bottom:10px; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:.78rem; }}
    th {{ font-family:'Share Tech Mono',monospace; font-size:.6rem; color:var(--muted);
          text-transform:uppercase; letter-spacing:.06em; padding:8px 6px; text-align:left;
          border-bottom:1px solid var(--border); }}
    td {{ padding:7px 6px; border-bottom:1px solid #0e1628; }}
    .badge {{ padding:2px 8px; border-radius:3px; font-family:'Rajdhani',sans-serif; font-size:.72rem; font-weight:700; }}
    .long  {{ background:rgba(0,255,170,.1); color:var(--green); border:1px solid rgba(0,255,170,.2); }}
    .short {{ background:rgba(255,58,92,.1);  color:var(--red);   border:1px solid rgba(255,58,92,.2); }}
    .tp-badge {{ padding:2px 6px; border-radius:3px; font-size:.68rem; font-weight:700;
                 background:rgba(0,255,170,.1); color:var(--green); }}
    .sl-badge {{ padding:2px 6px; border-radius:3px; font-size:.68rem; font-weight:700;
                 background:rgba(255,58,92,.1); color:var(--red); }}
    @media(max-width:700px) {{ .stat-row {{ grid-template-columns:repeat(3,1fr); }} }}
  </style>
</head>
<body>
  <div class="page-header">
    <div class="page-title">Backtest Report</div>
    <div class="page-sub">VWAP CROSSOVER STRATEGY · Généré le {now}</div>
  </div>
  <div class="cards">{cards_html}</div>
  <script>{chart_js}</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 Rapport HTML généré : {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest VWAP Strategy")
    parser.add_argument("--symbol",   default=None,    help="XAUUSD | US100 | US500 (défaut: tous)")
    parser.add_argument("--interval", default="1h",    help="5m | 15m | 1h | 4h | 1d (défaut: 1h)")
    parser.add_argument("--period",   default="180d",  help="Période yfinance ex: 60d, 180d (défaut: 180d)")
    parser.add_argument("--capital",  default=10000,   type=float, help="Capital initial en $ (défaut: 10000)")
    parser.add_argument("--risk",     default=1.0,     type=float, help="% risqué par trade (défaut: 1.0)")
    parser.add_argument("--output",   default="backtest_report.html", help="Chemin du rapport HTML")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else ["XAUUSD", "US100", "US500"]

    all_results = []
    for sym in symbols:
        res = run_backtest(
            symbol=sym,
            interval=args.interval,
            period=args.period,
            initial_capital=args.capital,
            risk_pct=args.risk,
        )
        if res:
            all_results.append(res)

    if all_results:
        report_path = generate_html_report(all_results, output_path=args.output)
        # Ouvrir automatiquement dans le navigateur
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")