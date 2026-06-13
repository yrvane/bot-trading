// Configuration
let chart = null;
let candlestickSeries = null;
let vwapSeries = null;
let ws = null;
let currentSymbol = 'SP500';

// Initialisation
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM chargé, initialisation...");
    initChart();
    initWebSocket();
    loadTradesHistory();
    loadPositions();
    
    document.getElementById('symbol').addEventListener('change', (e) => {
        currentSymbol = e.target.value;
        loadHistory();
        reconnectWebSocket();
    });
    
    document.getElementById('timeframe').addEventListener('change', () => {
        loadHistory();
    });
    
    // Rafraîchir périodiquement
    setInterval(() => {
        loadTradesHistory();
        loadPositions();
    }, 10000);
});

function initChart() {
    console.log("Initialisation du graphique...");
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error("Élément chart non trouvé !");
        return;
    }
    
    chart = LightweightCharts.createChart(chartElement, {
        width: chartElement.clientWidth,
        height: 500,
        layout: { background: { color: '#0a0e1a' }, textColor: '#e0e0e0' },
        grid: { vertLines: { color: '#1a1f2e' }, horzLines: { color: '#1a1f2e' } },
        timeScale: { borderColor: '#2a3040', timeVisible: true }
    });
    
    candlestickSeries = chart.addCandlestickSeries({
        upColor: '#00ff88',
        downColor: '#ff4444',
        borderVisible: false
    });
    
    vwapSeries = chart.addLineSeries({
        color: '#ffaa00',
        lineWidth: 2
    });
    
    chart.timeScale().fitContent();
    window.addEventListener('resize', () => {
        if (chart) {
            chart.applyOptions({ width: document.getElementById('chart').clientWidth });
        }
    });
    
    // Charger l'historique
    loadHistory();
}

function initWebSocket() {
    if (ws) ws.close();
    ws = new WebSocket(`ws://127.0.0.1:8000/ws/${currentSymbol}`);
    
    ws.onopen = () => {
        console.log("WebSocket connecté");
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log("Message WebSocket:", data.type);
            
            if (data.type === 'price') {
                updatePrice(data.price);
            } else if (data.type === 'new_trade') {
                showToast(`📈 Nouveau trade ${data.trade.type} ouvert à ${data.trade.entry_price.toFixed(2)}`, 'success');
                loadTradesHistory();
                loadPositions();
                updateSignalBox(data.trade.type, data.trade.entry_price);
            } else if (data.type === 'trade_closed') {
                const pnlColor = data.pnl >= 0 ? '✅' : '❌';
                showToast(`${pnlColor} Trade fermé (${data.reason}) | P&L: ${data.pnl.toFixed(2)}%`, 
                         data.pnl >= 0 ? 'success' : 'error');
                loadTradesHistory();
                loadPositions();
                resetSignalBox();
            } else if (data.type === 'position_update') {
                loadPositions();
            }
        } catch(e) {
            console.error("Erreur parsing message:", e);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function reconnectWebSocket() {
    initWebSocket();
}

function loadHistory() {
    console.log("Chargement de l'historique pour", currentSymbol);
    fetch(`/api/history/${currentSymbol}?bars=200`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error("Erreur historique:", data.error);
                return;
            }
            if (!data.dates || data.dates.length === 0) {
                console.warn("Pas de données historiques");
                return;
            }
            
            console.log(`Données reçues: ${data.dates.length} bougies`);
            
            const candlestickData = data.dates.map((date, i) => ({
                time: date,
                open: data.open[i],
                high: data.high[i],
                low: data.low[i],
                close: data.close[i]
            }));
            candlestickSeries.setData(candlestickData);
            
            const vwapData = data.dates.map((date, i) => ({
                time: date,
                value: data.vwap[i]
            }));
            vwapSeries.setData(vwapData);
            chart.timeScale().fitContent();
        })
        .catch(err => console.error("Erreur fetch historique:", err));
}

function loadTradesHistory() {
    console.log("Chargement de l'historique des trades");
    fetch('/api/trades')
        .then(res => res.json())
        .then(trades => {
            const container = document.getElementById('tradesList');
            if (!container) {
                console.error("Container tradesList non trouvé");
                return;
            }
            
            if (!trades || trades.length === 0) {
                container.innerHTML = '<div class="trade-row">Aucun trade</div>';
                return;
            }
            
            let wins = 0;
            let totalPnl = 0;
            
            container.innerHTML = trades.slice(-10).reverse().map(trade => {
                if (trade.pnl > 0) wins++;
                totalPnl += trade.pnl;
                const pnlClass = trade.pnl > 0 ? 'trade-win' : (trade.pnl < 0 ? 'trade-loss' : '');
                const date = new Date(trade.date).toLocaleString();
                const exitPrice = trade.exit_price && trade.exit_price > 0 ? trade.exit_price.toFixed(2) : 'En cours';
                const pnlDisplay = trade.pnl !== 0 ? `${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}%` : 'En cours';
                
                return `
                    <div class="trade-row">
                        <span>${date}</span>
                        <span>${trade.symbol}</span>
                        <span class="${pnlClass}">${trade.type}</span>
                        <span>${trade.entry_price.toFixed(2)}</span>
                        <span>${exitPrice}</span>
                        <span class="${pnlClass}">${pnlDisplay}</span>
                    </div>
                `;
            }).join('');
            
            const winRateEl = document.getElementById('winRate');
            const totalTradesEl = document.getElementById('totalTrades');
            const totalPnlEl = document.getElementById('totalPnl');
            
            if (winRateEl) winRateEl.textContent = trades.length ? `${((wins / trades.length) * 100).toFixed(0)}%` : '--';
            if (totalTradesEl) totalTradesEl.textContent = trades.length;
            if (totalPnlEl) totalPnlEl.textContent = `${totalPnl > 0 ? '+' : ''}${totalPnl.toFixed(2)}%`;
        })
        .catch(err => console.error("Erreur fetch trades:", err));
}

function loadPositions() {
    fetch('/api/positions')
        .then(res => res.json())
        .then(positions => {
            const container = document.getElementById('positionsList');
            if (!container) return;
            
            if (!positions || positions.length === 0) {
                container.innerHTML = '<div class="trade-row">Aucune position ouverte</div>';
                return;
            }
            
            container.innerHTML = positions.map(pos => `
                <div class="trade-row">
                    <span>${pos.symbol}</span>
                    <span class="${pos.type === 'LONG' ? 'trade-win' : 'trade-loss'}">${pos.type}</span>
                    <span>Entrée: ${pos.entry_price.toFixed(2)}</span>
                    <span>SL: ${pos.sl.toFixed(2)}</span>
                    <span>TP: ${pos.tp.toFixed(2)}</span>
                    <span>🔴 Ouvert</span>
                </div>
            `).join('');
        })
        .catch(err => console.error("Erreur fetch positions:", err));
}

function updatePrice(price) {
    const priceEl = document.getElementById('currentPrice');
    if (priceEl) priceEl.textContent = price.toFixed(2);
    
    // Récupérer le VWAP depuis l'historique
    fetch(`/api/history/${currentSymbol}?bars=1`)
        .then(res => res.json())
        .then(data => {
            if (data.vwap && data.vwap.length > 0) {
                const vwap = data.vwap[data.vwap.length - 1];
                const vwapEl = document.getElementById('vwapValue');
                if (vwapEl) vwapEl.textContent = vwap.toFixed(2);
                
                const change = price - vwap;
                const changeEl = document.getElementById('priceChange');
                if (changeEl) {
                    changeEl.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}`;
                    changeEl.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                }
            }
        })
        .catch(err => console.error("Erreur fetch VWAP:", err));
}

function updateSignalBox(signal, price) {
    const signalBox = document.getElementById('signalBox');
    if (!signalBox) return;
    signalBox.className = `signal-box ${signal === 'LONG' ? 'long' : 'short'}`;
    signalBox.innerHTML = `
        <strong>${signal === 'LONG' ? '📈 POSITION LONGUE OUVERTE' : '📉 POSITION SHORT OUVERTE'}</strong><br>
        Prix entrée: ${price.toFixed(2)}<br>
        Suivi automatique en cours...
    `;
}

function resetSignalBox() {
    const signalBox = document.getElementById('signalBox');
    if (!signalBox) return;
    signalBox.className = 'signal-box no-signal';
    signalBox.innerHTML = '<span>⏸️ En attente de signal</span>';
}

function showToast(message, type) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: ${type === 'success' ? '#00ff88' : '#ff4444'};
        color: ${type === 'success' ? '#0a0e1a' : 'white'};
        padding: 12px 20px;
        border-radius: 10px;
        z-index: 1000;
        animation: fadeInOut 3s ease-in-out;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// CSS animation pour le toast
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateX(100px); }
        15% { opacity: 1; transform: translateX(0); }
        85% { opacity: 1; transform: translateX(0); }
        100% { opacity: 0; transform: translateX(100px); }
    }
    .trade-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 15px;
        border-bottom: 1px solid #2a3040;
        font-size: 0.85rem;
    }
    .trade-win { color: #00ff88; }
    .trade-loss { color: #ff4444; }
    .positions-section, .trades-section {
        margin-top: 20px;
        background: #1a1f2e;
        border-radius: 16px;
        padding: 15px;
        border: 1px solid #2a3040;
    }
    .positions-section h3, .trades-section h3 {
        margin-bottom: 15px;
        font-size: 1rem;
    }
`;
document.head.appendChild(style);