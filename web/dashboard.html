<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .status-bar {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 10px;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 16px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.healthy { background: #4ade80; }
        .status-dot.degraded { background: #fbbf24; }
        .status-dot.unhealthy { background: #ef4444; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .card h3 {
            margin-bottom: 15px;
            color: #4f46e5;
            font-size: 1.2rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .kpi-item {
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            border-radius: 10px;
            border-left: 4px solid #4f46e5;
        }

        .kpi-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1e293b;
            margin-bottom: 5px;
        }

        .kpi-label {
            font-size: 0.9rem;
            color: #64748b;
        }

        .kpi-positive { border-left-color: #10b981; }
        .kpi-negative { border-left-color: #ef4444; }

        .strategy-list {
            list-style: none;
        }

        .strategy-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
            border-radius: 8px;
            border-left: 4px solid #6366f1;
        }

        .strategy-rank {
            background: #4f46e5;
            color: white;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9rem;
        }

        .strategy-name {
            font-weight: 600;
            text-transform: capitalize;
        }

        .strategy-score {
            font-weight: bold;
            color: #1e293b;
        }

        .regime-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .regime-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .regime-bull {
            background: #dcfce7;
            color: #166534;
        }

        .regime-bear {
            background: #fef2f2;
            color: #991b1b;
        }

        .regime-range {
            background: #fefce8;
            color: #a16207;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 15px;
        }

        .news-item {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 6px;
            border-left: 3px solid #e2e8f0;
        }

        .news-positive { border-left-color: #10b981; background: #f0fdf4; }
        .news-negative { border-left-color: #ef4444; background: #fef2f2; }
        .news-neutral { border-left-color: #6b7280; background: #f9fafb; }

        .news-title {
            font-weight: 600;
            margin-bottom: 4px;
            font-size: 0.9rem;
        }

        .news-source {
            font-size: 0.8rem;
            color: #6b7280;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }

        .error {
            background: #fef2f2;
            color: #991b1b;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ef4444;
        }

        .refresh-info {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            margin-top: 20px;
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .kpi-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🤖 Trading Bot Dashboard</h1>
            <div class="status-bar">
                <div class="status-indicator">
                    <div class="status-dot healthy" id="statusDot"></div>
                    <span id="systemStatus">Connecting...</span>
                </div>
                <div class="status-indicator">
                    <span id="lastUpdate">Never</span>
                </div>
                <div class="status-indicator">
                    <span id="alertCount">0 alerts</span>
                </div>
            </div>
        </div>

        <div class="grid" id="mainGrid">
            <!-- KPI Card -->
            <div class="card">
                <h3>📊 Key Performance Indicators</h3>
                <div class="kpi-grid" id="kpiGrid">
                    <div class="loading">Loading KPIs...</div>
                </div>
            </div>

            <!-- Strategy Rankings -->
            <div class="card">
                <h3>🏆 Top Strategies</h3>
                <ul class="strategy-list" id="strategyList">
                    <li class="loading">Loading strategies...</li>
                </ul>
            </div>

            <!-- Market Regime -->
            <div class="card">
                <h3>📈 Market Regime</h3>
                <div id="regimeContent">
                    <div class="loading">Loading regime data...</div>
                </div>
            </div>

            <!-- News Sentiment -->
            <div class="card">
                <h3>📰 Recent Headlines</h3>
                <div id="newsContent">
                    <div class="loading">Loading news...</div>
                </div>
            </div>
        </div>

        <!-- P&L Chart -->
        <div class="card">
            <h3>💰 P&L Trend</h3>
            <div class="chart-container">
                <canvas id="pnlChart"></canvas>
            </div>
        </div>

        <div class="refresh-info">
            🔄 Auto-refreshing every 30 seconds
        </div>
    </div>

    <script>
        class TradingDashboard {
            constructor() {
                this.apiUrl = 'http://127.0.0.1:8000';
                this.refreshInterval = 30000; // 30 seconds
                this.pnlChart = null;
                this.pnlData = [];
                
                this.init();
            }

            init() {
                this.setupChart();
                this.loadData();
                
                // Auto-refresh
                setInterval(() => {
                    this.loadData();
                }, this.refreshInterval);
            }

            async loadData() {
                try {
                    const response = await fetch(`${this.apiUrl}/status`);
                    const data = await response.json();
                    
                    this.updateSystemStatus(data);
                    this.updateKPIs(data.kpis);
                    this.updateStrategies(data.top_strategies);
                    this.updateRegime(data.intelligence.market_regime);
                    this.updateNews(data.intelligence.news_sentiment);
                    this.updatePnLChart(data.kpis);
                    
                    document.getElementById('lastUpdate').textContent = 
                        new Date().toLocaleTimeString();
                        
                } catch (error) {
                    this.showError('Failed to load data: ' + error.message);
                }
                
                // Load alerts separately
                this.loadAlerts();
            }

            async loadAlerts() {
                try {
                    const response = await fetch(`${this.apiUrl}/alerts`);
                    const data = await response.json();
                    
                    const alertCount = data.count;
                    document.getElementById('alertCount').textContent = 
                        `${alertCount} alert${alertCount !== 1 ? 's' : ''}`;
                        
                } catch (error) {
                    console.warn('Failed to load alerts:', error);
                }
            }

            updateSystemStatus(data) {
                const statusDot = document.getElementById('statusDot');
                const statusText = document.getElementById('systemStatus');
                
                const status = data.system_status || 'unknown';
                
                statusDot.className = `status-dot ${status}`;
                statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            }

            updateKPIs(kpis) {
                const kpiGrid = document.getElementById('kpiGrid');
                
                if (kpis.error) {
                    kpiGrid.innerHTML = `<div class="error">${kpis.error}</div>`;
                    return;
                }

                const kpiItems = [
                    { 
                        label: 'Gross P&L', 
                        value: this.formatCurrency(kpis.gross_pnl || 0),
                        class: (kpis.gross_pnl || 0) >= 0 ? 'kpi-positive' : 'kpi-negative'
                    },
                    { 
                        label: 'Win Rate', 
                        value: `${(kpis.win_rate || 0).toFixed(1)}%`,
                        class: (kpis.win_rate || 0) >= 50 ? 'kpi-positive' : 'kpi-negative'
                    },
                    { 
                        label: 'Profit Factor', 
                        value: kpis.profit_factor || '0.00',
                        class: (kpis.profit_factor || 0) >= 1 ? 'kpi-positive' : 'kpi-negative'
                    },
                    { 
                        label: 'Total Trades', 
                        value: kpis.total_trades || 0,
                        class: ''
                    }
                ];

                kpiGrid.innerHTML = kpiItems.map(item => `
                    <div class="kpi-item ${item.class}">
                        <div class="kpi-value">${item.value}</div>
                        <div class="kpi-label">${item.label}</div>
                    </div>
                `).join('');
            }

            updateStrategies(strategies) {
                const strategyList = document.getElementById('strategyList');
                
                if (!strategies || strategies.length === 0) {
                    strategyList.innerHTML = '<li class="error">No strategy data available</li>';
                    return;
                }

                strategyList.innerHTML = strategies.map(strategy => `
                    <li class="strategy-item">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div class="strategy-rank">${strategy.rank}</div>
                            <div class="strategy-name">${strategy.strategy}</div>
                        </div>
                        <div class="strategy-score">${strategy.score.toFixed(1)}</div>
                    </li>
                `).join('');
            }

            updateRegime(regimeData) {
                const regimeContent = document.getElementById('regimeContent');
                
                if (!regimeData || !regimeData.available) {
                    regimeContent.innerHTML = '<div class="error">Regime data not available</div>';
                    return;
                }

                const { bull_count, bear_count, range_count, regime_details } = regimeData;
                
                let badgesHtml = '';
                if (bull_count > 0) badgesHtml += `<span class="regime-badge regime-bull">🐂 ${bull_count} Bull</span>`;
                if (bear_count > 0) badgesHtml += `<span class="regime-badge regime-bear">🐻 ${bear_count} Bear</span>`;
                if (range_count > 0) badgesHtml += `<span class="regime-badge regime-range">📦 ${range_count} Range</span>`;

                let detailsHtml = '';
                if (regime_details) {
                    detailsHtml = Object.entries(regime_details).map(([symbol, info]) => {
                        const confidence = (info.confidence * 100).toFixed(0);
                        const regimeEmoji = info.regime === 'bull' ? '🐂' : info.regime === 'bear' ? '🐻' : '📦';
                        return `
                            <div style="margin: 8px 0; padding: 8px; background: #f8fafc; border-radius: 6px;">
                                ${regimeEmoji} <strong>${symbol}</strong>: ${info.regime} (${confidence}% confidence)
                            </div>
                        `;
                    }).join('');
                }

                regimeContent.innerHTML = `
                    <div class="regime-badges">${badgesHtml}</div>
                    <div style="margin-top: 15px;">${detailsHtml}</div>
                `;
            }

            updateNews(newsData) {
                const newsContent = document.getElementById('newsContent');
                
                if (!newsData || !newsData.available) {
                    newsContent.innerHTML = '<div class="error">News data not available</div>';
                    return;
                }

                const headlines = newsData.recent_headlines || [];
                
                if (headlines.length === 0) {
                    newsContent.innerHTML = '<div>No recent headlines</div>';
                    return;
                }

                newsContent.innerHTML = headlines.map(headline => {
                    const sentiment = headline.sentiment_compound;
                    const sentimentClass = sentiment > 0.1 ? 'news-positive' : 
                                         sentiment < -0.1 ? 'news-negative' : 'news-neutral';
                    
                    return `
                        <div class="news-item ${sentimentClass}">
                            <div class="news-title">${headline.title}</div>
                            <div class="news-source">${headline.source} • ${sentiment.toFixed(2)}</div>
                        </div>
                    `;
                }).join('');
            }

            updatePnLChart(kpis) {
                if (!kpis || kpis.error) return;

                const currentPnL = kpis.gross_pnl || 0;
                const timestamp = new Date().toLocaleTimeString();

                // Add new data point
                this.pnlData.push({
                    time: timestamp,
                    pnl: currentPnL
                });

                // Keep only last 20 points
                if (this.pnlData.length > 20) {
                    this.pnlData.shift();
                }

                // Update chart
                if (this.pnlChart) {
                    this.pnlChart.data.labels = this.pnlData.map(d => d.time);
                    this.pnlChart.data.datasets[0].data = this.pnlData.map(d => d.pnl);
                    this.pnlChart.update('none'); // No animation for real-time updates
                }
            }

            setupChart() {
                const ctx = document.getElementById('pnlChart').getContext('2d');
                
                this.pnlChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'P&L ($)',
                            data: [],
                            borderColor: '#4f46e5',
                            backgroundColor: 'rgba(79, 70, 229, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.1)'
                                }
                            },
                            x: {
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.1)'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        }
                    }
                });
            }

            formatCurrency(amount) {
                return new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD'
                }).format(amount);
            }

            showError(message) {
                const mainGrid = document.getElementById('mainGrid');
                mainGrid.innerHTML = `<div class="error">${message}</div>`;
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new TradingDashboard();
        });
    </script>
</body>
</html>