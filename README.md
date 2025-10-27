# Stock Dependency & Ripple Effect Analysis Platform

A comprehensive platform for analyzing stock interdependencies and simulating ripple effects across financial markets.

## Overview

This platform implements a multi-layer dependency network for stocks with:
- **Nodes**: Stocks/entities
- **Edges**: Ownership, correlation, sector, supplier/customer, sentiment relationships
- **Weights**: Normalized strength of dependencies
- **Propagation**: Matrix-based simulation with damping factors

## Features

### Data Ingestion
- **EDGAR Parser**: SEC filings for ownership relationships
- **Market Data**: Real-time and historical price data via Yahoo Finance/Alpha Vantage
- **News & Sentiment**: News articles and sentiment analysis integration

### Analytics Engine
- **Correlation Analysis**: Rolling correlations, Granger causality, partial correlations
- **Graph Construction**: Multi-layer dependency networks in Neo4j
- **Ripple Simulation**: Matrix-based propagation with configurable damping
- **Risk Metrics**: Systemic risk scoring and centrality measures

### Visualization
- **Interactive Dashboard**: Dash-based web application
- **Network Graphs**: Real-time dependency visualization
- **Impact Analysis**: Heatmaps, sector exposure, time series
- **Simulation UI**: Configure shocks and view propagation results

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Neo4j 4.4+
- Redis (optional, for caching)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ripple_data
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Database setup**:
   ```bash
   # PostgreSQL
   psql -d your_database -f postgres_schema_ripple.sql
   
   # Neo4j
   # Run neo4j_import_ripple.cypher in Neo4j Browser
   ```

4. **Configuration**:
   Create a `.env` file:
   ```env
   # Database
   POSTGRES_HOST=localhost
   POSTGRES_DB=ripple_db
   POSTGRES_USER=your_user
   POSTGRES_PASSWORD=your_password
   
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   
   # APIs
   ALPHA_VANTAGE_API_KEY=your_key
   NEWS_API_KEY=your_key
   
   # Analytics
   DEFAULT_DAMPING_FACTOR=0.85
   CORRELATION_WINDOWS=[30,90,180]
   ```

## Usage

### Enhanced Features

#### 🎛️ Interactive Dashboard
- **Real-time Network Visualization**: Dynamic network graphs with 45 financial institutions
- **Advanced Control Panel**: Intuitive parameter controls with preset scenarios  
- **Multi-tab Interface**: Dashboard, 🧪 **Advanced Simulation**, Correlations, Sectors, Analytics, System monitoring
- **Statistics Header**: Live platform metrics and data coverage information
- **Responsive Design**: Modern Bootstrap 5.1.3 styling with Font Awesome icons

#### 🧪 **NEW: Advanced Simulation Laboratory**
- **5 Simulation Types**: Matrix Propagation, Monte Carlo, Stress Testing, Scenario Analysis, Systemic Risk
- **Interactive Configuration**: Real-time parameter adjustment with 6 predefined scenarios
- **Advanced Visualizations**: Impact waterfalls, confidence intervals, risk heatmaps, sector analysis
- **Scenario Management**: Save, load, and manage custom stress test scenarios
- **Risk Analytics**: VaR/CVaR, concentration indices, diversification ratios
- **Export Capabilities**: JSON, CSV, Excel export with comprehensive reporting

#### 📊 Comprehensive Analytics  
- **Shock Propagation Simulation**: Matrix-based ripple effect modeling with enhanced convergence
- **Network Analysis**: 3,942+ correlation relationships in Neo4j with centrality metrics
- **Monte Carlo Simulations**: Uncertainty quantification with up to 10,000 runs
- **Portfolio Impact Analysis**: Market cap weighted impact assessment
- **Systemic Risk Assessment**: Institution-level importance ranking and risk scoring
- **Real-time Updates**: 30-second refresh intervals for live data integration

**💡 Quick Start Examples:**
- Banking Crisis: Select JPM, set -5% shock, click "Run Simulation"
- Payment Disruption: Use "💳 Payment Shock" preset button
- Cross-Sector Analysis: Compare technology vs financial correlations
- Export Analysis: Use download buttons on charts and tables

### Data Ingestion
Ingest market data for specific tickers:
```bash
# Technology stocks
python main.py --mode ingest --tickers AAPL MSFT GOOGL AMZN TSLA NVDA

# Major banks
python main.py --mode ingest --tickers JPM BAC WFC C GS MS

# Payment processors
python main.py --mode ingest --tickers V MA AXP

# Insurance companies
python main.py --mode ingest --tickers AIG MET PRU AFL

# Asset management
python main.py --mode ingest --tickers BLK SCHW BRK-B
```

### Correlation Analysis
Compute correlations between financial services stocks:
```bash
# Banking sector analysis
python main.py --mode analyze --tickers JPM BAC WFC C GS MS

# Payment systems correlation
python main.py --mode analyze --tickers V MA AXP

# Cross-sector analysis
python main.py --mode analyze --tickers JPM V AAPL BLK AIG
```

### Ripple Simulation
Simulate shock propagation across financial networks:
```bash
# Banking crisis simulation
python main.py --mode simulate --seed-ticker JPM --shock -0.05

# Payment system disruption
python main.py --mode simulate --seed-ticker V --shock -0.03

# Market-wide technology shock
python main.py --mode simulate --seed-ticker AAPL --shock -0.08
```

### ETL Pipeline
Run the complete data pipeline:
```bash
python scripts/etl_pipeline.py
```

### Financial Services Pipeline Test
Test the comprehensive financial services data pipeline:
```bash
python test_financial_pipeline.py
```

### Run Tests
Execute the test suite:
```bash
python tests/test_platform.py
```

## Current Database Status

Our platform now includes comprehensive financial services data:

**📊 Database Statistics:**
- **Total Stocks**: 45 tickers across multiple sectors  
- **Total Records**: 11,205 price records
- **Data Coverage**: 1 year (Oct 2024 - Oct 2025)

**🏦 Financial Services Coverage (39 stocks):**
- **Major Banks**: JPM, BAC, WFC, C, GS, MS, USB, TFC, PNC, COF
- **Regional Banks**: RF, KEY, FITB, CFG, STT, ZION, NTRS, BK, BBT, SYF  
- **Payment Systems**: V, MA, AXP
- **Asset Management**: BLK, SCHW, BRK-B
- **Insurance**: AIG, MET, PRU, AFL, MMC, AON, ALL, TRV, PGR, CB
- **Exchanges & Data**: CME, ICE, SPGI

**💻 Technology Stocks (6 stocks):**
- AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA

**🔗 Correlation Analysis:**
- 3,942+ correlation relationships computed
- Rolling 30-day correlation windows
- Cross-sector dependency mapping

**🌊 Ripple Simulation Capabilities:**
- Multi-sector shock propagation modeling
- Banking system stress testing
- Payment network disruption analysis
- Technology sector impact assessment

## Project Structure

```
ripple_data/
├── config/
│   └── settings.py          # Configuration management
├── src/
│   ├── analytics/           # Correlation, propagation, graph building
│   ├── data_ingestion/      # EDGAR, market data, news loaders
│   ├── database/           # PostgreSQL, Neo4j, Redis managers
│   ├── visualization/      # Dash dashboard
│   └── utils.py           # Utility functions
├── scripts/
│   └── etl_pipeline.py    # Automated data pipeline
├── tests/
│   └── test_platform.py  # Test suite
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
├── postgres_schema_ripple.sql    # Database schema
├── neo4j_import_ripple.cypher   # Graph database setup
└── README.md
```

## 🚀 Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd ripple_data
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp config/settings.py.template config/settings.py
   # Edit database connections in config/settings.py
   ```

3. **Initialize Data**:
   ```bash
   python main.py --mode setup
   python main.py --mode etl
   ```

4. **Launch Enhanced Dashboard**:
   ```bash
   python main.py --mode dashboard --port 8055
   ```

5. **Access Interface**: 
   - Open [http://localhost:8055](http://localhost:8055)
   - Navigate to the 🧪 **Simulation** tab for advanced features
   - Try predefined scenarios: Bank Crisis, Tech Shock, Monte Carlo Analysis

## 📱 Enhanced Dashboard Usage

### Main Interface
- **🎛️ Dashboard**: Primary simulation interface with network visualization
- **🧪 Simulation**: **NEW** Advanced simulation laboratory with 5 simulation types
- **🔗 Correlations**: Real-time correlation matrix and relationship analysis  
- **🏦 Sectors**: Financial services sector breakdown and performance metrics
- **📊 Analytics**: Advanced risk metrics and statistical analysis
- **⚙️ System**: Platform status, database statistics, and configuration

### Advanced Simulation Laboratory
1. **Choose Simulation Type**: 
   - Matrix Propagation (traditional)
   - Monte Carlo (uncertainty analysis)
   - Stress Testing (multiple shock levels)
   - Scenario Analysis (cross-scenario comparison)
   - Systemic Risk Assessment (comprehensive ranking)

2. **Configure Parameters**: 
   - Seed ticker from 45 financial institutions
   - Shock magnitude (-50% to -1%)
   - Damping factor (0.1 to 0.99)
   - Monte Carlo runs (100 to 10,000)

3. **Use Predefined Scenarios**: 6 expert-designed scenarios
   - 🏦 Major Bank Crisis
   - 💻 Technology Sector Shock  
   - 🛡️ Insurance Sector Crisis
   - 💳 Payment System Disruption
   - 📉 Market Correction
   - 🔍 Systemic Risk Assessment

4. **Analyze Results**: Multiple visualization options
   - 📊 Impact Waterfall Charts
   - 📈 Statistical Distribution Analysis
   - 🌡️ Risk Intensity Heatmaps
   - 🏢 Sector Impact Breakdown
   - 🎲 Monte Carlo Confidence Intervals
   - 📊 Risk Metrics Dashboard

5. **Export & Save**: Comprehensive data export and scenario management

## API Usage Example

```python
# Traditional Analytics
from src.analytics import RipplePropagator

propagator = RipplePropagator()
results = propagator.simulate_shock_propagation(
    seed_ticker='JPM',
    shock_magnitude=-0.05,
    damping_factor=0.85
)

# Enhanced Simulation Engine
from src.simulation import SimulationEngine, SimulationConfig, SimulationType

engine = SimulationEngine()
config = SimulationConfig(
    simulation_type=SimulationType.MONTE_CARLO,
    seed_ticker='JPM',
    shock_magnitude=-0.15,
    damping_factor=0.80,
    monte_carlo_runs=2000
)

results = engine.run_simulation(config)
print(f"Simulation: {len(results.results_df)} stocks analyzed")
print(f"Execution time: {results.execution_time:.2f}s")
print(f"Mean impact: {results.statistics['mean_impact']:.2%}")
```

## Quick Start Example

## License

This project is licensed under the MIT License.

---

**Note**: This is a research and educational platform. Not intended for production trading without proper risk management and compliance review.