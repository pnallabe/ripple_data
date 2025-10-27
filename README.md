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

### Dashboard Mode
Launch the interactive web dashboard:
```bash
python main.py --mode dashboard --port 8050
```

Navigate to `http://localhost:8050` to access the platform.

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

## Quick Start Example

```python
from src.analytics import RipplePropagator

# Initialize propagator
propagator = RipplePropagator()

# Run simulation
results = propagator.simulate_shock_propagation(
    seed_ticker='AAPL',
    shock_magnitude=-0.05,  # 5% negative shock
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    damping_factor=0.85
)

print(results[['ticker', 'final_impact', 'cumulative_impact']].head())
```

## License

This project is licensed under the MIT License.

---

**Note**: This is a research and educational platform. Not intended for production trading without proper risk management and compliance review.