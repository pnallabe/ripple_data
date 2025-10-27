# System Design Document (SDD)

**Title:** Stock Dependency & Ripple Effect Analysis Platform  
**Version:** 1.0  
**Author:** Pradeep Nallabelli  
**Date:** October 2025

## 1. Purpose
Analyze interdependencies among publicly traded stocks and simulate ripple effects caused by shocks (price, news, sentiment) in one stock across others.

## 2. Scope
**In Scope**
- Ingest EDGAR, market APIs, and news/sentiment.
- Extract corporate ownership, compute statistical dependencies, and build a multi-layer graph.
- Simulate ripple propagation and visualize results.

**Out of Scope**
- Trading execution, order management, proprietary ML beyond PoC.

## 3. System Overview
A multi-layer dependency network:
- Nodes = stocks/entities
- Edges = ownership, correlation, sector, supplier/customer, sentiment
- Weights = strength of dependency (normalized)

Propagation model uses adjacency matrix multiplication with damping.

## 4. Architecture Overview
Logical layers:
- Data Sources (EDGAR, market APIs, news)
- Data Ingestion & ETL
- Storage: PostgreSQL (time-series, metadata), Neo4j (graph)
- Analytics: Correlation, propagation, ML (VAR/GNN)
- Visualization: Dash/Streamlit + interactive graph

## 5. Components
- **Data Ingestion:** EDGAR Parser, Market Data Ingestor, News/Sentiment Loader
- **Data Storage:** PostgreSQL (companies, prices, events, edges_staging), Neo4j (Company nodes, relationships)
- **Analytics:** Correlation matrices, adjacency construction, matrix-based propagation, GNN/VAR models for learned influence
- **Visualization:** Interactive dashboards, heatmaps, network explorer

## 6. Data Flow
EDGAR / APIs → Ingestion ETL → PostgreSQL (raw & timeseries) → Neo4j (relationships) → Analytics Engine → Dashboard

## 7. Propagation Model (simplified)
Let A be the adjacency/influence matrix. Given initial shock ΔP₀:
ΔP_{t+1} = α * A * ΔP_t
Stop when converged or after fixed iterations. α is damping factor.

## 8. Database Schema (high-level)
- companies(company_id, ticker, name, sector, cik, market_cap, ...)
- prices(price_id, company_id, ticker, trade_date, open, close, adj_close, volume, source)
- events(event_id, company_id, ticker, event_type, event_date, magnitude, headline, body)
- edges_staging(edge_id, source_ticker, target_ticker, relation_type, weight, info)
- ticker_map(ticker, company_id)

## 9. Neo4j Graph Model
- Node: :Company {ticker, name, sector, market_cap, ...}
- Relationships: :OWNS, :CORRELATED_WITH, :INFLUENCES, :SUPPLIER_OF, :CUSTOMER_OF, :SENTIMENT_CORR
- Relationship properties: weight, source, source_date, meta

## 10. Analytics & Modeling
- Rolling correlations (30/90/180d)
- Granger causality / partial correlations for directionality
- VAR or GNN models for learned influence
- Metrics: total propagated market-cap impact, top affected nodes, centrality measures

## 11. Visualization
- Ripple simulation UI (select ticker, shock size, damping)
- Interactive network graph and top-k impact table
- Sector exposure heatmap and time slider for rolling snapshots

## 12. Security & Deployment
- Secrets in AWS Secrets Manager, role-based access, HTTPS
- Containerized (Docker), deployable on AWS/GCP
- Use Redis for caching and Neo4j Aura or self-managed Neo4j for graph DB

## 13. Next Steps
- Prototype: ingest top 20 financial tickers, compute correlation graph, push to Neo4j, run propagation
- Implement ETL, schedule nightly jobs, build dashboard