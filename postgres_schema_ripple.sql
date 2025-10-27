-- PostgreSQL DDL for Stock Dependency & Ripple Effect Analysis Platform
-- Run in psql or via a Postgres client. Assumes database already created.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    company_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker TEXT NOT NULL,
    exchange TEXT,
    name TEXT NOT NULL,
    cik TEXT,
    isin TEXT,
    cusip TEXT,
    sector TEXT,
    industry TEXT,
    country TEXT,
    market_cap BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_companies_ticker ON companies (ticker);
CREATE INDEX IF NOT EXISTS ix_companies_sector ON companies (sector);
CREATE INDEX IF NOT EXISTS ix_companies_name ON companies USING gin (to_tsvector('english', name));

-- Prices table
CREATE TABLE IF NOT EXISTS prices (
    price_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(company_id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open NUMERIC(18,6),
    high NUMERIC(18,6),
    low NUMERIC(18,6),
    close NUMERIC(18,6),
    adj_close NUMERIC(18,6),
    volume BIGINT,
    source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_prices_ticker_date ON prices (ticker, trade_date);
CREATE INDEX IF NOT EXISTS ix_prices_company_date ON prices (company_id, trade_date DESC);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(company_id),
    ticker TEXT,
    event_type TEXT,
    event_date TIMESTAMP WITH TIME ZONE,
    magnitude NUMERIC,
    headline TEXT,
    body TEXT,
    source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_events_company_date ON events (company_id, event_date DESC);
CREATE INDEX IF NOT EXISTS ix_events_ticker_date ON events (ticker, event_date DESC);

-- Edges staging table
CREATE TABLE IF NOT EXISTS edges_staging (
    edge_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_ticker TEXT NOT NULL,
    target_ticker TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight DOUBLE PRECISION,
    info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_edges_src_tgt ON edges_staging (source_ticker, target_ticker);
CREATE INDEX IF NOT EXISTS ix_edges_type ON edges_staging (relation_type);

-- Ticker map
CREATE TABLE IF NOT EXISTS ticker_map (
    ticker TEXT PRIMARY KEY,
    company_id UUID REFERENCES companies(company_id)
);

-- Upsert function (example)
CREATE OR REPLACE FUNCTION upsert_company(p_ticker TEXT, p_name TEXT)
RETURNS UUID LANGUAGE plpgsql AS $$
DECLARE
    v_company_id UUID;
BEGIN
    SELECT company_id INTO v_company_id FROM companies WHERE ticker = p_ticker;
    IF v_company_id IS NULL THEN
        INSERT INTO companies (ticker, name) VALUES (p_ticker, p_name) RETURNING company_id INTO v_company_id;
    ELSE
        UPDATE companies SET name = p_name, updated_at = now() WHERE company_id = v_company_id;
    END IF;
    RETURN v_company_id;
END;
$$;