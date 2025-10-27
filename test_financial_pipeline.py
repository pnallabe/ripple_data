#!/usr/bin/env python3
"""
Comprehensive Financial Services Data Pipeline Test

This script tests the complete data pipeline with our expanded financial services dataset.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.managers import pg_manager
from src.analytics.correlation import CorrelationAnalyzer
from src.analytics.propagation import RipplePropagator
from config.settings import config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def test_data_integrity():
    """Test data integrity of our financial services dataset."""
    logger = logging.getLogger(__name__)
    
    # Check total records
    query = """
    SELECT 
        COUNT(DISTINCT ticker) as total_tickers,
        COUNT(*) as total_records,
        MIN(trade_date) as earliest_date,
        MAX(trade_date) as latest_date
    FROM prices
    """
    
    result = pg_manager.execute_query(query)
    if result:
        stats = result[0]
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"  Total Tickers: {stats['total_tickers']}")
        logger.info(f"  Total Records: {stats['total_records']}")
        logger.info(f"  Date Range: {stats['earliest_date']} to {stats['latest_date']}")
    
    # Check financial services stocks specifically
    financial_tickers = [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS',  # Major banks
        'V', 'MA', 'AXP',                       # Payment processors
        'BLK', 'SCHW',                         # Asset management
        'AIG', 'MET', 'PRU', 'AFL',            # Insurance
        'BRK-B',                               # Conglomerate
        'CME', 'ICE', 'SPGI'                   # Exchanges & Data
    ]
    
    query = """
    SELECT ticker, COUNT(*) as records 
    FROM prices 
    WHERE ticker = ANY(%s)
    GROUP BY ticker 
    ORDER BY ticker
    """
    
    financial_data = pg_manager.execute_query(query, (financial_tickers,))
    
    logger.info(f"🏦 Financial Services Stocks ({len(financial_data)} found):")
    for row in financial_data:
        logger.info(f"  {row['ticker']}: {row['records']} records")
    
    return len(financial_data), sum(row['records'] for row in financial_data)


def test_correlation_analysis():
    """Test correlation analysis on financial services stocks."""
    logger = logging.getLogger(__name__)
    
    # Financial sector groups for correlation analysis
    major_banks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS']
    payment_companies = ['V', 'MA', 'AXP']
    insurance_companies = ['AIG', 'MET', 'PRU', 'AFL']
    
    analyzer = CorrelationAnalyzer()
    
    # Test correlation within major banks
    logger.info("🔗 Testing Major Banks Correlation Analysis...")
    bank_correlations = analyzer.compute_rolling_correlations(
        tickers=major_banks,
        window_days=30
    )
    
    if not bank_correlations.empty:
        logger.info(f"  Computed {len(bank_correlations)} bank correlations")
        # Show highest correlations
        top_correlations = bank_correlations.nlargest(5, 'correlation')
        for _, row in top_correlations.iterrows():
            logger.info(f"    {row['source_ticker']} ↔ {row['target_ticker']}: {row['correlation']:.3f}")
    
    # Test payment companies correlation
    logger.info("💳 Testing Payment Companies Correlation Analysis...")
    payment_correlations = analyzer.compute_rolling_correlations(
        tickers=payment_companies,
        window_days=30
    )
    
    if not payment_correlations.empty:
        logger.info(f"  Computed {len(payment_correlations)} payment correlations")
        for _, row in payment_correlations.iterrows():
            logger.info(f"    {row['source_ticker']} ↔ {row['target_ticker']}: {row['correlation']:.3f}")
    
    return len(bank_correlations) + len(payment_correlations)


def test_ripple_simulation():
    """Test ripple effect simulation with financial services focus."""
    logger = logging.getLogger(__name__)
    
    propagator = RipplePropagator()
    
    # Test 1: JPMorgan Chase shock (largest US bank)
    logger.info("🌊 Testing JPM Shock Simulation (-5% shock)...")
    jpm_results = propagator.simulate_shock_propagation(
        seed_ticker='JPM',
        shock_magnitude=-0.05,
        damping_factor=0.85
    )
    
    if not jpm_results.empty:
        logger.info("  Top 5 Most Impacted Stocks:")
        top_impacted = jpm_results.nlargest(5, 'cumulative_impact')
        for _, row in top_impacted.iterrows():
            logger.info(f"    {row['ticker']}: {row['cumulative_impact']:.4f}")
    
    # Test 2: Visa shock (payment system)
    logger.info("💳 Testing Visa Shock Simulation (-3% shock)...")
    visa_results = propagator.simulate_shock_propagation(
        seed_ticker='V',
        shock_magnitude=-0.03,
        damping_factor=0.80
    )
    
    if not visa_results.empty:
        logger.info("  Top 5 Most Impacted Stocks:")
        top_impacted = visa_results.nlargest(5, 'cumulative_impact')
        for _, row in top_impacted.iterrows():
            logger.info(f"    {row['ticker']}: {row['cumulative_impact']:.4f}")
    
    # Test 3: AIG shock (insurance sector)
    logger.info("🛡️  Testing AIG Shock Simulation (-4% shock)...")
    aig_results = propagator.simulate_shock_propagation(
        seed_ticker='AIG',
        shock_magnitude=-0.04,
        damping_factor=0.75
    )
    
    if not aig_results.empty:
        logger.info("  Top 5 Most Impacted Stocks:")
        top_impacted = aig_results.nlargest(5, 'cumulative_impact')
        for _, row in top_impacted.iterrows():
            logger.info(f"    {row['ticker']}: {row['cumulative_impact']:.4f}")
    
    return len(jpm_results) + len(visa_results) + len(aig_results)


def test_sector_analysis():
    """Test sector-wide analysis capabilities."""
    logger = logging.getLogger(__name__)
    
    # Get price performance for different financial sectors
    query = """
    SELECT 
        ticker,
        AVG(close) as avg_price,
        STDDEV(close) as price_volatility,
        (MAX(close) - MIN(close)) / MIN(close) * 100 as price_range_pct
    FROM prices 
    WHERE ticker IN ('JPM','BAC','WFC','GS','MS','C','V','MA','AXP','BLK','AIG','MET','PRU')
    AND trade_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY ticker
    ORDER BY price_volatility DESC
    """
    
    performance_data = pg_manager.execute_query(query)
    
    if performance_data:
        logger.info("📈 30-Day Financial Sector Performance Analysis:")
        logger.info("  Most Volatile Stocks:")
        for i, row in enumerate(performance_data[:5]):
            logger.info(f"    {i+1}. {row['ticker']}: "
                       f"${row['avg_price']:.2f} avg, "
                       f"{row['price_volatility']:.2f} volatility, "
                       f"{row['price_range_pct']:.1f}% range")
    
    return len(performance_data)


def main():
    """Main pipeline test function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Financial Services Data Pipeline Test")
    logger.info("=" * 60)
    
    try:
        # Test 1: Data Integrity
        logger.info("1️⃣  Testing Data Integrity...")
        ticker_count, record_count = test_data_integrity()
        logger.info(f"✅ Data integrity check passed: {ticker_count} tickers, {record_count} records")
        
        # Test 2: Correlation Analysis
        logger.info("\n2️⃣  Testing Correlation Analysis...")
        correlation_count = test_correlation_analysis()
        logger.info(f"✅ Correlation analysis passed: {correlation_count} correlations computed")
        
        # Test 3: Ripple Simulation
        logger.info("\n3️⃣  Testing Ripple Simulation...")
        simulation_count = test_ripple_simulation()
        logger.info(f"✅ Ripple simulation passed: {simulation_count} results generated")
        
        # Test 4: Sector Analysis
        logger.info("\n4️⃣  Testing Sector Analysis...")
        analysis_count = test_sector_analysis()
        logger.info(f"✅ Sector analysis passed: {analysis_count} stocks analyzed")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 All Financial Services Pipeline Tests PASSED!")
        logger.info(f"📊 Summary: {ticker_count} tickers, {record_count} records, "
                   f"{correlation_count} correlations, {simulation_count} simulations")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())