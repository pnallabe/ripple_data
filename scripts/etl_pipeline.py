"""ETL script for data pipeline automation."""

import sys
from pathlib import Path
import logging
from datetime import datetime, date, timedelta
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src import setup_logging, timed_operation
from src.data_ingestion import MarketDataIngestor, EDGARParser, NewsLoader
from src.analytics import CorrelationAnalyzer, GraphBuilder
from src.database import pg_manager, neo4j_manager
from config.settings import config


@timed_operation("ETL Pipeline")
def run_etl_pipeline(tickers: List[str] = None, full_refresh: bool = False):
    """Run the complete ETL pipeline."""
    logger = logging.getLogger(__name__)
    
    # Get tickers to process
    if not tickers:
        # Get active tickers from database
        query = "SELECT DISTINCT ticker FROM companies WHERE ticker IS NOT NULL LIMIT 50"
        ticker_results = pg_manager.execute_query(query)
        tickers = [row['ticker'] for row in ticker_results]
    
    if not tickers:
        logger.error("No tickers found for processing")
        return
    
    logger.info(f"Processing {len(tickers)} tickers: {tickers}")
    
    try:
        # 1. Market Data Ingestion
        with timed_operation("Market Data Ingestion"):
            market_ingestor = MarketDataIngestor()
            
            # Get company info first
            company_data = market_ingestor.get_multiple_company_info(tickers)
            if not company_data.empty:
                # Upsert companies
                for _, company in company_data.iterrows():
                    pg_manager.execute_non_query(
                        "SELECT upsert_company(%s, %s)",
                        (company['ticker'], company.get('name', ''))
                    )
                logger.info(f"Updated {len(company_data)} company records")
            
            # Get price data
            price_data = market_ingestor.ingest(tickers=tickers)
            if not price_data.empty:
                # Insert price data (handle duplicates)
                price_data = price_data.drop_duplicates(subset=['ticker', 'trade_date'])
                
                # Use upsert logic for prices
                for _, row in price_data.iterrows():
                    pg_manager.execute_non_query("""
                        INSERT INTO prices (ticker, trade_date, open, high, low, close, adj_close, volume, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, trade_date) 
                        DO UPDATE SET 
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source
                    """, (
                        row['ticker'], row['trade_date'], row.get('open'), row.get('high'),
                        row.get('low'), row['close'], row['adj_close'], 
                        row.get('volume'), row.get('source', 'yahoo')
                    ))
                
                logger.info(f"Processed {len(price_data)} price records")
        
        # 2. News Data Ingestion (if news API configured)
        if config.api.news_api_key:
            with timed_operation("News Data Ingestion"):
                news_loader = NewsLoader()
                
                for ticker in tickers[:10]:  # Limit to avoid API limits
                    try:
                        news_data = news_loader.ingest(tickers=[ticker])
                        if not news_data.empty:
                            # Insert news events
                            for _, event in news_data.iterrows():
                                pg_manager.execute_non_query("""
                                    INSERT INTO events (ticker, event_type, event_date, headline, body, magnitude, source)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT DO NOTHING
                                """, (
                                    event['ticker'], event.get('event_type', 'NEWS'),
                                    event['event_date'], event.get('headline', ''),
                                    event.get('body', ''), event.get('magnitude', 0.0),
                                    event.get('source', 'news_api')
                                ))
                            
                            logger.info(f"Processed {len(news_data)} news events for {ticker}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to process news for {ticker}: {e}")
                        continue
        
        # 3. Correlation Analysis
        with timed_operation("Correlation Analysis"):
            analyzer = CorrelationAnalyzer()
            
            # Compute correlations for multiple windows
            for window in config.analytics.correlation_windows:
                correlations = analyzer.compute_rolling_correlations(
                    tickers=tickers,
                    window_days=window
                )
                
                if not correlations.empty:
                    # Update graph with correlations
                    graph_builder = GraphBuilder()
                    result = graph_builder.update_graph_from_correlations(correlations)
                    logger.info(f"Created {result.get('relationships_created', 0)} "
                              f"correlation relationships ({window}d window)")
        
        # 4. EDGAR Data Processing (if available)
        with timed_operation("EDGAR Data Processing"):
            edgar_parser = EDGARParser()
            
            for ticker in tickers[:5]:  # Limit EDGAR processing to avoid rate limits
                try:
                    ownership_data = edgar_parser.get_institutional_ownership(ticker)
                    if not ownership_data.empty:
                        # Process ownership relationships
                        graph_builder = GraphBuilder()
                        result = graph_builder.update_graph_from_ownership(ownership_data)
                        logger.info(f"Created {result.get('relationships_created', 0)} "
                                  f"ownership relationships for {ticker}")
                
                except Exception as e:
                    logger.warning(f"Failed to process EDGAR data for {ticker}: {e}")
                    continue
        
        # 5. Graph Statistics and Cleanup
        with timed_operation("Graph Analysis"):
            graph_builder = GraphBuilder()
            stats = graph_builder.get_graph_statistics(tickers)
            logger.info(f"Graph statistics: {stats}")
            
            # Compute centrality metrics
            centrality_metrics = graph_builder.compute_centrality_metrics(tickers)
            if not centrality_metrics.empty:
                logger.info(f"Computed centrality metrics for {len(centrality_metrics)} nodes")
        
        logger.info("ETL pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point for ETL script."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Sample tickers for testing
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        'XOM', 'CVX', 'COP', 'SLB', 'HAL'
    ]
    
    try:
        run_etl_pipeline(tickers=sample_tickers)
        logger.info("ETL script completed successfully")
    except Exception as e:
        logger.error(f"ETL script failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())