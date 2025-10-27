"""Main application entry point."""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import setup_logging
from src.visualization import RippleDashboard
from src.data_ingestion import MarketDataIngestor, EDGARParser, NewsLoader
from src.analytics import CorrelationAnalyzer, RipplePropagator, GraphBuilder
from src.database import pg_manager, neo4j_manager
from config.settings import config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Stock Dependency & Ripple Effect Analysis Platform")
    parser.add_argument("--mode", choices=["dashboard", "ingest", "analyze", "simulate"], 
                       default="dashboard", help="Application mode")
    parser.add_argument("--tickers", nargs="+", help="List of ticker symbols")
    parser.add_argument("--seed-ticker", help="Seed ticker for simulation")
    parser.add_argument("--shock", type=float, default=-0.05, help="Shock magnitude")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Stock Ripple Platform in {args.mode} mode")
    
    try:
        if args.mode == "dashboard":
            # Run dashboard
            dashboard = RippleDashboard()
            logger.info(f"Starting dashboard on port {args.port}")
            dashboard.run_server(debug=args.debug, port=args.port)
            
        elif args.mode == "ingest":
            # Run data ingestion
            if not args.tickers:
                logger.error("Tickers required for ingestion mode")
                return 1
            
            logger.info(f"Ingesting data for tickers: {args.tickers}")
            
            # Market data ingestion
            market_ingestor = MarketDataIngestor()
            price_data = market_ingestor.ingest(tickers=args.tickers)
            logger.info(f"Ingested {len(price_data)} price records")
            
            # Store in database (simplified - would need proper ETL)
            if not price_data.empty:
                try:
                    # Insert into PostgreSQL
                    pg_manager.insert_dataframe(price_data, 'prices')
                    logger.info("Price data stored in PostgreSQL")
                except Exception as e:
                    logger.warning(f"Could not store in PostgreSQL: {e}")
                    logger.info("Data ingestion completed but not stored (database not available)")
                    # Show sample of ingested data
                    print("\nSample of ingested data:")
                    print(price_data.head(10).to_string(index=False))
            
        elif args.mode == "analyze":
            # Run correlation analysis
            if not args.tickers:
                logger.error("Tickers required for analysis mode")
                return 1
            
            logger.info(f"Analyzing correlations for tickers: {args.tickers}")
            
            analyzer = CorrelationAnalyzer()
            correlations = analyzer.compute_static_correlations(args.tickers)
            
            if not correlations.empty:
                logger.info(f"Computed {len(correlations)} correlation relationships")
                # Store in graph database
                graph_builder = GraphBuilder()
                result = graph_builder.update_graph_from_correlations(correlations)
                logger.info(f"Created {result.get('relationships_created', 0)} graph relationships")
            
        elif args.mode == "simulate":
            # Run ripple simulation
            if not args.seed_ticker:
                logger.error("Seed ticker required for simulation mode")
                return 1
            
            logger.info(f"Running ripple simulation for {args.seed_ticker}")
            
            propagator = RipplePropagator()
            results = propagator.simulate_shock_propagation(
                seed_ticker=args.seed_ticker,
                shock_magnitude=args.shock,
                tickers=args.tickers
            )
            
            if not results.empty:
                logger.info(f"Simulation completed for {len(results)} stocks")
                # Print top impacted stocks
                top_impacted = results.nlargest(10, 'final_impact', keep='all')
                print("\nTop 10 Most Impacted Stocks:")
                print(top_impacted[['ticker', 'final_impact', 'cumulative_impact']].to_string(index=False))
            else:
                logger.warning("No simulation results generated")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())