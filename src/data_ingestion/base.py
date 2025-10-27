"""Base classes for data ingestion."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, date
import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def transform_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into standardized format."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the transformed data."""
        pass
    
    def ingest(self, **kwargs) -> pd.DataFrame:
        """Main ingestion pipeline."""
        try:
            self.logger.info(f"Starting data ingestion for {self.name}")
            
            # Fetch raw data
            raw_data = self.fetch_data(**kwargs)
            self.logger.info(f"Fetched {len(raw_data)} records from {self.name}")
            
            # Transform data
            transformed_data = self.transform_data(raw_data)
            self.logger.info(f"Transformed data: {len(transformed_data)} records")
            
            # Validate data
            if not self.validate_data(transformed_data):
                raise ValueError(f"Data validation failed for {self.name}")
            
            self.logger.info(f"Successfully ingested data from {self.name}")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error ingesting data from {self.name}: {e}")
            raise


class CompanyData:
    """Standard company data structure."""
    
    def __init__(self, ticker: str, name: str, cik: Optional[str] = None,
                 sector: Optional[str] = None, industry: Optional[str] = None,
                 market_cap: Optional[float] = None, **kwargs):
        self.ticker = ticker
        self.name = name
        self.cik = cik
        self.sector = sector
        self.industry = industry
        self.market_cap = market_cap
        self.metadata = kwargs


class PriceData:
    """Standard price data structure."""
    
    def __init__(self, ticker: str, trade_date: date, open_price: float,
                 high: float, low: float, close: float, adj_close: float,
                 volume: int, source: str):
        self.ticker = ticker
        self.trade_date = trade_date
        self.open_price = open_price
        self.high = high
        self.low = low
        self.close = close
        self.adj_close = adj_close
        self.volume = volume
        self.source = source


class EventData:
    """Standard event data structure."""
    
    def __init__(self, ticker: str, event_type: str, event_date: datetime,
                 headline: str, body: Optional[str] = None, 
                 magnitude: Optional[float] = None, source: str = None):
        self.ticker = ticker
        self.event_type = event_type
        self.event_date = event_date
        self.headline = headline
        self.body = body
        self.magnitude = magnitude
        self.source = source