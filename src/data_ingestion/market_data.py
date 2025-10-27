"""Market data ingestor for stock prices and market information."""

import yfinance as yf
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, date, timedelta
import pandas as pd
import time

from .base import DataIngestor, PriceData
from config.settings import config

import logging
logger = logging.getLogger(__name__)


class MarketDataIngestor(DataIngestor):
    """Ingestor for market data from various sources."""
    
    def __init__(self):
        super().__init__("MarketData")
        self.alpha_vantage_key = config.api.alpha_vantage_api_key
        self.rate_limit = 0.2  # 5 requests per second for Alpha Vantage
    
    def fetch_data(self, tickers: Union[str, List[str]], 
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   source: str = "yahoo") -> pd.DataFrame:
        """Fetch price data for given tickers."""
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Set default date range
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=config.app.price_data_lookback_days)
        
        all_data = []
        
        for ticker in tickers:
            try:
                if source.lower() == "yahoo":
                    data = self._fetch_yahoo_data(ticker, start_date, end_date)
                elif source.lower() == "alpha_vantage":
                    data = self._fetch_alpha_vantage_data(ticker)
                else:
                    raise ValueError(f"Unsupported data source: {source}")
                
                if not data.empty:
                    data['ticker'] = ticker
                    data['source'] = source
                    all_data.append(data)
                
                time.sleep(self.rate_limit)  # Rate limiting
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {ticker}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def transform_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw price data into standardized format."""
        if raw_data.empty:
            return raw_data
        
        # Ensure required columns exist
        required_columns = ['ticker', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        
        # Rename columns to match our schema if needed
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Date': 'trade_date'
        }
        
        transformed = raw_data.rename(columns=column_mapping)
        
        # Ensure trade_date is a date column
        if 'trade_date' not in transformed.columns and transformed.index.name == 'Date':
            transformed = transformed.reset_index()
            transformed = transformed.rename(columns={'Date': 'trade_date'})
        
        # Convert date column
        if 'trade_date' in transformed.columns:
            transformed['trade_date'] = pd.to_datetime(transformed['trade_date']).dt.date
        
        # Add adj_close if missing (use close as fallback)
        if 'adj_close' not in transformed.columns:
            transformed['adj_close'] = transformed['close']
        
        # Fill missing values with forward fill, then backward fill
        numeric_columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_columns:
            if col in transformed.columns:
                transformed[col] = pd.to_numeric(transformed[col], errors='coerce')
                transformed[col] = transformed[col].fillna(method='ffill').fillna(method='bfill')
        
        return transformed
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate price data."""
        if data.empty:
            return True  # Empty data is valid
        
        required_columns = ['ticker', 'trade_date', 'close']
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Expected: {required_columns}")
            return False
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in data.columns:
                if (data[col] < 0).any():
                    self.logger.error(f"Negative prices found in column: {col}")
                    return False
        
        # Check that high >= low
        if 'high' in data.columns and 'low' in data.columns:
            if (data['high'] < data['low']).any():
                self.logger.error("High prices less than low prices found")
                return False
        
        return True
    
    def _fetch_yahoo_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No data found for ticker: {ticker}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            return data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_alpha_vantage_data(self, ticker: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(data['Error Message'])
            
            if 'Time Series (Daily)' not in data:
                raise ValueError("No time series data in response")
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df_data = []
            
            for date_str, values in time_series.items():
                row = {
                    'trade_date': datetime.strptime(date_str, '%Y-%m-%d').date(),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'adj_close': float(values['5. adjusted close']),
                    'volume': int(values['6. volume'])
                }
                df_data.append(row)
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_company_info(self, ticker: str) -> Dict:
        """Get company information and metadata."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'country': info.get('country', ''),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            self.logger.error(f"Failed to get company info for {ticker}: {e}")
            return {'ticker': ticker}
    
    def get_multiple_company_info(self, tickers: List[str]) -> pd.DataFrame:
        """Get company information for multiple tickers."""
        companies = []
        
        for ticker in tickers:
            try:
                info = self.get_company_info(ticker)
                companies.append(info)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.logger.warning(f"Failed to get info for {ticker}: {e}")
                continue
        
        return pd.DataFrame(companies)
    
    def get_real_time_quote(self, ticker: str) -> Dict:
        """Get real-time quote for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d", interval="1m")
            
            if data.empty:
                return {}
            
            latest = data.iloc[-1]
            return {
                'ticker': ticker,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'timestamp': latest.name.to_pydatetime()
            }
        except Exception as e:
            self.logger.error(f"Failed to get real-time quote for {ticker}: {e}")
            return {}