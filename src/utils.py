"""Utility functions for the Stock Ripple Platform."""

import logging
import logging.config
import os
import sys
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, date
import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import config


def setup_logging(log_level: str = None, log_file: str = None) -> None:
    """Setup logging configuration."""
    log_level = log_level or config.app.log_level
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file
    if not log_file:
        log_file = log_dir / f"ripple_platform_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'simple',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filename': str(log_file),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            'src': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'config': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'file']
        }
    }
    
    logging.config.dictConfig(logging_config)


def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format."""
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation: 1-5 uppercase letters
    ticker = ticker.strip().upper()
    return ticker.isalpha() and 1 <= len(ticker) <= 5


def validate_tickers(tickers: List[str]) -> List[str]:
    """Validate and clean list of ticker symbols."""
    valid_tickers = []
    
    for ticker in tickers:
        if validate_ticker(ticker):
            valid_tickers.append(ticker.strip().upper())
    
    return list(set(valid_tickers))  # Remove duplicates


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount."""
    if pd.isna(amount) or amount is None:
        return "N/A"
    
    if amount >= 1e12:
        return f"${amount/1e12:.2f}T"
    elif amount >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.2f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.2f}K"
    else:
        return f"${amount:.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage value."""
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value * 100:.{decimal_places}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that handles zero division."""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def normalize_weights(weights: List[float], method: str = 'sum') -> List[float]:
    """Normalize weights using different methods."""
    weights_array = np.array(weights)
    
    if method == 'sum':
        total = np.sum(np.abs(weights_array))
        return (weights_array / total if total != 0 else weights_array).tolist()
    
    elif method == 'max':
        max_val = np.max(np.abs(weights_array))
        return (weights_array / max_val if max_val != 0 else weights_array).tolist()
    
    elif method == 'l2':
        l2_norm = np.linalg.norm(weights_array)
        return (weights_array / l2_norm if l2_norm != 0 else weights_array).tolist()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def clean_dataframe(df: pd.DataFrame, 
                   drop_duplicates: bool = True,
                   drop_na_columns: List[str] = None,
                   fill_na_values: Dict[str, Any] = None) -> pd.DataFrame:
    """Clean pandas DataFrame with common operations."""
    df_clean = df.copy()
    
    # Drop duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Drop rows with NA in specified columns
    if drop_na_columns:
        df_clean = df_clean.dropna(subset=drop_na_columns)
    
    # Fill NA values
    if fill_na_values:
        df_clean = df_clean.fillna(fill_na_values)
    
    return df_clean


def serialize_datetime(obj: Any) -> Any:
    """JSON serializer for datetime objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results_to_json(results: Dict[str, Any], filepath: str) -> None:
    """Save results dictionary to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=serialize_datetime)
    except Exception as e:
        logging.error(f"Error saving results to {filepath}: {e}")


def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """Load results dictionary from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading results from {filepath}: {e}")
        return {}


def create_directory_structure(base_path: str) -> None:
    """Create the standard directory structure for the platform."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "logs",
        "reports",
        "exports",
        "cache"
    ]
    
    base_path = Path(base_path)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)


def get_business_days_between(start_date: date, end_date: date) -> int:
    """Get number of business days between two dates."""
    return len(pd.bdate_range(start_date, end_date))


def get_trading_calendar(start_date: date, 
                        end_date: date, 
                        holidays: List[date] = None) -> List[date]:
    """Get trading calendar (business days excluding holidays)."""
    business_days = pd.bdate_range(start_date, end_date)
    
    if holidays:
        # Remove holidays
        trading_days = [day.date() for day in business_days if day.date() not in holidays]
    else:
        trading_days = [day.date() for day in business_days]
    
    return trading_days


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """Calculate returns from price series."""
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return calculation method: {method}")


def calculate_volatility(returns: pd.Series, 
                        window: int = None, 
                        annualize: bool = True) -> Union[float, pd.Series]:
    """Calculate volatility from returns series."""
    if window:
        # Rolling volatility
        vol = returns.rolling(window=window).std()
    else:
        # Single volatility value
        vol = returns.std()
    
    # Annualize (assuming daily returns)
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.0,
                          annualize: bool = True) -> float:
    """Calculate Sharpe ratio from returns series."""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    
    mean_excess_return = excess_returns.mean()
    volatility = excess_returns.std()
    
    if volatility == 0:
        return 0.0
    
    sharpe = mean_excess_return / volatility
    
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    
    return sharpe


def calculate_max_drawdown(price_series: pd.Series) -> Dict[str, Any]:
    """Calculate maximum drawdown from price series."""
    # Calculate cumulative returns
    cumulative = (1 + price_series.pct_change()).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    
    # Find recovery date
    recovery_date = None
    if max_drawdown_date is not None:
        post_drawdown = drawdown[max_drawdown_date:]
        recovery_points = post_drawdown[post_drawdown >= 0]
        if not recovery_points.empty:
            recovery_date = recovery_points.index[0]
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_date': max_drawdown_date,
        'recovery_date': recovery_date,
        'drawdown_duration': (recovery_date - max_drawdown_date).days if recovery_date else None
    }


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        self.logger.info(f"Completed {self.operation_name} in {duration.total_seconds():.2f} seconds")


class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> List[str]:
        """Validate price data DataFrame and return list of issues."""
        issues = []
        
        required_columns = ['ticker', 'trade_date', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for high > low consistency
        if 'high' in df.columns and 'low' in df.columns:
            if (df['high'] < df['low']).any():
                issues.append("High prices less than low prices found")
        
        # Check for reasonable volume values
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                issues.append("Negative volume values found")
        
        # Check for valid dates
        if 'trade_date' in df.columns:
            try:
                pd.to_datetime(df['trade_date'])
            except:
                issues.append("Invalid date format in trade_date column")
        
        return issues
    
    @staticmethod
    def validate_correlation_data(df: pd.DataFrame) -> List[str]:
        """Validate correlation data DataFrame."""
        issues = []
        
        required_columns = ['source_ticker', 'target_ticker', 'correlation']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Check correlation bounds
        if 'correlation' in df.columns:
            if (df['correlation'] < -1).any() or (df['correlation'] > 1).any():
                issues.append("Correlation values outside [-1, 1] range")
        
        return issues


# Global performance timer instance
def timed_operation(operation_name: str):
    """Decorator for timing operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator