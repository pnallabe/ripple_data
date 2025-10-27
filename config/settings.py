"""Configuration management for the Stock Ripple Platform."""

from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
import os
from pathlib import Path


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    # PostgreSQL Settings
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="ripple_db", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    
    # Neo4j Settings
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    
    # Redis Settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    class Config:
        env_file = ".env"


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    # Market Data APIs
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    yahoo_finance_enabled: bool = Field(default=True, env="YAHOO_FINANCE_ENABLED")
    
    # News APIs
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    
    # SEC EDGAR API
    edgar_user_agent: str = Field(default="ripple-platform contact@example.com", env="EDGAR_USER_AGENT")
    
    class Config:
        env_file = ".env"


class AnalyticsConfig(BaseSettings):
    """Analytics configuration settings."""
    
    # Correlation settings
    correlation_windows: list = Field(default=[30, 90, 180], env="CORRELATION_WINDOWS")
    min_periods: int = Field(default=20, env="MIN_PERIODS")
    
    # Ripple propagation settings
    default_damping_factor: float = Field(default=0.85, env="DEFAULT_DAMPING_FACTOR")
    max_iterations: int = Field(default=100, env="MAX_ITERATIONS")
    convergence_threshold: float = Field(default=1e-6, env="CONVERGENCE_THRESHOLD")
    
    # ML Model settings
    var_model_lags: int = Field(default=5, env="VAR_MODEL_LAGS")
    gnn_hidden_dim: int = Field(default=64, env="GNN_HIDDEN_DIM")
    gnn_num_layers: int = Field(default=3, env="GNN_NUM_LAYERS")
    
    class Config:
        env_file = ".env"


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # App settings
    app_name: str = Field(default="Stock Ripple Platform", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Data refresh settings
    data_refresh_interval_hours: int = Field(default=24, env="DATA_REFRESH_INTERVAL_HOURS")
    price_data_lookback_days: int = Field(default=365, env="PRICE_DATA_LOOKBACK_DAYS")
    
    # Processing settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    
    class Config:
        env_file = ".env"


class Config:
    """Main configuration class combining all settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path:
            os.environ["ENV_FILE"] = str(config_path)
        
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.analytics = AnalyticsConfig()
        self.app = AppConfig()
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return (
            f"postgresql://{self.database.postgres_user}:"
            f"{self.database.postgres_password}@"
            f"{self.database.postgres_host}:{self.database.postgres_port}/"
            f"{self.database.postgres_db}"
        )
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        auth = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return f"redis://{auth}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"


# Global config instance
config = Config()