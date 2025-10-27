"""News and sentiment data loader."""

import requests
import feedparser
from typing import Dict, List, Optional, Union
from datetime import datetime, date, timedelta
import pandas as pd
import re
import time

from .base import DataIngestor, EventData
from config.settings import config

import logging
logger = logging.getLogger(__name__)


class NewsLoader(DataIngestor):
    """Loader for news and sentiment data."""
    
    def __init__(self):
        super().__init__("NewsLoader")
        self.news_api_key = config.api.news_api_key
        self.rate_limit = 0.5  # 2 requests per second
    
    def fetch_data(self, tickers: Union[str, List[str]], 
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   source: str = "newsapi") -> pd.DataFrame:
        """Fetch news data for given tickers."""
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Set default date range (last 30 days)
        if not end_date:
            end_date = date.today()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        all_articles = []
        
        for ticker in tickers:
            try:
                if source.lower() == "newsapi":
                    articles = self._fetch_newsapi_data(ticker, start_date, end_date)
                elif source.lower() == "rss":
                    articles = self._fetch_rss_data(ticker)
                else:
                    raise ValueError(f"Unsupported news source: {source}")
                
                for article in articles:
                    article['ticker'] = ticker
                    article['source'] = source
                    all_articles.append(article)
                
                time.sleep(self.rate_limit)  # Rate limiting
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch news for {ticker}: {e}")
                continue
        
        return pd.DataFrame(all_articles)
    
    def transform_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw news data into standardized format."""
        if raw_data.empty:
            return raw_data
        
        transformed = raw_data.copy()
        
        # Standardize date format
        if 'publishedAt' in transformed.columns:
            transformed['event_date'] = pd.to_datetime(transformed['publishedAt'])
        elif 'published' in transformed.columns:
            transformed['event_date'] = pd.to_datetime(transformed['published'])
        
        # Standardize content fields
        if 'title' in transformed.columns:
            transformed['headline'] = transformed['title']
        
        if 'description' in transformed.columns:
            transformed['body'] = transformed['description']
        elif 'summary' in transformed.columns:
            transformed['body'] = transformed['summary']
        
        # Add event type
        transformed['event_type'] = 'NEWS'
        
        # Calculate sentiment magnitude (placeholder - would use actual sentiment analysis)
        transformed['magnitude'] = transformed.apply(self._calculate_sentiment_score, axis=1)
        
        # Clean up text fields
        if 'headline' in transformed.columns:
            transformed['headline'] = transformed['headline'].fillna('').astype(str)
        if 'body' in transformed.columns:
            transformed['body'] = transformed['body'].fillna('').astype(str)
        
        return transformed
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate news data."""
        if data.empty:
            return True
        
        required_columns = ['ticker', 'headline', 'event_date']
        if not all(col in data.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Expected: {required_columns}")
            return False
        
        # Check for valid dates
        if 'event_date' in data.columns:
            try:
                pd.to_datetime(data['event_date'])
            except Exception:
                self.logger.error("Invalid date format in event_date column")
                return False
        
        return True
    
    def _fetch_newsapi_data(self, ticker: str, start_date: date, end_date: date) -> List[Dict]:
        """Fetch news from NewsAPI."""
        if not self.news_api_key:
            raise ValueError("NewsAPI key not configured")
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"{ticker}" OR "{self._get_company_name(ticker)}"',
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': self.news_api_key,
            'pageSize': 100
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            
            return data.get('articles', [])
            
        except Exception as e:
            self.logger.error(f"NewsAPI error for {ticker}: {e}")
            return []
    
    def _fetch_rss_data(self, ticker: str) -> List[Dict]:
        """Fetch news from RSS feeds."""
        # Financial news RSS feeds
        rss_feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=8-K&dateb=&owner=exclude&count=10&output=atom"
        ]
        
        articles = []
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Check if article is relevant to the ticker
                    title = entry.get('title', '').upper()
                    summary = entry.get('summary', '').upper()
                    
                    if ticker.upper() in title or ticker.upper() in summary:
                        article = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source_name': feed.feed.get('title', 'RSS')
                        }
                        articles.append(article)
                
                time.sleep(0.1)  # Brief pause between feeds
                
            except Exception as e:
                self.logger.warning(f"Failed to parse RSS feed {feed_url}: {e}")
                continue
        
        return articles
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for ticker (simplified lookup)."""
        # This would ideally use a company lookup service or database
        # For now, just return the ticker
        return ticker
    
    def _calculate_sentiment_score(self, row: pd.Series) -> float:
        """Calculate sentiment score for news article (placeholder)."""
        # This is a very basic sentiment analysis
        # In production, you'd use a proper sentiment analysis library like VADER, TextBlob, or a transformer model
        
        text = f"{row.get('headline', '')} {row.get('body', '')}".lower()
        
        # Simple keyword-based sentiment
        positive_words = ['gain', 'profit', 'growth', 'increase', 'rise', 'boost', 'strong', 'positive', 'beat', 'exceed']
        negative_words = ['loss', 'decline', 'fall', 'drop', 'weak', 'negative', 'miss', 'concern', 'risk', 'warning']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Simple scoring: positive score - negative score, normalized
        score = (positive_count - negative_count) / max(len(text.split()), 1)
        
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, score))
    
    def get_earnings_events(self, ticker: str) -> pd.DataFrame:
        """Get earnings announcement events."""
        try:
            # This would typically use an earnings calendar API
            # For now, return empty DataFrame as placeholder
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to get earnings events for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_sec_filings_events(self, ticker: str) -> pd.DataFrame:
        """Get SEC filing events (8-K, 10-K, 10-Q)."""
        try:
            # Use RSS feed from SEC for recent filings
            # This is a simplified implementation
            feed_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=exclude&count=20&output=atom"
            
            feed = feedparser.parse(feed_url)
            filings = []
            
            for entry in feed.entries:
                filing = {
                    'ticker': ticker,
                    'event_type': 'SEC_FILING',
                    'headline': entry.get('title', ''),
                    'body': entry.get('summary', ''),
                    'event_date': pd.to_datetime(entry.get('published', '')),
                    'source': 'SEC_EDGAR',
                    'magnitude': 0.0  # Neutral for filings
                }
                filings.append(filing)
            
            return pd.DataFrame(filings)
            
        except Exception as e:
            self.logger.error(f"Failed to get SEC filings for {ticker}: {e}")
            return pd.DataFrame()


class SentimentAnalyzer:
    """Sentiment analysis utility."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SentimentAnalyzer")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        # Placeholder for advanced sentiment analysis
        # In production, you'd use libraries like:
        # - VADER (vaderSentiment)
        # - TextBlob
        # - Transformers (BERT, RoBERTa)
        # - FinBERT (finance-specific BERT)
        
        return {
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze sentiment for a DataFrame of text."""
        if text_column not in df.columns:
            return df
        
        sentiments = []
        for text in df[text_column]:
            sentiment = self.analyze_text(str(text))
            sentiments.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiments)
        return pd.concat([df, sentiment_df], axis=1)