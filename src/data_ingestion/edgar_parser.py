"""EDGAR data parser for SEC filings."""

import requests
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd
from bs4 import BeautifulSoup
import time

from .base import DataIngestor, CompanyData
from config.settings import config

import logging
logger = logging.getLogger(__name__)


class EDGARParser(DataIngestor):
    """Parser for SEC EDGAR filings to extract ownership relationships."""
    
    def __init__(self):
        super().__init__("EDGAR")
        self.base_url = "https://www.sec.gov"
        self.headers = {
            'User-Agent': config.api.edgar_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        self.rate_limit = 0.1  # 10 requests per second limit
    
    def fetch_data(self, cik: Optional[str] = None, ticker: Optional[str] = None,
                   form_type: str = "13F-HR", limit: int = 10) -> pd.DataFrame:
        """Fetch EDGAR filings for a company."""
        if not cik and not ticker:
            raise ValueError("Either CIK or ticker must be provided")
        
        if ticker and not cik:
            cik = self._get_cik_from_ticker(ticker)
        
        # Clean CIK (pad to 10 digits)
        cik = str(cik).zfill(10)
        
        # Get company filings
        filings_url = f"{self.base_url}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': form_type,
            'dateb': '',
            'owner': 'exclude',
            'count': limit,
            'output': 'atom'
        }
        
        response = self._make_request(filings_url, params)
        return self._parse_filings_response(response.content)
    
    def transform_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform EDGAR filings into ownership relationships."""
        relationships = []
        
        for _, filing in raw_data.iterrows():
            try:
                # Parse the filing document
                holdings = self._parse_13f_holdings(filing['document_url'])
                for holding in holdings:
                    relationships.append({
                        'source_ticker': filing['ticker'],
                        'target_ticker': holding.get('ticker'),
                        'target_name': holding.get('name'),
                        'relation_type': 'OWNS',
                        'weight': holding.get('ownership_percent', 0.0),
                        'shares': holding.get('shares', 0),
                        'market_value': holding.get('market_value', 0.0),
                        'filing_date': filing['filing_date'],
                        'source': 'EDGAR_13F'
                    })
                time.sleep(self.rate_limit)  # Rate limiting
            except Exception as e:
                self.logger.warning(f"Failed to parse filing {filing['document_url']}: {e}")
                continue
        
        return pd.DataFrame(relationships)
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate ownership relationship data."""
        required_columns = ['source_ticker', 'target_ticker', 'relation_type', 'weight']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for valid weight values
        if data['weight'].isnull().any() or (data['weight'] < 0).any():
            return False
        
        return True
    
    def _get_cik_from_ticker(self, ticker: str) -> str:
        """Get CIK from ticker symbol."""
        # Use SEC company tickers JSON
        tickers_url = f"{self.base_url}/files/company_tickers.json"
        response = self._make_request(tickers_url)
        
        tickers_data = response.json()
        for company_info in tickers_data.values():
            if company_info['ticker'].upper() == ticker.upper():
                return str(company_info['cik_str'])
        
        raise ValueError(f"CIK not found for ticker: {ticker}")
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Make HTTP request with rate limiting and error handling."""
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            time.sleep(self.rate_limit)
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def _parse_filings_response(self, content: bytes) -> pd.DataFrame:
        """Parse ATOM feed response from EDGAR."""
        try:
            root = ET.fromstring(content)
            entries = []
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                filing = {
                    'title': entry.find('atom:title', ns).text if entry.find('atom:title', ns) is not None else '',
                    'filing_date': entry.find('atom:updated', ns).text if entry.find('atom:updated', ns) is not None else '',
                    'document_url': ''
                }
                
                # Extract document URL from links
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'text/html':
                        filing['document_url'] = self.base_url + link.get('href', '')
                        break
                
                entries.append(filing)
            
            return pd.DataFrame(entries)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse EDGAR response: {e}")
            return pd.DataFrame()
    
    def _parse_13f_holdings(self, document_url: str) -> List[Dict]:
        """Parse 13F-HR filing to extract holdings."""
        try:
            response = self._make_request(document_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            holdings = []
            
            # Look for holdings table - this is simplified
            # Real implementation would need to handle various 13F formats
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        try:
                            holding = {
                                'name': cells[0].get_text(strip=True),
                                'ticker': self._extract_ticker_from_text(cells[0].get_text()),
                                'shares': self._parse_number(cells[1].get_text()),
                                'market_value': self._parse_number(cells[2].get_text()),
                                'ownership_percent': self._parse_percentage(cells[3].get_text())
                            }
                            if holding['name']:  # Only add if we have a company name
                                holdings.append(holding)
                        except (ValueError, IndexError):
                            continue
            
            return holdings
        except Exception as e:
            self.logger.error(f"Failed to parse 13F holdings from {document_url}: {e}")
            return []
    
    def _extract_ticker_from_text(self, text: str) -> Optional[str]:
        """Extract ticker symbol from company name text."""
        # Simple regex to find ticker patterns (3-5 uppercase letters)
        match = re.search(r'\b([A-Z]{2,5})\b', text)
        return match.group(1) if match else None
    
    def _parse_number(self, text: str) -> float:
        """Parse number from text, handling commas and formatting."""
        cleaned = re.sub(r'[^0-9.-]', '', text)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    def _parse_percentage(self, text: str) -> float:
        """Parse percentage from text."""
        # Extract percentage value
        match = re.search(r'([\d.]+)%?', text)
        if match:
            return float(match.group(1))
        return 0.0
    
    def get_insider_ownership(self, ticker: str) -> pd.DataFrame:
        """Get insider ownership data from Form 4 filings."""
        try:
            cik = self._get_cik_from_ticker(ticker)
            # Fetch Form 4 filings (insider transactions)
            return self.fetch_data(cik=cik, form_type="4", limit=50)
        except Exception as e:
            self.logger.error(f"Failed to get insider ownership for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_institutional_ownership(self, ticker: str) -> pd.DataFrame:
        """Get institutional ownership from 13F filings."""
        try:
            cik = self._get_cik_from_ticker(ticker)
            return self.fetch_data(cik=cik, form_type="13F-HR", limit=20)
        except Exception as e:
            self.logger.error(f"Failed to get institutional ownership for {ticker}: {e}")
            return pd.DataFrame()