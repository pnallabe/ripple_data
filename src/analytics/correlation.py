"""Correlation analysis for stock dependencies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler

from src.database import pg_manager
from config.settings import config

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzer for computing various types of correlations between stocks."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CorrelationAnalyzer")
        self.min_periods = config.analytics.min_periods
        self.correlation_windows = config.analytics.correlation_windows
    
    def compute_rolling_correlations(self, 
                                   tickers: List[str], 
                                   window_days: int = 90,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Compute rolling correlations between stocks."""
        try:
            # Get price data
            price_data = self._get_price_data(tickers, start_date, end_date)
            
            if price_data.empty:
                self.logger.warning("No price data found for correlation analysis")
                return pd.DataFrame()
            
            # Pivot data to have tickers as columns
            price_pivot = price_data.pivot_table(
                index='trade_date', 
                columns='ticker', 
                values='adj_close'
            )
            
            # Calculate returns
            returns = price_pivot.pct_change().dropna()
            
            # Compute rolling correlations
            correlations = []
            
            for i, ticker1 in enumerate(tickers):
                for j, ticker2 in enumerate(tickers):
                    if i >= j:  # Only compute upper triangle
                        continue
                    
                    if ticker1 not in returns.columns or ticker2 not in returns.columns:
                        continue
                    
                    # Rolling correlation
                    rolling_corr = returns[ticker1].rolling(window=window_days).corr(returns[ticker2])
                    
                    for date, corr_value in rolling_corr.dropna().items():
                        correlations.append({
                            'source_ticker': ticker1,
                            'target_ticker': ticker2,
                            'correlation': corr_value,
                            'window_days': window_days,
                            'date': date,
                            'relation_type': 'CORRELATED_WITH'
                        })
            
            return pd.DataFrame(correlations)
            
        except Exception as e:
            self.logger.error(f"Error computing rolling correlations: {e}")
            return pd.DataFrame()
    
    def compute_static_correlations(self, 
                                  tickers: List[str],
                                  lookback_days: int = 252,
                                  correlation_type: str = 'pearson') -> pd.DataFrame:
        """Compute static correlation matrix for given period."""
        try:
            # Get price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            price_data = self._get_price_data(tickers, start_date, end_date)
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Pivot and calculate returns
            price_pivot = price_data.pivot_table(
                index='trade_date',
                columns='ticker', 
                values='adj_close'
            )
            
            returns = price_pivot.pct_change().dropna()
            
            # Compute correlation matrix
            if correlation_type == 'spearman':
                corr_matrix = returns.corr(method='spearman')
            else:
                corr_matrix = returns.corr(method='pearson')
            
            # Convert to edge list format
            correlations = []
            for i, ticker1 in enumerate(corr_matrix.index):
                for j, ticker2 in enumerate(corr_matrix.columns):
                    if i >= j:  # Skip diagonal and lower triangle
                        continue
                    
                    correlations.append({
                        'source_ticker': ticker1,
                        'target_ticker': ticker2,
                        'correlation': corr_matrix.loc[ticker1, ticker2],
                        'correlation_type': correlation_type,
                        'lookback_days': lookback_days,
                        'date': end_date.date(),
                        'relation_type': 'CORRELATED_WITH'
                    })
            
            return pd.DataFrame(correlations)
            
        except Exception as e:
            self.logger.error(f"Error computing static correlations: {e}")
            return pd.DataFrame()
    
    def compute_granger_causality(self, 
                                ticker1: str, 
                                ticker2: str,
                                max_lags: int = 5,
                                lookback_days: int = 252) -> Dict:
        """Compute Granger causality between two stocks."""
        try:
            # Get price data for both stocks
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            price_data = self._get_price_data([ticker1, ticker2], start_date, end_date)
            
            if price_data.empty:
                return {}
            
            # Pivot and calculate returns
            price_pivot = price_data.pivot_table(
                index='trade_date',
                columns='ticker',
                values='adj_close'
            )
            
            if ticker1 not in price_pivot.columns or ticker2 not in price_pivot.columns:
                return {}
            
            returns = price_pivot.pct_change().dropna()
            
            # Prepare data for Granger causality test
            data = returns[[ticker1, ticker2]].dropna()
            
            if len(data) < max_lags * 2:
                self.logger.warning(f"Insufficient data for Granger causality test: {len(data)} observations")
                return {}
            
            # Test if ticker1 Granger-causes ticker2
            try:
                gc_result_1to2 = grangercausalitytests(data[[ticker2, ticker1]], max_lags, verbose=False)
                gc_result_2to1 = grangercausalitytests(data[[ticker1, ticker2]], max_lags, verbose=False)
                
                # Extract p-values for different lags
                p_values_1to2 = [gc_result_1to2[lag][0]['ssr_ftest'][1] for lag in range(1, max_lags + 1)]
                p_values_2to1 = [gc_result_2to1[lag][0]['ssr_ftest'][1] for lag in range(1, max_lags + 1)]
                
                # Find optimal lag (minimum p-value)
                best_lag_1to2 = np.argmin(p_values_1to2) + 1
                best_lag_2to1 = np.argmin(p_values_2to1) + 1
                
                return {
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'gc_1to2_pvalue': min(p_values_1to2),
                    'gc_2to1_pvalue': min(p_values_2to1),
                    'best_lag_1to2': best_lag_1to2,
                    'best_lag_2to1': best_lag_2to1,
                    'gc_1to2_significant': min(p_values_1to2) < 0.05,
                    'gc_2to1_significant': min(p_values_2to1) < 0.05
                }
                
            except Exception as e:
                self.logger.warning(f"Granger causality test failed for {ticker1}-{ticker2}: {e}")
                return {}
            
        except Exception as e:
            self.logger.error(f"Error computing Granger causality: {e}")
            return {}
    
    def compute_partial_correlations(self, 
                                   tickers: List[str],
                                   control_variables: Optional[List[str]] = None,
                                   lookback_days: int = 252) -> pd.DataFrame:
        """Compute partial correlations controlling for other variables."""
        try:
            # Get price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            all_tickers = tickers + (control_variables or [])
            price_data = self._get_price_data(all_tickers, start_date, end_date)
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Pivot and calculate returns
            price_pivot = price_data.pivot_table(
                index='trade_date',
                columns='ticker',
                values='adj_close'
            )
            
            returns = price_pivot.pct_change().dropna()
            
            # Standardize returns
            scaler = StandardScaler()
            returns_scaled = pd.DataFrame(
                scaler.fit_transform(returns),
                index=returns.index,
                columns=returns.columns
            )
            
            # Compute partial correlation matrix
            partial_corrs = []
            
            for i, ticker1 in enumerate(tickers):
                for j, ticker2 in enumerate(tickers):
                    if i >= j or ticker1 not in returns_scaled.columns or ticker2 not in returns_scaled.columns:
                        continue
                    
                    # Compute partial correlation
                    partial_corr = self._compute_partial_correlation(
                        returns_scaled, ticker1, ticker2, control_variables
                    )
                    
                    partial_corrs.append({
                        'source_ticker': ticker1,
                        'target_ticker': ticker2,
                        'partial_correlation': partial_corr,
                        'control_variables': control_variables,
                        'lookback_days': lookback_days,
                        'date': end_date.date(),
                        'relation_type': 'PARTIAL_CORR'
                    })
            
            return pd.DataFrame(partial_corrs)
            
        except Exception as e:
            self.logger.error(f"Error computing partial correlations: {e}")
            return pd.DataFrame()
    
    def _get_price_data(self, 
                       tickers: List[str], 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get price data from database."""
        # Set default dates
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=365)
        
        # Query price data
        query = """
        SELECT ticker, trade_date, adj_close, volume
        FROM prices 
        WHERE ticker = ANY(%s)
        AND trade_date BETWEEN %s AND %s
        ORDER BY ticker, trade_date
        """
        
        try:
            return pg_manager.read_dataframe(
                query, 
                (tickers, start_date.date(), end_date.date())
            )
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def _compute_partial_correlation(self, 
                                   data: pd.DataFrame, 
                                   var1: str, 
                                   var2: str,
                                   control_vars: Optional[List[str]] = None) -> float:
        """Compute partial correlation between two variables."""
        if not control_vars or len(control_vars) == 0:
            # If no control variables, return regular correlation
            return data[var1].corr(data[var2])
        
        try:
            # Use matrix inversion method for partial correlation
            vars_of_interest = [var1, var2] + control_vars
            available_vars = [v for v in vars_of_interest if v in data.columns]
            
            if len(available_vars) < 2:
                return 0.0
            
            # Compute correlation matrix
            corr_matrix = data[available_vars].corr()
            
            # Compute partial correlation using precision matrix
            precision_matrix = np.linalg.pinv(corr_matrix.values)
            
            # Partial correlation coefficient
            idx1 = available_vars.index(var1)
            idx2 = available_vars.index(var2)
            
            partial_corr = -precision_matrix[idx1, idx2] / np.sqrt(
                precision_matrix[idx1, idx1] * precision_matrix[idx2, idx2]
            )
            
            return partial_corr if not np.isnan(partial_corr) else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error computing partial correlation: {e}")
            return 0.0
    
    def compute_sector_correlations(self, lookback_days: int = 252) -> pd.DataFrame:
        """Compute correlations grouped by sectors."""
        try:
            # Get company sector information
            sector_query = """
            SELECT DISTINCT ticker, sector 
            FROM companies 
            WHERE sector IS NOT NULL AND sector != ''
            """
            
            sector_data = pg_manager.read_dataframe(sector_query)
            
            if sector_data.empty:
                return pd.DataFrame()
            
            # Group tickers by sector
            sectors = {}
            for _, row in sector_data.iterrows():
                sector = row['sector']
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(row['ticker'])
            
            all_correlations = []
            
            # Compute correlations within and between sectors
            for sector1, tickers1 in sectors.items():
                for sector2, tickers2 in sectors.items():
                    if sector1 == sector2:
                        # Within-sector correlations
                        corrs = self.compute_static_correlations(tickers1, lookback_days)
                        corrs['sector_relationship'] = 'WITHIN_SECTOR'
                        corrs['source_sector'] = sector1
                        corrs['target_sector'] = sector2
                    else:
                        # Cross-sector correlations (sample to avoid too many combinations)
                        sample_tickers1 = tickers1[:5]  # Limit to top 5 per sector
                        sample_tickers2 = tickers2[:5]
                        
                        for ticker1 in sample_tickers1:
                            for ticker2 in sample_tickers2:
                                corr_result = self.compute_static_correlations([ticker1, ticker2], lookback_days)
                                if not corr_result.empty:
                                    corr_result['sector_relationship'] = 'CROSS_SECTOR'
                                    corr_result['source_sector'] = sector1
                                    corr_result['target_sector'] = sector2
                                    all_correlations.append(corr_result)
                    
                    if not corrs.empty:
                        all_correlations.append(corrs)
            
            if all_correlations:
                return pd.concat(all_correlations, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error computing sector correlations: {e}")
            return pd.DataFrame()
    
    def compute_correlation_networks(self, 
                                   tickers: List[str],
                                   threshold: float = 0.3,
                                   lookback_days: int = 252) -> pd.DataFrame:
        """Compute correlation network with threshold filtering."""
        try:
            correlations = self.compute_static_correlations(tickers, lookback_days)
            
            if correlations.empty:
                return correlations
            
            # Filter by threshold (absolute value)
            filtered_corrs = correlations[
                np.abs(correlations['correlation']) >= threshold
            ].copy()
            
            # Add network metrics
            filtered_corrs['weight'] = np.abs(filtered_corrs['correlation'])
            filtered_corrs['direction'] = np.where(
                filtered_corrs['correlation'] > 0, 'POSITIVE', 'NEGATIVE'
            )
            
            return filtered_corrs
            
        except Exception as e:
            self.logger.error(f"Error computing correlation networks: {e}")
            return pd.DataFrame()