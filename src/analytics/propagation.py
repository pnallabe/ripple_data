"""Ripple effect propagation simulation engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

from .graph_builder import GraphBuilder
from src.database import neo4j_manager, pg_manager
from config.settings import config

logger = logging.getLogger(__name__)


class RipplePropagator:
    """Engine for simulating ripple effects in stock dependency networks."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RipplePropagator")
        self.graph_builder = GraphBuilder()
        self.default_damping = config.analytics.default_damping_factor
        self.max_iterations = config.analytics.max_iterations
        self.convergence_threshold = config.analytics.convergence_threshold
    
    def simulate_shock_propagation(self, 
                                 seed_ticker: str,
                                 shock_magnitude: float,
                                 tickers: Optional[List[str]] = None,
                                 damping_factor: Optional[float] = None,
                                 max_iterations: Optional[int] = None,
                                 convergence_threshold: Optional[float] = None) -> pd.DataFrame:
        """Simulate shock propagation through the network."""
        try:
            # Use provided parameters or defaults
            alpha = damping_factor or self.default_damping
            max_iter = max_iterations or self.max_iterations
            tol = convergence_threshold or self.convergence_threshold
            
            # Build adjacency matrix
            adj_matrix, ticker_list = self.graph_builder.build_adjacency_matrix(tickers)
            
            if adj_matrix.size == 0:
                self.logger.warning("No adjacency matrix available for propagation")
                return pd.DataFrame()
            
            # Check if seed ticker is in the network
            if seed_ticker not in ticker_list:
                self.logger.error(f"Seed ticker {seed_ticker} not found in network")
                return pd.DataFrame()
            
            # Initialize shock vector
            n = len(ticker_list)
            seed_idx = ticker_list.index(seed_ticker)
            
            # Initial shock
            delta_0 = np.zeros(n)
            delta_0[seed_idx] = shock_magnitude
            
            # Propagation simulation
            results = self._matrix_propagation(adj_matrix, delta_0, alpha, max_iter, tol)
            
            # Create results DataFrame
            propagation_results = []
            for i, ticker in enumerate(ticker_list):
                propagation_results.append({
                    'ticker': ticker,
                    'initial_shock': delta_0[i],
                    'final_impact': results['final_impact'][i],
                    'cumulative_impact': results['cumulative_impact'][i],
                    'max_impact': results['max_impact'][i],
                    'iterations_to_peak': results['iterations_to_peak'][i],
                    'seed_ticker': seed_ticker,
                    'shock_magnitude': shock_magnitude,
                    'damping_factor': alpha,
                    'converged': results['converged'],
                    'total_iterations': results['iterations'],
                    'simulation_timestamp': datetime.now()
                })
            
            df_results = pd.DataFrame(propagation_results)
            
            # Sort by impact magnitude
            df_results = df_results.sort_values('final_impact', key=abs, ascending=False)
            
            return df_results
            
        except Exception as e:
            self.logger.error(f"Error in shock propagation simulation: {e}")
            return pd.DataFrame()
    
    def simulate_multiple_shocks(self, 
                               shock_scenarios: List[Dict],
                               tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Simulate multiple shock scenarios."""
        try:
            all_results = []
            
            for i, scenario in enumerate(shock_scenarios):
                scenario_id = f"scenario_{i+1}"
                seed_ticker = scenario.get('seed_ticker')
                shock_magnitude = scenario.get('shock_magnitude', -0.05)
                damping_factor = scenario.get('damping_factor')
                
                if not seed_ticker:
                    self.logger.warning(f"No seed ticker specified for {scenario_id}")
                    continue
                
                results = self.simulate_shock_propagation(
                    seed_ticker=seed_ticker,
                    shock_magnitude=shock_magnitude,
                    tickers=tickers,
                    damping_factor=damping_factor
                )
                
                if not results.empty:
                    results['scenario_id'] = scenario_id
                    results['scenario_name'] = scenario.get('name', scenario_id)
                    all_results.append(results)
            
            if all_results:
                return pd.concat(all_results, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in multiple shock simulation: {e}")
            return pd.DataFrame()
    
    def compute_systemic_risk_metrics(self, 
                                    tickers: Optional[List[str]] = None,
                                    shock_magnitude: float = -0.05) -> pd.DataFrame:
        """Compute systemic risk metrics for each stock."""
        try:
            if not tickers:
                # Get all tickers from the network
                query = "MATCH (c:Company) RETURN c.ticker AS ticker"
                ticker_results = neo4j_manager.execute_query(query)
                tickers = [r['ticker'] for r in ticker_results]
            
            risk_metrics = []
            
            # Simulate shock from each ticker
            for seed_ticker in tickers:
                try:
                    results = self.simulate_shock_propagation(
                        seed_ticker=seed_ticker,
                        shock_magnitude=shock_magnitude,
                        tickers=tickers
                    )
                    
                    if not results.empty:
                        # Calculate systemic impact metrics
                        total_impact = results['final_impact'].abs().sum()
                        affected_nodes = (results['final_impact'].abs() > 1e-6).sum()
                        max_secondary_impact = results[results['ticker'] != seed_ticker]['final_impact'].abs().max()
                        
                        risk_metrics.append({
                            'ticker': seed_ticker,
                            'systemic_impact_total': total_impact,
                            'affected_nodes_count': affected_nodes,
                            'max_secondary_impact': max_secondary_impact,
                            'systemic_risk_score': total_impact * affected_nodes / len(tickers),
                            'shock_magnitude': shock_magnitude
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed systemic risk calculation for {seed_ticker}: {e}")
                    continue
            
            df_risk = pd.DataFrame(risk_metrics)
            
            if not df_risk.empty:
                # Normalize risk scores
                df_risk['systemic_risk_score_normalized'] = (
                    df_risk['systemic_risk_score'] / df_risk['systemic_risk_score'].max()
                )
                
                # Rank by systemic risk
                df_risk = df_risk.sort_values('systemic_risk_score', ascending=False)
                df_risk['systemic_risk_rank'] = range(1, len(df_risk) + 1)
            
            return df_risk
            
        except Exception as e:
            self.logger.error(f"Error computing systemic risk metrics: {e}")
            return pd.DataFrame()
    
    def analyze_contagion_pathways(self, 
                                 seed_ticker: str,
                                 target_ticker: str,
                                 tickers: Optional[List[str]] = None) -> Dict:
        """Analyze contagion pathways between two stocks.""" 
        try:
            # Build NetworkX graph for path analysis
            G = self.graph_builder.build_networkx_graph(tickers)
            
            if seed_ticker not in G.nodes() or target_ticker not in G.nodes():
                return {'error': 'One or both tickers not found in network'}
            
            # Find shortest paths
            try:
                import networkx as nx
                shortest_paths = list(nx.all_shortest_paths(G, seed_ticker, target_ticker))
                path_lengths = [len(path) - 1 for path in shortest_paths]
                
                # Analyze path weights
                path_analyses = []
                for path in shortest_paths[:5]:  # Limit to top 5 paths
                    path_weight = 1.0
                    path_edges = []
                    
                    for i in range(len(path) - 1):
                        edge_data = G.get_edge_data(path[i], path[i+1])
                        weight = edge_data.get('weight', 0.0) if edge_data else 0.0
                        path_weight *= weight
                        
                        path_edges.append({
                            'from': path[i],
                            'to': path[i+1],
                            'weight': weight,
                            'relation_type': edge_data.get('relation_type', '') if edge_data else ''
                        })
                    
                    path_analyses.append({
                        'path': path,
                        'path_length': len(path) - 1,
                        'path_weight': path_weight,
                        'edges': path_edges
                    })
                
                return {
                    'seed_ticker': seed_ticker,
                    'target_ticker': target_ticker,
                    'shortest_path_length': min(path_lengths) if path_lengths else None,
                    'num_shortest_paths': len(shortest_paths),
                    'path_analyses': path_analyses,
                    'direct_connection': G.has_edge(seed_ticker, target_ticker),
                    'direct_weight': G.get_edge_data(seed_ticker, target_ticker, {}).get('weight', 0.0) if G.has_edge(seed_ticker, target_ticker) else 0.0
                }
                
            except ImportError:
                return {'error': 'NetworkX not available for path analysis'}
            except nx.NetworkXNoPath:
                return {
                    'seed_ticker': seed_ticker,
                    'target_ticker': target_ticker,
                    'no_path': True,
                    'direct_connection': G.has_edge(seed_ticker, target_ticker),
                    'direct_weight': G.get_edge_data(seed_ticker, target_ticker, {}).get('weight', 0.0) if G.has_edge(seed_ticker, target_ticker) else 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing contagion pathways: {e}")
            return {'error': str(e)}
    
    def _matrix_propagation(self, 
                          adj_matrix: np.ndarray, 
                          initial_shock: np.ndarray,
                          damping_factor: float,
                          max_iterations: int,
                          convergence_threshold: float) -> Dict:
        """Perform matrix-based propagation simulation."""
        try:
            n = adj_matrix.shape[0]
            
            # Initialize tracking arrays
            delta_t = initial_shock.copy()
            cumulative_impact = initial_shock.copy()
            max_impact = np.abs(initial_shock.copy())
            iterations_to_peak = np.zeros(n, dtype=int)
            
            # Track convergence
            converged = False
            iteration_impacts = [delta_t.copy()]
            
            for t in range(max_iterations):
                # Propagation step: ΔP_{t+1} = α * A * ΔP_t
                delta_next = damping_factor * adj_matrix.dot(delta_t)
                
                # Update cumulative impact
                cumulative_impact += delta_next
                
                # Update max impact and iteration to peak
                abs_delta_next = np.abs(delta_next)
                peak_mask = abs_delta_next > max_impact
                max_impact[peak_mask] = abs_delta_next[peak_mask]
                iterations_to_peak[peak_mask] = t + 1
                
                # Check convergence
                delta_diff = np.linalg.norm(delta_next - delta_t)
                if delta_diff < convergence_threshold:
                    converged = True
                    break
                
                delta_t = delta_next
                iteration_impacts.append(delta_t.copy())
            
            return {
                'final_impact': delta_t,
                'cumulative_impact': cumulative_impact,
                'max_impact': max_impact,
                'iterations_to_peak': iterations_to_peak,
                'converged': converged,
                'iterations': len(iteration_impacts) - 1,
                'iteration_history': iteration_impacts
            }
            
        except Exception as e:
            self.logger.error(f"Error in matrix propagation: {e}")
            return {
                'final_impact': np.zeros_like(initial_shock),
                'cumulative_impact': np.zeros_like(initial_shock),
                'max_impact': np.zeros_like(initial_shock),
                'iterations_to_peak': np.zeros(len(initial_shock), dtype=int),
                'converged': False,
                'iterations': 0,
                'iteration_history': []
            }
    
    def compute_impact_attribution(self, 
                                 propagation_results: pd.DataFrame,
                                 top_k: int = 10) -> pd.DataFrame:
        """Compute impact attribution analysis."""
        try:
            if propagation_results.empty:
                return pd.DataFrame()
            
            # Get top impacted stocks
            top_impacted = propagation_results.nlargest(top_k, 'final_impact', keep='all')
            
            attribution_data = []
            
            for _, stock in top_impacted.iterrows():
                # Get market cap for impact scaling
                market_cap_query = """
                SELECT market_cap FROM companies WHERE ticker = %s
                """
                market_cap_result = pg_manager.execute_query(market_cap_query, (stock['ticker'],))
                market_cap = market_cap_result[0]['market_cap'] if market_cap_result else None
                
                # Calculate dollar impact if market cap available
                dollar_impact = None
                if market_cap and stock['final_impact']:
                    dollar_impact = market_cap * stock['final_impact']
                
                attribution_data.append({
                    'ticker': stock['ticker'],
                    'final_impact_pct': stock['final_impact'] * 100,
                    'market_cap': market_cap,
                    'dollar_impact': dollar_impact,
                    'cumulative_impact_pct': stock['cumulative_impact'] * 100,
                    'max_impact_pct': stock['max_impact'] * 100,
                    'iterations_to_peak': stock['iterations_to_peak'],
                    'impact_rank': len(attribution_data) + 1
                })
            
            df_attribution = pd.DataFrame(attribution_data)
            
            # Calculate total portfolio impact if dollar impacts available
            if df_attribution['dollar_impact'].notna().any():
                total_dollar_impact = df_attribution['dollar_impact'].sum()
                df_attribution['impact_contribution_pct'] = (
                    df_attribution['dollar_impact'] / total_dollar_impact * 100
                )
            
            return df_attribution
            
        except Exception as e:
            self.logger.error(f"Error computing impact attribution: {e}")
            return pd.DataFrame()