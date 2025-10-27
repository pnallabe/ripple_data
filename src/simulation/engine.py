"""Advanced simulation engine for ripple effect analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass, asdict
from enum import Enum

from src.analytics.propagation import RipplePropagator
from src.analytics.graph_builder import GraphBuilder
from src.database import neo4j_manager, pg_manager
from config.settings import config

logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """Types of simulation models."""
    MATRIX_PROPAGATION = "matrix_propagation"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    SCENARIO_ANALYSIS = "scenario_analysis"
    SYSTEMIC_RISK = "systemic_risk"


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    simulation_type: SimulationType
    seed_ticker: str
    shock_magnitude: float
    damping_factor: float = 0.85
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    include_tickers: Optional[List[str]] = None
    exclude_tickers: Optional[List[str]] = None
    min_correlation: float = 0.0
    enable_feedback_loops: bool = True
    monte_carlo_runs: int = 1000
    confidence_level: float = 0.95
    time_horizon_days: int = 30
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class SimulationResults:
    """Container for simulation results."""
    simulation_id: str
    config: SimulationConfig
    results_df: pd.DataFrame
    statistics: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    convergence_info: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class SimulationEngine:
    """Advanced simulation engine with multiple modeling approaches."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimulationEngine")
        self.propagator = RipplePropagator()
        self.graph_builder = GraphBuilder()
        self.active_simulations = {}
        self.simulation_history = []
        
    def run_simulation(self, config: SimulationConfig) -> SimulationResults:
        """Run simulation based on configuration."""
        start_time = time.time()
        simulation_id = self._generate_simulation_id()
        
        try:
            self.logger.info(f"Starting simulation {simulation_id}: {config.simulation_type.value}")
            
            # Store active simulation
            self.active_simulations[simulation_id] = {
                'config': config,
                'start_time': start_time,
                'status': 'running'
            }
            
            # Route to appropriate simulation method
            if config.simulation_type == SimulationType.MATRIX_PROPAGATION:
                results_df, stats, convergence = self._run_matrix_simulation(config)
            elif config.simulation_type == SimulationType.MONTE_CARLO:
                results_df, stats, convergence = self._run_monte_carlo_simulation(config)
            elif config.simulation_type == SimulationType.STRESS_TEST:
                results_df, stats, convergence = self._run_stress_test(config)
            elif config.simulation_type == SimulationType.SCENARIO_ANALYSIS:
                results_df, stats, convergence = self._run_scenario_analysis(config)
            elif config.simulation_type == SimulationType.SYSTEMIC_RISK:
                results_df, stats, convergence = self._run_systemic_risk_analysis(config)
            else:
                raise ValueError(f"Unsupported simulation type: {config.simulation_type}")
            
            execution_time = time.time() - start_time
            
            # Create results object
            results = SimulationResults(
                simulation_id=simulation_id,
                config=config,
                results_df=results_df,
                statistics=stats,
                execution_time=execution_time,
                timestamp=datetime.now(),
                convergence_info=convergence
            )
            
            # Update active simulations
            self.active_simulations[simulation_id]['status'] = 'completed'
            self.active_simulations[simulation_id]['results'] = results
            
            # Add to history
            self.simulation_history.append(results)
            
            self.logger.info(f"Simulation {simulation_id} completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Simulation {simulation_id} failed after {execution_time:.2f}s: {e}")
            
            # Update status
            if simulation_id in self.active_simulations:
                self.active_simulations[simulation_id]['status'] = 'failed'
                self.active_simulations[simulation_id]['error'] = str(e)
            
            raise
    
    def run_batch_simulations(self, configs: List[SimulationConfig], 
                            max_workers: int = 4) -> List[SimulationResults]:
        """Run multiple simulations in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulations
            future_to_config = {
                executor.submit(self.run_simulation, config): config 
                for config in configs
            }
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch simulation failed for {config.seed_ticker}: {e}")
        
        return results
    
    def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get status of a simulation."""
        if simulation_id not in self.active_simulations:
            return {'error': 'Simulation not found'}
        
        return self.active_simulations[simulation_id]
    
    def cancel_simulation(self, simulation_id: str) -> bool:
        """Cancel a running simulation."""
        if simulation_id in self.active_simulations:
            self.active_simulations[simulation_id]['status'] = 'cancelled'
            return True
        return False
    
    def get_simulation_history(self, limit: int = 50) -> List[SimulationResults]:
        """Get recent simulation history."""
        return self.simulation_history[-limit:]
    
    def _run_matrix_simulation(self, config: SimulationConfig) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run matrix-based propagation simulation."""
        try:
            # Get the network
            tickers = self._get_simulation_tickers(config)
            
            # Run propagation simulation
            results_df = self.propagator.simulate_shock_propagation(
                seed_ticker=config.seed_ticker,
                shock_magnitude=config.shock_magnitude,
                tickers=tickers,
                damping_factor=config.damping_factor,
                max_iterations=config.max_iterations,
                convergence_threshold=config.convergence_threshold
            )
            
            if results_df.empty:
                raise ValueError("No propagation results generated")
            
            # Calculate statistics
            stats = self._calculate_simulation_statistics(results_df, config)
            
            # Convergence info
            convergence = {
                'converged': results_df.iloc[0]['converged'] if 'converged' in results_df.columns else False,
                'iterations': results_df.iloc[0]['total_iterations'] if 'total_iterations' in results_df.columns else 0,
                'method': 'matrix_propagation'
            }
            
            return results_df, stats, convergence
            
        except Exception as e:
            self.logger.error(f"Matrix simulation error: {e}")
            raise
    
    def _run_monte_carlo_simulation(self, config: SimulationConfig) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run Monte Carlo simulation with uncertainty."""
        try:
            tickers = self._get_simulation_tickers(config)
            n_runs = config.monte_carlo_runs
            
            all_results = []
            
            for run in range(n_runs):
                # Add noise to shock magnitude and damping factor
                noisy_shock = np.random.normal(config.shock_magnitude, abs(config.shock_magnitude) * 0.1)
                noisy_damping = np.random.normal(config.damping_factor, 0.05)
                noisy_damping = np.clip(noisy_damping, 0.1, 0.99)
                
                # Run single simulation
                run_results = self.propagator.simulate_shock_propagation(
                    seed_ticker=config.seed_ticker,
                    shock_magnitude=noisy_shock,
                    tickers=tickers,
                    damping_factor=noisy_damping,
                    max_iterations=config.max_iterations
                )
                
                if not run_results.empty:
                    run_results['monte_carlo_run'] = run
                    all_results.append(run_results)
            
            if not all_results:
                raise ValueError("No Monte Carlo results generated")
            
            # Combine all runs
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Calculate statistics across runs
            summary_stats = []
            for ticker in combined_df['ticker'].unique():
                ticker_data = combined_df[combined_df['ticker'] == ticker]['final_impact']
                
                summary_stats.append({
                    'ticker': ticker,
                    'mean_impact': ticker_data.mean(),
                    'std_impact': ticker_data.std(),
                    'min_impact': ticker_data.min(),
                    'max_impact': ticker_data.max(),
                    'percentile_5': ticker_data.quantile(0.05),
                    'percentile_95': ticker_data.quantile(0.95),
                    'var_95': ticker_data.quantile(0.05),  # Value at Risk
                    'cvar_95': ticker_data[ticker_data <= ticker_data.quantile(0.05)].mean(),  # Conditional VaR
                    'final_impact': ticker_data.mean(),  # Use mean as primary result
                    'monte_carlo_runs': n_runs
                })
            
            results_df = pd.DataFrame(summary_stats)
            results_df = results_df.sort_values('mean_impact', key=abs, ascending=False)
            
            # Statistics
            stats = self._calculate_simulation_statistics(results_df, config)
            stats.update({
                'monte_carlo_runs': n_runs,
                'uncertainty_analysis': True,
                'confidence_level': config.confidence_level
            })
            
            convergence = {
                'method': 'monte_carlo',
                'runs_completed': len(all_results),
                'total_runs': n_runs
            }
            
            return results_df, stats, convergence
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation error: {e}")
            raise
    
    def _run_stress_test(self, config: SimulationConfig) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run stress test with multiple shock magnitudes."""
        try:
            tickers = self._get_simulation_tickers(config)
            
            # Define stress test scenarios
            stress_levels = [-0.01, -0.02, -0.05, -0.10, -0.15, -0.20, -0.30]
            
            all_results = []
            
            for stress_level in stress_levels:
                stress_results = self.propagator.simulate_shock_propagation(
                    seed_ticker=config.seed_ticker,
                    shock_magnitude=stress_level,
                    tickers=tickers,
                    damping_factor=config.damping_factor,
                    max_iterations=config.max_iterations
                )
                
                if not stress_results.empty:
                    stress_results['stress_level'] = stress_level
                    stress_results['stress_level_pct'] = stress_level * 100
                    all_results.append(stress_results)
            
            if not all_results:
                raise ValueError("No stress test results generated")
            
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Calculate stress sensitivity metrics
            sensitivity_stats = []
            for ticker in combined_df['ticker'].unique():
                ticker_data = combined_df[combined_df['ticker'] == ticker]
                
                # Linear regression to find sensitivity
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    ticker_data['stress_level'], ticker_data['final_impact']
                )
                
                sensitivity_stats.append({
                    'ticker': ticker,
                    'stress_sensitivity': slope,
                    'stress_r_squared': r_value ** 2,
                    'base_case_impact': ticker_data[ticker_data['stress_level'] == -0.05]['final_impact'].iloc[0] if not ticker_data[ticker_data['stress_level'] == -0.05].empty else 0,
                    'severe_stress_impact': ticker_data[ticker_data['stress_level'] == -0.20]['final_impact'].iloc[0] if not ticker_data[ticker_data['stress_level'] == -0.20].empty else 0,
                    'extreme_stress_impact': ticker_data[ticker_data['stress_level'] == -0.30]['final_impact'].iloc[0] if not ticker_data[ticker_data['stress_level'] == -0.30].empty else 0,
                    'final_impact': ticker_data[ticker_data['stress_level'] == config.shock_magnitude]['final_impact'].iloc[0] if not ticker_data[ticker_data['stress_level'] == config.shock_magnitude].empty else 0
                })
            
            results_df = pd.DataFrame(sensitivity_stats)
            results_df = results_df.sort_values('stress_sensitivity', key=abs, ascending=False)
            
            # Statistics
            stats = self._calculate_simulation_statistics(results_df, config)
            stats.update({
                'stress_test_levels': len(stress_levels),
                'sensitivity_analysis': True
            })
            
            convergence = {
                'method': 'stress_test',
                'scenarios_completed': len(all_results),
                'stress_levels': stress_levels
            }
            
            return results_df, stats, convergence
            
        except Exception as e:
            self.logger.error(f"Stress test error: {e}")
            raise
    
    def _run_scenario_analysis(self, config: SimulationConfig) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run scenario analysis with multiple seed tickers."""
        try:
            tickers = self._get_simulation_tickers(config)
            
            # Define scenarios - use top systemic risk stocks as seeds
            systemic_risk_df = self.propagator.compute_systemic_risk_metrics(tickers)
            
            if systemic_risk_df.empty:
                # Fallback to using major banks/financial institutions
                scenario_seeds = [config.seed_ticker]
                if config.include_tickers:
                    scenario_seeds.extend(config.include_tickers[:4])
            else:
                scenario_seeds = systemic_risk_df.head(5)['ticker'].tolist()
            
            all_results = []
            
            for i, seed_ticker in enumerate(scenario_seeds):
                scenario_results = self.propagator.simulate_shock_propagation(
                    seed_ticker=seed_ticker,
                    shock_magnitude=config.shock_magnitude,
                    tickers=tickers,
                    damping_factor=config.damping_factor,
                    max_iterations=config.max_iterations
                )
                
                if not scenario_results.empty:
                    scenario_results['scenario_id'] = f"scenario_{i+1}"
                    scenario_results['scenario_seed'] = seed_ticker
                    all_results.append(scenario_results)
            
            if not all_results:
                raise ValueError("No scenario analysis results generated")
            
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Calculate cross-scenario statistics
            cross_scenario_stats = []
            for ticker in combined_df['ticker'].unique():
                ticker_data = combined_df[combined_df['ticker'] == ticker]
                
                cross_scenario_stats.append({
                    'ticker': ticker,
                    'mean_cross_scenario_impact': ticker_data['final_impact'].mean(),
                    'max_cross_scenario_impact': ticker_data['final_impact'].max(),
                    'min_cross_scenario_impact': ticker_data['final_impact'].min(),
                    'std_cross_scenario_impact': ticker_data['final_impact'].std(),
                    'scenarios_affected': len(ticker_data[ticker_data['final_impact'].abs() > 1e-6]),
                    'final_impact': ticker_data['final_impact'].mean()
                })
            
            results_df = pd.DataFrame(cross_scenario_stats)
            results_df = results_df.sort_values('mean_cross_scenario_impact', key=abs, ascending=False)
            
            # Statistics
            stats = self._calculate_simulation_statistics(results_df, config)
            stats.update({
                'scenarios_analyzed': len(scenario_seeds),
                'cross_scenario_analysis': True
            })
            
            convergence = {
                'method': 'scenario_analysis',
                'scenarios_completed': len(all_results),
                'scenario_seeds': scenario_seeds
            }
            
            return results_df, stats, convergence
            
        except Exception as e:
            self.logger.error(f"Scenario analysis error: {e}")
            raise
    
    def _run_systemic_risk_analysis(self, config: SimulationConfig) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run comprehensive systemic risk analysis."""
        try:
            tickers = self._get_simulation_tickers(config)
            
            # Calculate systemic risk metrics for all stocks
            systemic_risk_df = self.propagator.compute_systemic_risk_metrics(
                tickers=tickers,
                shock_magnitude=config.shock_magnitude
            )
            
            if systemic_risk_df.empty:
                raise ValueError("No systemic risk results generated")
            
            # Add additional risk metrics
            systemic_risk_df['contagion_potential'] = (
                systemic_risk_df['systemic_impact_total'] * 
                systemic_risk_df['affected_nodes_count'] / 
                len(tickers) if len(tickers) > 0 else 0
            )
            
            # Calculate network centrality
            centrality_df = self.graph_builder.compute_centrality_metrics(tickers)
            
            if not centrality_df.empty:
                # Merge with centrality metrics
                systemic_risk_df = systemic_risk_df.merge(
                    centrality_df, on='ticker', how='left'
                )
                
                # Composite risk score
                systemic_risk_df['composite_risk_score'] = (
                    0.4 * systemic_risk_df['systemic_risk_score_normalized'] +
                    0.3 * systemic_risk_df.get('pagerank', 0) +
                    0.2 * systemic_risk_df.get('betweenness_centrality', 0) +
                    0.1 * systemic_risk_df.get('eigenvector_centrality', 0)
                )
            
            # Use systemic_impact_total as final_impact for consistency
            systemic_risk_df['final_impact'] = systemic_risk_df['systemic_impact_total']
            
            # Statistics
            stats = self._calculate_simulation_statistics(systemic_risk_df, config)
            stats.update({
                'systemic_risk_analysis': True,
                'total_nodes_analyzed': len(tickers),
                'network_density': len(systemic_risk_df[systemic_risk_df['affected_nodes_count'] > 1]) / len(systemic_risk_df) if len(systemic_risk_df) > 0 else 0
            })
            
            convergence = {
                'method': 'systemic_risk_analysis',
                'nodes_analyzed': len(tickers),
                'risk_metrics_computed': True
            }
            
            return systemic_risk_df, stats, convergence
            
        except Exception as e:
            self.logger.error(f"Systemic risk analysis error: {e}")
            raise
    
    def _get_simulation_tickers(self, config: SimulationConfig) -> Optional[List[str]]:
        """Get list of tickers for simulation based on config."""
        if config.include_tickers:
            return config.include_tickers
        
        # Get all tickers from network
        try:
            query = "MATCH (c:Company) RETURN c.ticker AS ticker"
            results = neo4j_manager.execute_query(query)
            all_tickers = [r['ticker'] for r in results]
            
            if config.exclude_tickers:
                all_tickers = [t for t in all_tickers if t not in config.exclude_tickers]
            
            return all_tickers
            
        except Exception as e:
            self.logger.warning(f"Could not get tickers from Neo4j: {e}")
            return None
    
    def _calculate_simulation_statistics(self, results_df: pd.DataFrame, 
                                       config: SimulationConfig) -> Dict[str, Any]:
        """Calculate comprehensive statistics for simulation results."""
        if results_df.empty:
            return {}
        
        stats = {
            'total_stocks_analyzed': len(results_df),
            'seed_ticker': config.seed_ticker,
            'shock_magnitude': config.shock_magnitude,
            'damping_factor': config.damping_factor,
            'simulation_type': config.simulation_type.value,
            
            # Impact statistics
            'total_absolute_impact': results_df['final_impact'].abs().sum(),
            'mean_impact': results_df['final_impact'].mean(),
            'median_impact': results_df['final_impact'].median(),
            'std_impact': results_df['final_impact'].std(),
            'max_impact': results_df['final_impact'].max(),
            'min_impact': results_df['final_impact'].min(),
            
            # Distribution statistics
            'stocks_with_positive_impact': len(results_df[results_df['final_impact'] > 0]),
            'stocks_with_negative_impact': len(results_df[results_df['final_impact'] < 0]),
            'stocks_with_significant_impact': len(results_df[results_df['final_impact'].abs() > 0.01]),
            
            # Risk metrics
            'impact_concentration': results_df['final_impact'].abs().max() / results_df['final_impact'].abs().sum() if results_df['final_impact'].abs().sum() > 0 else 0,
            'impact_dispersion': results_df['final_impact'].std() / abs(results_df['final_impact'].mean()) if results_df['final_impact'].mean() != 0 else 0,
        }
        
        # Percentile analysis
        percentiles = [5, 10, 25, 75, 90, 95]
        for p in percentiles:
            stats[f'impact_percentile_{p}'] = results_df['final_impact'].quantile(p/100)
        
        return stats
    
    def _generate_simulation_id(self) -> str:
        """Generate unique simulation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import uuid
        short_uuid = str(uuid.uuid4())[:8]
        return f"sim_{timestamp}_{short_uuid}"
    
    def export_simulation_results(self, results: SimulationResults, 
                                output_path: str, formats: List[str] = ['json', 'csv']) -> Dict[str, str]:
        """Export simulation results to various formats."""
        try:
            exported_files = {}
            
            for fmt in formats:
                if fmt == 'json':
                    # Export complete results as JSON
                    filename = f"{output_path}_{results.simulation_id}.json"
                    
                    export_data = {
                        'simulation_id': results.simulation_id,
                        'config': asdict(results.config),
                        'statistics': results.statistics,
                        'execution_time': results.execution_time,
                        'timestamp': results.timestamp.isoformat(),
                        'convergence_info': results.convergence_info,
                        'results': results.results_df.to_dict('records')
                    }
                    
                    import json
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    
                    exported_files['json'] = filename
                
                elif fmt == 'csv':
                    # Export results DataFrame as CSV
                    filename = f"{output_path}_{results.simulation_id}.csv"
                    results.results_df.to_csv(filename, index=False)
                    exported_files['csv'] = filename
                
                elif fmt == 'excel':
                    # Export to Excel with multiple sheets
                    filename = f"{output_path}_{results.simulation_id}.xlsx"
                    
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        results.results_df.to_excel(writer, sheet_name='Results', index=False)
                        
                        # Statistics sheet
                        stats_df = pd.DataFrame([results.statistics]).T
                        stats_df.columns = ['Value']
                        stats_df.to_excel(writer, sheet_name='Statistics')
                        
                        # Configuration sheet
                        config_df = pd.DataFrame([asdict(results.config)]).T
                        config_df.columns = ['Value']
                        config_df.to_excel(writer, sheet_name='Configuration')
                    
                    exported_files['excel'] = filename
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting simulation results: {e}")
            return {}