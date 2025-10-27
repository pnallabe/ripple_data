"""Results analysis and reporting for simulation outputs."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from .engine import SimulationResults, SimulationConfig, SimulationType
from src.database import pg_manager, neo4j_manager

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Advanced analysis and reporting for simulation results."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ResultsAnalyzer")
    
    def analyze_single_simulation(self, results: SimulationResults) -> Dict[str, Any]:
        """Comprehensive analysis of single simulation results."""
        try:
            if results.results_df.empty:
                return {'error': 'No results to analyze'}
            
            analysis = {
                'simulation_id': results.simulation_id,
                'simulation_type': results.config.simulation_type.value,
                'execution_summary': self._analyze_execution_summary(results),
                'impact_analysis': self._analyze_impact_distribution(results.results_df),
                'risk_metrics': self._calculate_risk_metrics(results.results_df, results.config),
                'network_effects': self._analyze_network_effects(results.results_df, results.config),
                'sector_analysis': self._analyze_sector_impacts(results.results_df),
                'convergence_analysis': self._analyze_convergence(results),
                'recommendations': self._generate_recommendations(results)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing simulation {results.simulation_id}: {e}")
            return {'error': str(e)}
    
    def compare_simulations(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Compare multiple simulation results."""
        try:
            if not results_list:
                return {'error': 'No simulations to compare'}
            
            comparison = {
                'simulation_count': len(results_list),
                'comparison_summary': self._compare_execution_summary(results_list),
                'impact_comparison': self._compare_impact_distributions(results_list),
                'risk_comparison': self._compare_risk_metrics(results_list),
                'sector_comparison': self._compare_sector_effects(results_list),
                'scenario_ranking': self._rank_scenarios_by_severity(results_list),
                'correlation_analysis': self._analyze_result_correlations(results_list),
                'sensitivity_analysis': self._analyze_parameter_sensitivity(results_list)
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing simulations: {e}")
            return {'error': str(e)}
    
    def generate_risk_report(self, results: Union[SimulationResults, List[SimulationResults]],
                           output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """Generate comprehensive risk report."""
        try:
            if isinstance(results, SimulationResults):
                results_list = [results]
            else:
                results_list = results
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'simulations_analyzed': len(results_list),
                    'report_type': 'comprehensive_risk_analysis'
                },
                'executive_summary': self._generate_executive_summary(results_list),
                'detailed_analysis': self._generate_detailed_analysis(results_list),
                'risk_assessment': self._generate_risk_assessment(results_list),
                'recommendations': self._generate_comprehensive_recommendations(results_list),
                'appendix': self._generate_report_appendix(results_list)
            }
            
            if output_format == 'json':
                return json.dumps(report, indent=2, default=str)
            elif output_format == 'markdown':
                return self._format_report_as_markdown(report)
            else:
                return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {'error': str(e)}
    
    def calculate_portfolio_impact(self, results: SimulationResults,
                                 portfolio_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Calculate portfolio-level impact from simulation results."""
        try:
            if results.results_df.empty:
                return {'error': 'No results for portfolio analysis'}
            
            # Get market cap data for weighting if no weights provided
            if portfolio_weights is None:
                portfolio_weights = self._get_market_cap_weights(results.results_df['ticker'].tolist())
            
            portfolio_analysis = {
                'total_portfolio_impact': 0.0,
                'weighted_average_impact': 0.0,
                'impact_variance': 0.0,
                'concentration_risk': 0.0,
                'sector_contributions': {},
                'top_contributors': [],
                'portfolio_var': 0.0,
                'portfolio_cvar': 0.0
            }
            
            # Calculate weighted impacts
            weighted_impacts = []
            for _, stock in results.results_df.iterrows():
                ticker = stock['ticker']
                impact = stock['final_impact']
                weight = portfolio_weights.get(ticker, 0.0)
                
                weighted_impact = weight * impact
                weighted_impacts.append({
                    'ticker': ticker,
                    'weight': weight,
                    'impact': impact,
                    'weighted_impact': weighted_impact,
                    'contribution_to_portfolio': weighted_impact
                })
            
            weighted_df = pd.DataFrame(weighted_impacts)
            
            # Portfolio metrics
            portfolio_analysis['total_portfolio_impact'] = weighted_df['weighted_impact'].sum()
            portfolio_analysis['weighted_average_impact'] = weighted_df['weighted_impact'].mean()
            portfolio_analysis['impact_variance'] = weighted_df['weighted_impact'].var()
            
            # Concentration risk (HHI of weights)
            portfolio_analysis['concentration_risk'] = sum(w**2 for w in portfolio_weights.values())
            
            # Top contributors
            portfolio_analysis['top_contributors'] = (
                weighted_df.nlargest(10, 'weighted_impact')[['ticker', 'weighted_impact', 'weight']]
                .to_dict('records')
            )
            
            # Risk metrics
            if len(weighted_df) > 0:
                portfolio_analysis['portfolio_var'] = weighted_df['weighted_impact'].quantile(0.05)
                portfolio_analysis['portfolio_cvar'] = (
                    weighted_df[weighted_df['weighted_impact'] <= portfolio_analysis['portfolio_var']]
                    ['weighted_impact'].mean()
                )
            
            # Sector analysis
            sector_impacts = self._analyze_sector_impacts(results.results_df)
            if sector_impacts:
                portfolio_analysis['sector_contributions'] = sector_impacts
            
            return portfolio_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio impact: {e}")
            return {'error': str(e)}
    
    def identify_systemic_nodes(self, results_list: List[SimulationResults],
                              threshold: float = 0.01) -> Dict[str, Any]:
        """Identify systemically important nodes from multiple simulations."""
        try:
            if not results_list:
                return {'error': 'No simulations provided'}
            
            # Aggregate impact data across simulations
            all_impacts = []
            
            for results in results_list:
                if not results.results_df.empty:
                    sim_data = results.results_df.copy()
                    sim_data['simulation_id'] = results.simulation_id
                    sim_data['seed_ticker'] = results.config.seed_ticker
                    all_impacts.append(sim_data)
            
            if not all_impacts:
                return {'systemic_nodes': []}
            
            combined_df = pd.concat(all_impacts, ignore_index=True)
            
            # Calculate systemic importance metrics
            systemic_analysis = []
            
            for ticker in combined_df['ticker'].unique():
                ticker_data = combined_df[combined_df['ticker'] == ticker]
                
                # Metrics
                avg_impact = ticker_data['final_impact'].mean()
                max_impact = ticker_data['final_impact'].max()
                impact_volatility = ticker_data['final_impact'].std()
                times_significant = len(ticker_data[ticker_data['final_impact'].abs() > threshold])
                
                # Systemic importance score
                systemic_score = (
                    0.4 * abs(avg_impact) +
                    0.3 * abs(max_impact) +
                    0.2 * impact_volatility +
                    0.1 * (times_significant / len(ticker_data))
                )
                
                systemic_analysis.append({
                    'ticker': ticker,
                    'average_impact': avg_impact,
                    'max_impact': max_impact,
                    'impact_volatility': impact_volatility,
                    'times_significant': times_significant,
                    'systemic_importance_score': systemic_score,
                    'simulations_analyzed': len(ticker_data)
                })
            
            systemic_df = pd.DataFrame(systemic_analysis)
            systemic_df = systemic_df.sort_values('systemic_importance_score', ascending=False)
            
            # Identify top systemic nodes
            top_systemic = systemic_df.head(20).to_dict('records')
            
            return {
                'systemic_nodes': top_systemic,
                'analysis_summary': {
                    'total_nodes_analyzed': len(systemic_df),
                    'simulations_processed': len(results_list),
                    'significance_threshold': threshold,
                    'top_systemic_count': len(top_systemic)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying systemic nodes: {e}")
            return {'error': str(e)}
    
    def _analyze_execution_summary(self, results: SimulationResults) -> Dict[str, Any]:
        """Analyze execution summary."""
        return {
            'execution_time': results.execution_time,
            'timestamp': results.timestamp.isoformat(),
            'simulation_type': results.config.simulation_type.value,
            'seed_ticker': results.config.seed_ticker,
            'shock_magnitude': results.config.shock_magnitude,
            'damping_factor': results.config.damping_factor,
            'converged': results.convergence_info.get('converged', False),
            'iterations': results.convergence_info.get('iterations', 0),
            'stocks_analyzed': len(results.results_df)
        }
    
    def _analyze_impact_distribution(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact distribution statistics."""
        if results_df.empty:
            return {}
        
        impacts = results_df['final_impact']
        
        return {
            'total_absolute_impact': impacts.abs().sum(),
            'mean_impact': impacts.mean(),
            'median_impact': impacts.median(),
            'std_impact': impacts.std(),
            'skewness': impacts.skew(),
            'kurtosis': impacts.kurtosis(),
            'min_impact': impacts.min(),
            'max_impact': impacts.max(),
            'positive_impacts': len(impacts[impacts > 0]),
            'negative_impacts': len(impacts[impacts < 0]),
            'significant_impacts': len(impacts[impacts.abs() > 0.01]),
            'percentiles': {
                '5th': impacts.quantile(0.05),
                '25th': impacts.quantile(0.25),
                '75th': impacts.quantile(0.75),
                '95th': impacts.quantile(0.95)
            }
        }
    
    def _calculate_risk_metrics(self, results_df: pd.DataFrame, config: SimulationConfig) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        if results_df.empty:
            return {}
        
        impacts = results_df['final_impact']
        
        # Basic risk metrics
        var_95 = impacts.quantile(0.05)
        var_99 = impacts.quantile(0.01)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = impacts[impacts <= var_95].mean() if len(impacts[impacts <= var_95]) > 0 else var_95
        cvar_99 = impacts[impacts <= var_99].mean() if len(impacts[impacts <= var_99]) > 0 else var_99
        
        return {
            'value_at_risk': {
                '95%': var_95,
                '99%': var_99
            },
            'conditional_var': {
                '95%': cvar_95,
                '99%': cvar_99
            },
            'maximum_loss': impacts.min(),
            'expected_loss': impacts[impacts < 0].mean() if len(impacts[impacts < 0]) > 0 else 0,
            'tail_risk_ratio': abs(cvar_95 / var_95) if var_95 != 0 else 0,
            'concentration_index': (impacts**2).sum() / (impacts.sum()**2) if impacts.sum() != 0 else 0,
            'diversification_ratio': len(impacts[impacts.abs() > 0.001]) / len(impacts) if len(impacts) > 0 else 0
        }
    
    def _analyze_network_effects(self, results_df: pd.DataFrame, config: SimulationConfig) -> Dict[str, Any]:
        """Analyze network propagation effects."""
        if results_df.empty:
            return {}
        
        # Network effect metrics
        seed_impact = results_df[results_df['ticker'] == config.seed_ticker]['final_impact'].iloc[0] if len(results_df[results_df['ticker'] == config.seed_ticker]) > 0 else config.shock_magnitude
        
        secondary_impacts = results_df[results_df['ticker'] != config.seed_ticker]['final_impact']
        
        return {
            'primary_shock': config.shock_magnitude,
            'seed_final_impact': seed_impact,
            'amplification_factor': abs(seed_impact / config.shock_magnitude) if config.shock_magnitude != 0 else 1,
            'secondary_impact_total': secondary_impacts.abs().sum(),
            'secondary_impact_mean': secondary_impacts.mean(),
            'contagion_ratio': len(secondary_impacts[secondary_impacts.abs() > 0.001]) / len(secondary_impacts) if len(secondary_impacts) > 0 else 0,
            'network_multiplier': secondary_impacts.abs().sum() / abs(config.shock_magnitude) if config.shock_magnitude != 0 else 0,
            'cascade_depth': self._estimate_cascade_depth(results_df, config)
        }
    
    def _analyze_sector_impacts(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impacts by sector."""
        try:
            # Get sector information
            tickers = results_df['ticker'].tolist()
            if not tickers:
                return {}
            
            # Query sector data
            query = """
            SELECT ticker, sector, industry 
            FROM companies 
            WHERE ticker = ANY(%s)
            """
            sector_data = pd.DataFrame(pg_manager.execute_query(query, (tickers,)))
            
            if sector_data.empty:
                return {}
            
            # Merge with results
            results_with_sectors = results_df.merge(sector_data, on='ticker', how='left')
            
            # Analyze by sector
            sector_analysis = {}
            
            for sector in results_with_sectors['sector'].dropna().unique():
                sector_stocks = results_with_sectors[results_with_sectors['sector'] == sector]
                
                sector_analysis[sector] = {
                    'stock_count': len(sector_stocks),
                    'total_impact': sector_stocks['final_impact'].sum(),
                    'average_impact': sector_stocks['final_impact'].mean(),
                    'max_impact': sector_stocks['final_impact'].max(),
                    'min_impact': sector_stocks['final_impact'].min(),
                    'most_affected_stock': sector_stocks.loc[sector_stocks['final_impact'].abs().idxmax(), 'ticker'] if len(sector_stocks) > 0 else None
                }
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing sector impacts: {e}")
            return {}
    
    def _analyze_convergence(self, results: SimulationResults) -> Dict[str, Any]:
        """Analyze simulation convergence."""
        convergence_info = results.convergence_info
        
        analysis = {
            'converged': convergence_info.get('converged', False),
            'iterations_used': convergence_info.get('iterations', 0),
            'max_iterations_allowed': results.config.max_iterations,
            'convergence_threshold': results.config.convergence_threshold,
            'method': convergence_info.get('method', 'unknown')
        }
        
        # Add method-specific analysis
        if results.config.simulation_type == SimulationType.MONTE_CARLO:
            analysis['monte_carlo_runs'] = convergence_info.get('runs_completed', 0)
            analysis['stability_achieved'] = convergence_info.get('runs_completed', 0) >= results.config.monte_carlo_runs
        
        return analysis
    
    def _generate_recommendations(self, results: SimulationResults) -> List[Dict[str, str]]:
        """Generate actionable recommendations from results."""
        recommendations = []
        
        if results.results_df.empty:
            return recommendations
        
        # Analyze results for recommendations
        impacts = results.results_df['final_impact']
        most_affected = results.results_df.loc[impacts.abs().idxmax()]
        
        # High impact recommendation
        if abs(most_affected['final_impact']) > 0.1:
            recommendations.append({
                'category': 'Risk Management',
                'priority': 'High',
                'recommendation': f"Monitor {most_affected['ticker']} closely as it shows the highest impact ({most_affected['final_impact']:.2%})",
                'rationale': 'High individual stock impact suggests concentrated risk'
            })
        
        # Systemic risk recommendation
        significant_impacts = len(impacts[impacts.abs() > 0.01])
        if significant_impacts > len(impacts) * 0.3:
            recommendations.append({
                'category': 'Systemic Risk',
                'priority': 'Medium',
                'recommendation': f"Review system-wide risk controls as {significant_impacts} stocks show significant impact",
                'rationale': 'Widespread impact suggests systemic vulnerability'
            })
        
        # Convergence recommendation
        if not results.convergence_info.get('converged', False):
            recommendations.append({
                'category': 'Model Calibration',
                'priority': 'Medium',
                'recommendation': 'Consider increasing max iterations or adjusting convergence threshold',
                'rationale': 'Simulation did not fully converge'
            })
        
        return recommendations
    
    def _compare_execution_summary(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Compare execution summaries across simulations."""
        return {
            'total_simulations': len(results_list),
            'simulation_types': list(set(r.config.simulation_type.value for r in results_list)),
            'average_execution_time': np.mean([r.execution_time for r in results_list]),
            'total_execution_time': sum(r.execution_time for r in results_list),
            'convergence_rate': sum(1 for r in results_list if r.convergence_info.get('converged', False)) / len(results_list)
        }
    
    def _compare_impact_distributions(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Compare impact distributions across simulations."""
        all_impacts = []
        simulation_summaries = []
        
        for results in results_list:
            if not results.results_df.empty:
                impacts = results.results_df['final_impact']
                all_impacts.extend(impacts.tolist())
                
                simulation_summaries.append({
                    'simulation_id': results.simulation_id,
                    'seed_ticker': results.config.seed_ticker,
                    'mean_impact': impacts.mean(),
                    'max_impact': impacts.max(),
                    'total_absolute_impact': impacts.abs().sum()
                })
        
        return {
            'simulation_summaries': simulation_summaries,
            'aggregate_statistics': {
                'overall_mean': np.mean(all_impacts) if all_impacts else 0,
                'overall_std': np.std(all_impacts) if all_impacts else 0,
                'cross_simulation_correlation': self._calculate_cross_simulation_correlation(results_list)
            }
        }
    
    def _compare_risk_metrics(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Compare risk metrics across simulations."""
        risk_comparisons = []
        
        for results in results_list:
            if not results.results_df.empty:
                risk_metrics = self._calculate_risk_metrics(results.results_df, results.config)
                risk_metrics['simulation_id'] = results.simulation_id
                risk_metrics['seed_ticker'] = results.config.seed_ticker
                risk_comparisons.append(risk_metrics)
        
        return risk_comparisons
    
    def _rank_scenarios_by_severity(self, results_list: List[SimulationResults]) -> List[Dict[str, Any]]:
        """Rank scenarios by severity of impact."""
        scenario_rankings = []
        
        for results in results_list:
            if not results.results_df.empty:
                severity_score = (
                    results.results_df['final_impact'].abs().sum() * 0.4 +
                    abs(results.results_df['final_impact'].min()) * 0.3 +
                    len(results.results_df[results.results_df['final_impact'].abs() > 0.01]) * 0.3
                )
                
                scenario_rankings.append({
                    'simulation_id': results.simulation_id,
                    'seed_ticker': results.config.seed_ticker,
                    'shock_magnitude': results.config.shock_magnitude,
                    'severity_score': severity_score,
                    'total_absolute_impact': results.results_df['final_impact'].abs().sum(),
                    'max_individual_impact': abs(results.results_df['final_impact'].min())
                })
        
        return sorted(scenario_rankings, key=lambda x: x['severity_score'], reverse=True)
    
    def _estimate_cascade_depth(self, results_df: pd.DataFrame, config: SimulationConfig) -> int:
        """Estimate cascade depth based on impact decay."""
        # This is a simplified estimation - in practice you'd need iteration history
        impacts = results_df['final_impact'].abs().sort_values(ascending=False)
        
        # Find where impact drops below 1% of max
        max_impact = impacts.iloc[0] if len(impacts) > 0 else 0
        threshold = max_impact * 0.01
        
        cascade_depth = len(impacts[impacts > threshold])
        return min(cascade_depth, 5)  # Cap at 5 for reasonable display
    
    def _calculate_cross_simulation_correlation(self, results_list: List[SimulationResults]) -> float:
        """Calculate correlation between simulation results."""
        if len(results_list) < 2:
            return 0.0
        
        try:
            # Get common tickers across simulations
            common_tickers = set(results_list[0].results_df['ticker'])
            for results in results_list[1:]:
                common_tickers = common_tickers.intersection(set(results.results_df['ticker']))
            
            if len(common_tickers) < 2:
                return 0.0
            
            # Build correlation matrix
            impact_data = {}
            for i, results in enumerate(results_list):
                sim_impacts = {}
                for _, row in results.results_df.iterrows():
                    if row['ticker'] in common_tickers:
                        sim_impacts[row['ticker']] = row['final_impact']
                impact_data[f'sim_{i}'] = sim_impacts
            
            # Calculate average correlation
            impact_df = pd.DataFrame(impact_data)
            correlation_matrix = impact_df.corr()
            
            # Return average off-diagonal correlation
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    correlations.append(correlation_matrix.iloc[i, j])
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-simulation correlation: {e}")
            return 0.0
    
    def _get_market_cap_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Get market cap weights for portfolio analysis."""
        try:
            query = """
            SELECT ticker, market_cap 
            FROM companies 
            WHERE ticker = ANY(%s) AND market_cap IS NOT NULL
            """
            
            market_caps = pd.DataFrame(pg_manager.execute_query(query, (tickers,)))
            
            if market_caps.empty:
                # Equal weights if no market cap data
                return {ticker: 1.0/len(tickers) for ticker in tickers}
            
            # Calculate weights
            total_market_cap = market_caps['market_cap'].sum()
            weights = {}
            
            for _, row in market_caps.iterrows():
                weights[row['ticker']] = row['market_cap'] / total_market_cap
            
            # Add zero weights for missing tickers
            for ticker in tickers:
                if ticker not in weights:
                    weights[ticker] = 0.0
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error getting market cap weights: {e}")
            return {ticker: 1.0/len(tickers) for ticker in tickers}
    
    def _generate_executive_summary(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Generate executive summary for risk report."""
        if not results_list:
            return {}
        
        # Aggregate key metrics
        total_simulations = len(results_list)
        avg_execution_time = np.mean([r.execution_time for r in results_list])
        
        # Find most severe scenario
        severity_rankings = self._rank_scenarios_by_severity(results_list)
        most_severe = severity_rankings[0] if severity_rankings else {}
        
        return {
            'analysis_scope': {
                'simulations_analyzed': total_simulations,
                'analysis_date': datetime.now().date().isoformat(),
                'average_execution_time': avg_execution_time
            },
            'key_findings': {
                'most_severe_scenario': most_severe.get('seed_ticker'),
                'highest_severity_score': most_severe.get('severity_score', 0),
                'average_convergence_rate': sum(1 for r in results_list if r.convergence_info.get('converged', False)) / total_simulations
            },
            'risk_assessment': 'Detailed analysis follows in subsequent sections'
        }
    
    def _generate_detailed_analysis(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Generate detailed analysis section."""
        return {
            'impact_analysis': self._compare_impact_distributions(results_list),
            'risk_metrics': self._compare_risk_metrics(results_list),
            'scenario_comparison': self._rank_scenarios_by_severity(results_list),
            'systemic_analysis': self.identify_systemic_nodes(results_list)
        }
    
    def _generate_risk_assessment(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Generate risk assessment section."""
        # This would include more sophisticated risk assessment logic
        return {
            'overall_risk_level': 'Medium',  # This would be calculated
            'key_vulnerabilities': [],
            'risk_factors': []
        }
    
    def _generate_comprehensive_recommendations(self, results_list: List[SimulationResults]) -> List[Dict[str, str]]:
        """Generate comprehensive recommendations."""
        all_recommendations = []
        
        for results in results_list:
            recommendations = self._generate_recommendations(results)
            all_recommendations.extend(recommendations)
        
        # Deduplicate and prioritize
        return all_recommendations[:10]  # Return top 10
    
    def _generate_report_appendix(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Generate report appendix with technical details."""
        return {
            'methodology': 'Matrix-based propagation simulation with network effects',
            'simulation_parameters': [
                {
                    'simulation_id': r.simulation_id,
                    'type': r.config.simulation_type.value,
                    'shock_magnitude': r.config.shock_magnitude,
                    'damping_factor': r.config.damping_factor
                }
                for r in results_list
            ],
            'data_sources': ['Neo4j graph database', 'PostgreSQL company data'],
            'limitations': [
                'Model assumes static network structure',
                'Does not account for regulatory interventions',
                'Historical correlations may not predict future behavior'
            ]
        }
    
    def _format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown text."""
        # This would implement full Markdown formatting
        # For now, return a simplified version
        return f"""
# Risk Analysis Report

## Executive Summary
- Simulations Analyzed: {report['executive_summary'].get('analysis_scope', {}).get('simulations_analyzed', 0)}
- Analysis Date: {report['executive_summary'].get('analysis_scope', {}).get('analysis_date', 'Unknown')}

## Key Findings
{json.dumps(report['executive_summary'].get('key_findings', {}), indent=2)}

## Detailed Analysis
{json.dumps(report['detailed_analysis'], indent=2)}

## Recommendations
{json.dumps(report['recommendations'], indent=2)}
"""
        
    def _analyze_parameter_sensitivity(self, results_list: List[SimulationResults]) -> Dict[str, Any]:
        """Analyze sensitivity to simulation parameters."""
        # This would implement parameter sensitivity analysis
        return {
            'damping_factor_sensitivity': 'Analysis not implemented',
            'shock_magnitude_sensitivity': 'Analysis not implemented'
        }