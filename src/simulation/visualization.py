"""Visualization components for simulation results."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime

from .engine import SimulationResults, SimulationType
from .analysis import ResultsAnalyzer

logger = logging.getLogger(__name__)


class SimulationVisualizer:
    """Create interactive visualizations for simulation results."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimulationVisualizer")
        self.analyzer = ResultsAnalyzer()
        
        # Color schemes
        self.colors = {
            'positive': '#2E8B57',    # Sea Green
            'negative': '#DC143C',    # Crimson
            'neutral': '#708090',     # Slate Gray
            'primary': '#1f77b4',     # Blue
            'secondary': '#ff7f0e',   # Orange
            'accent': '#2ca02c',      # Green
            'warning': '#d62728',     # Red
            'info': '#9467bd'         # Purple
        }
    
    def create_impact_waterfall(self, results: SimulationResults, 
                               top_n: int = 15) -> go.Figure:
        """Create waterfall chart showing impact progression."""
        try:
            if results.results_df.empty:
                return self._create_empty_figure("No data available for waterfall chart")
            
            # Get top impacted stocks
            df_sorted = results.results_df.nlargest(top_n, 'final_impact', keep='all')
            
            # Prepare data for waterfall
            tickers = df_sorted['ticker'].tolist()
            impacts = df_sorted['final_impact'].tolist()
            
            # Create waterfall chart
            fig = go.Figure(go.Waterfall(
                name="Impact Waterfall",
                orientation="v",
                measure=["relative"] * len(impacts),
                x=tickers,
                textposition="outside",
                text=[f"{impact:.2%}" for impact in impacts],
                y=impacts,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": self.colors['positive']}},
                decreasing={"marker": {"color": self.colors['negative']}},
                totals={"marker": {"color": self.colors['neutral']}}
            ))
            
            fig.update_layout(
                title=f"Impact Waterfall - {results.config.seed_ticker} Shock",
                showlegend=False,
                xaxis_title="Stocks",
                yaxis_title="Impact (%)",
                yaxis_tickformat=".1%",
                height=600,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating waterfall chart: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_impact_distribution(self, results: SimulationResults) -> go.Figure:
        """Create histogram of impact distribution."""
        try:
            if results.results_df.empty:
                return self._create_empty_figure("No data available for distribution")
            
            impacts = results.results_df['final_impact']
            
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=impacts,
                nbinsx=30,
                name='Impact Distribution',
                marker_color=self.colors['primary'],
                opacity=0.7
            ))
            
            # Add mean line
            mean_impact = impacts.mean()
            fig.add_vline(
                x=mean_impact,
                line_dash="dash",
                line_color=self.colors['warning'],
                annotation_text=f"Mean: {mean_impact:.2%}"
            )
            
            # Add median line
            median_impact = impacts.median()
            fig.add_vline(
                x=median_impact,
                line_dash="dot",
                line_color=self.colors['accent'],
                annotation_text=f"Median: {median_impact:.2%}"
            )
            
            fig.update_layout(
                title="Impact Distribution Analysis",
                xaxis_title="Impact (%)",
                yaxis_title="Frequency",
                xaxis_tickformat=".1%",
                height=500,
                template="plotly_white",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating distribution chart: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_network_heatmap(self, results: SimulationResults) -> go.Figure:
        """Create heatmap showing network impact relationships."""
        try:
            if results.results_df.empty:
                return self._create_empty_figure("No data available for heatmap")
            
            # For now, create a simple impact intensity heatmap
            # In a full implementation, this would show actual network relationships
            
            df = results.results_df.head(20).copy()
            df['impact_category'] = pd.cut(df['final_impact'], 
                                         bins=5, 
                                         labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Create matrix for heatmap (simplified version)
            impact_matrix = df['final_impact'].values.reshape(-1, 1)
            
            fig = go.Figure(data=go.Heatmap(
                z=impact_matrix,
                x=['Impact Level'],
                y=df['ticker'],
                colorscale='RdYlBu_r',
                showscale=True,
                hovertemplate='Stock: %{y}<br>Impact: %{z:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Impact Intensity Heatmap",
                height=max(400, len(df) * 25),
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_sector_analysis(self, results: SimulationResults) -> go.Figure:
        """Create sector-wise impact analysis."""
        try:
            sector_analysis = self.analyzer._analyze_sector_impacts(results.results_df)
            
            if not sector_analysis:
                return self._create_empty_figure("No sector data available")
            
            # Prepare data
            sectors = list(sector_analysis.keys())
            avg_impacts = [sector_analysis[sector]['average_impact'] for sector in sectors]
            stock_counts = [sector_analysis[sector]['stock_count'] for sector in sectors]
            
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Average Impact by Sector', 'Stock Count by Sector'),
                specs=[[{"secondary_y": False}, {"type": "pie"}]]
            )
            
            # Bar chart for average impacts
            fig.add_trace(
                go.Bar(
                    x=sectors,
                    y=avg_impacts,
                    name='Avg Impact',
                    marker_color=self.colors['primary'],
                    text=[f"{impact:.2%}" for impact in avg_impacts],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Pie chart for stock distribution
            fig.add_trace(
                go.Pie(
                    labels=sectors,
                    values=stock_counts,
                    name="Stock Distribution",
                    textinfo='label+percent'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Sector Impact Analysis",
                height=500,
                template="plotly_white",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Sectors", row=1, col=1)
            fig.update_yaxes(title_text="Average Impact (%)", tickformat=".1%", row=1, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sector analysis: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_monte_carlo_confidence(self, results: SimulationResults) -> go.Figure:
        """Create confidence interval visualization for Monte Carlo results."""
        try:
            if results.config.simulation_type != SimulationType.MONTE_CARLO:
                return self._create_empty_figure("Not a Monte Carlo simulation")
            
            if results.results_df.empty:
                return self._create_empty_figure("No Monte Carlo data available")
            
            df = results.results_df.head(20).copy()
            
            # Check if Monte Carlo specific columns exist
            if 'percentile_5' not in df.columns or 'percentile_95' not in df.columns:
                return self._create_empty_figure("Monte Carlo confidence data not available")
            
            fig = go.Figure()
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=df['ticker'],
                y=df['percentile_95'],
                mode='lines',
                line=dict(color='rgba(0,100,80,0)'),
                showlegend=False,
                name='Upper 95%'
            ))
            
            fig.add_trace(go.Scatter(
                x=df['ticker'],
                y=df['percentile_5'],
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(0,100,80,0)'),
                name='90% Confidence Interval',
                showlegend=True
            ))
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=df['ticker'],
                y=df['mean_impact'],
                mode='lines+markers',
                name='Mean Impact',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Monte Carlo Confidence Intervals",
                xaxis_title="Stocks",
                yaxis_title="Impact (%)",
                yaxis_tickformat=".1%",
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating Monte Carlo visualization: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_scenario_comparison(self, results_list: List[SimulationResults]) -> go.Figure:
        """Create comparison chart for multiple scenarios."""
        try:
            if not results_list:
                return self._create_empty_figure("No scenarios to compare")
            
            # Get severity rankings
            rankings = self.analyzer._rank_scenarios_by_severity(results_list)
            
            if not rankings:
                return self._create_empty_figure("No ranking data available")
            
            # Prepare data
            scenario_names = [f"{r['seed_ticker']} ({r['shock_magnitude']:.1%})" for r in rankings]
            severity_scores = [r['severity_score'] for r in rankings]
            total_impacts = [r['total_absolute_impact'] for r in rankings]
            max_impacts = [r['max_individual_impact'] for r in rankings]
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Severity Score Ranking',
                    'Total Absolute Impact',
                    'Maximum Individual Impact',
                    'Scenario Comparison'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # Severity score ranking
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=severity_scores,
                    name='Severity Score',
                    marker_color=self.colors['warning'],
                    text=[f"{score:.2f}" for score in severity_scores],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Total impact
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=total_impacts,
                    name='Total Impact',
                    marker_color=self.colors['primary'],
                    text=[f"{impact:.2%}" for impact in total_impacts],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Max individual impact
            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=max_impacts,
                    name='Max Impact',
                    marker_color=self.colors['secondary'],
                    text=[f"{impact:.2%}" for impact in max_impacts],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # Scatter plot comparison
            fig.add_trace(
                go.Scatter(
                    x=total_impacts,
                    y=max_impacts,
                    mode='markers+text',
                    text=scenario_names,
                    textposition='top center',
                    name='Risk Profile',
                    marker=dict(
                        size=[score*2 for score in severity_scores],
                        color=severity_scores,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Severity Score")
                    )
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Scenario Comparison Analysis",
                height=800,
                template="plotly_white",
                showlegend=False
            )
            
            # Update axes
            fig.update_yaxes(title_text="Severity Score", row=1, col=1)
            fig.update_yaxes(title_text="Total Impact (%)", tickformat=".1%", row=1, col=2)
            fig.update_yaxes(title_text="Max Impact (%)", tickformat=".1%", row=2, col=1)
            fig.update_xaxes(title_text="Total Impact (%)", tickformat=".1%", row=2, col=2)
            fig.update_yaxes(title_text="Max Impact (%)", tickformat=".1%", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating scenario comparison: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_risk_dashboard(self, results: SimulationResults) -> Dict[str, go.Figure]:
        """Create comprehensive risk dashboard with multiple charts."""
        try:
            dashboard = {}
            
            # Main impact visualization
            dashboard['impact_waterfall'] = self.create_impact_waterfall(results)
            dashboard['impact_distribution'] = self.create_impact_distribution(results)
            
            # Network and sector analysis
            dashboard['network_heatmap'] = self.create_network_heatmap(results)
            dashboard['sector_analysis'] = self.create_sector_analysis(results)
            
            # Specialized visualizations based on simulation type
            if results.config.simulation_type == SimulationType.MONTE_CARLO:
                dashboard['confidence_intervals'] = self.create_monte_carlo_confidence(results)
            
            # Risk metrics summary
            dashboard['risk_metrics'] = self.create_risk_metrics_summary(results)
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating risk dashboard: {e}")
            return {'error': self._create_empty_figure(f"Dashboard Error: {str(e)}")}
    
    def create_risk_metrics_summary(self, results: SimulationResults) -> go.Figure:
        """Create risk metrics summary visualization."""
        try:
            risk_metrics = self.analyzer._calculate_risk_metrics(results.results_df, results.config)
            
            if not risk_metrics:
                return self._create_empty_figure("No risk metrics available")
            
            # Create gauge charts for key metrics
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=('VaR 95%', 'CVaR 95%', 'Concentration Index', 'Diversification Ratio')
            )
            
            # VaR 95%
            var_95 = risk_metrics.get('value_at_risk', {}).get('95%', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=abs(var_95) * 100,
                    title={'text': "VaR 95% (%)"},
                    gauge={
                        'axis': {'range': [None, 20]},
                        'bar': {'color': self.colors['warning']},
                        'steps': [
                            {'range': [0, 5], 'color': self.colors['accent']},
                            {'range': [5, 10], 'color': self.colors['secondary']},
                            {'range': [10, 20], 'color': self.colors['warning']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 15
                        }
                    }
                ),
                row=1, col=1
            )
            
            # CVaR 95%
            cvar_95 = risk_metrics.get('conditional_var', {}).get('95%', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=abs(cvar_95) * 100,
                    title={'text': "CVaR 95% (%)"},
                    gauge={
                        'axis': {'range': [None, 25]},
                        'bar': {'color': self.colors['warning']},
                        'steps': [
                            {'range': [0, 7], 'color': self.colors['accent']},
                            {'range': [7, 15], 'color': self.colors['secondary']},
                            {'range': [15, 25], 'color': self.colors['warning']}
                        ]
                    }
                ),
                row=1, col=2
            )
            
            # Concentration Index
            concentration = risk_metrics.get('concentration_index', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=concentration * 100,
                    title={'text': "Concentration (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.colors['info']},
                        'steps': [
                            {'range': [0, 30], 'color': self.colors['accent']},
                            {'range': [30, 60], 'color': self.colors['secondary']},
                            {'range': [60, 100], 'color': self.colors['warning']}
                        ]
                    }
                ),
                row=2, col=1
            )
            
            # Diversification Ratio
            diversification = risk_metrics.get('diversification_ratio', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=diversification * 100,
                    title={'text': "Diversification (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.colors['primary']},
                        'steps': [
                            {'range': [0, 30], 'color': self.colors['warning']},
                            {'range': [30, 70], 'color': self.colors['secondary']},
                            {'range': [70, 100], 'color': self.colors['accent']}
                        ]
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Risk Metrics Dashboard",
                height=600,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating risk metrics summary: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def create_time_series_simulation(self, results: SimulationResults,
                                    time_horizon: int = 30) -> go.Figure:
        """Create time series visualization of shock propagation."""
        try:
            # This would require iteration history from the simulation
            # For now, create a simplified decay visualization
            
            if 'iterations_to_peak' not in results.results_df.columns:
                return self._create_empty_figure("Time series data not available")
            
            # Get top 10 most affected stocks
            top_stocks = results.results_df.nlargest(10, 'final_impact')
            
            fig = go.Figure()
            
            for _, stock in top_stocks.iterrows():
                # Simulate decay pattern
                peak_time = stock.get('iterations_to_peak', 5)
                final_impact = stock['final_impact']
                
                # Create time series
                time_points = list(range(time_horizon))
                impact_series = []
                
                for t in time_points:
                    if t <= peak_time:
                        # Ramp up to peak
                        impact = final_impact * (t / peak_time) if peak_time > 0 else final_impact
                    else:
                        # Decay after peak
                        decay_factor = 0.9 ** (t - peak_time)
                        impact = final_impact * decay_factor
                    
                    impact_series.append(impact)
                
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=impact_series,
                    mode='lines+markers',
                    name=stock['ticker'],
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                title="Shock Propagation Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Impact (%)",
                yaxis_tickformat=".1%",
                height=500,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating time series: {e}")
            return self._create_empty_figure(f"Error: {str(e)}")
    
    def export_visualizations(self, dashboard: Dict[str, go.Figure], 
                            output_dir: str, formats: List[str] = ['html', 'png']) -> Dict[str, List[str]]:
        """Export visualizations to various formats."""
        try:
            from pathlib import Path
            import os
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            
            for chart_name, fig in dashboard.items():
                if isinstance(fig, go.Figure):
                    chart_files = []
                    
                    for fmt in formats:
                        if fmt == 'html':
                            filename = output_path / f"{chart_name}.html"
                            fig.write_html(str(filename))
                            chart_files.append(str(filename))
                        
                        elif fmt == 'png':
                            filename = output_path / f"{chart_name}.png"
                            fig.write_image(str(filename), width=1200, height=800)
                            chart_files.append(str(filename))
                        
                        elif fmt == 'pdf':
                            filename = output_path / f"{chart_name}.pdf"
                            fig.write_image(str(filename), width=1200, height=800)
                            chart_files.append(str(filename))
                    
                    exported_files[chart_name] = chart_files
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting visualizations: {e}")
            return {}
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.colors['neutral'])
        )
        fig.update_layout(
            title="No Data Available",
            template="plotly_white",
            height=400
        )
        return fig