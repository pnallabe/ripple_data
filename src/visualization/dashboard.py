"""Dash web application for ripple effect visualization."""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
import logging

from src.analytics import RipplePropagator, GraphBuilder, CorrelationAnalyzer
from src.database import pg_manager, neo4j_manager
from config.settings import config

logger = logging.getLogger(__name__)


class RippleDashboard:
    """Main dashboard application for ripple effect analysis."""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.propagator = RipplePropagator()
        self.graph_builder = GraphBuilder()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Initialize layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the main dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Stock Dependency & Ripple Effect Analysis Platform", 
                       className="header-title"),
                html.P("Analyze interdependencies and simulate ripple effects across stock networks",
                      className="header-subtitle")
            ], className="header"),
            
            # Main content
            html.Div([
                # Control Panel
                html.Div([
                    html.H3("Simulation Controls"),
                    
                    # Ticker selection
                    html.Div([
                        html.Label("Seed Ticker:"),
                        dcc.Dropdown(
                            id='seed-ticker-dropdown',
                            options=[],
                            value=None,
                            placeholder="Select seed ticker..."
                        )
                    ], className="control-group"),
                    
                    # Shock magnitude
                    html.Div([
                        html.Label("Shock Magnitude (%):"),
                        dcc.Slider(
                            id='shock-magnitude-slider',
                            min=-50,
                            max=50,
                            step=1,
                            value=-5,
                            marks={i: f"{i}%" for i in range(-50, 51, 10)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="control-group"),
                    
                    # Damping factor
                    html.Div([
                        html.Label("Damping Factor:"),
                        dcc.Slider(
                            id='damping-factor-slider',
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            value=0.85,
                            marks={i/10: f"{i/10:.1f}" for i in range(1, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="control-group"),
                    
                    # Run simulation button
                    html.Div([
                        html.Button("Run Simulation", 
                                  id="run-simulation-btn", 
                                  className="btn-primary",
                                  n_clicks=0)
                    ], className="control-group"),
                    
                    # Status indicator
                    html.Div(id="simulation-status", className="status-indicator")
                    
                ], className="control-panel"),
                
                # Visualization area
                html.Div([
                    # Tabs for different views
                    dcc.Tabs(id="visualization-tabs", value='network-view', children=[
                        dcc.Tab(label='Network View', value='network-view'),
                        dcc.Tab(label='Impact Analysis', value='impact-analysis'),
                        dcc.Tab(label='Time Series', value='time-series'),
                        dcc.Tab(label='Sector Analysis', value='sector-analysis')
                    ]),
                    
                    # Tab content
                    html.Div(id="tab-content")
                    
                ], className="visualization-area")
                
            ], className="main-content"),
            
            # Store components for data
            dcc.Store(id='simulation-results-store'),
            dcc.Store(id='network-data-store'),
            dcc.Store(id='ticker-list-store')
            
        ], className="app-container")
    
    def setup_callbacks(self):
        """Setup all dashboard callbacks."""
        
        @self.app.callback(
            [Output('ticker-list-store', 'data'),
             Output('seed-ticker-dropdown', 'options')],
            [Input('ticker-list-store', 'data')]
        )
        def load_ticker_list(stored_tickers):
            """Load available tickers from database."""
            try:
                if not stored_tickers:
                    # Query available tickers
                    query = """
                    SELECT DISTINCT ticker, name 
                    FROM companies 
                    WHERE ticker IS NOT NULL 
                    ORDER BY ticker
                    """
                    tickers_df = pg_manager.read_dataframe(query)
                    tickers_data = tickers_df.to_dict('records')
                else:
                    tickers_data = stored_tickers
                
                # Create dropdown options
                options = [
                    {'label': f"{row['ticker']} - {row['name'][:50]}...", 'value': row['ticker']}
                    for row in tickers_data
                ]
                
                return tickers_data, options
                
            except Exception as e:
                logger.error(f"Error loading tickers: {e}")
                return [], []
        
        @self.app.callback(
            [Output('simulation-results-store', 'data'),
             Output('simulation-status', 'children')],
            [Input('run-simulation-btn', 'n_clicks')],
            [State('seed-ticker-dropdown', 'value'),
             State('shock-magnitude-slider', 'value'),
             State('damping-factor-slider', 'value')]
        )
        def run_simulation(n_clicks, seed_ticker, shock_magnitude, damping_factor):
            """Run ripple effect simulation."""
            if n_clicks == 0 or not seed_ticker:
                return {}, ""
            
            try:
                # Show loading status
                status = html.Div([
                    html.I(className="fas fa-spinner fa-spin"),
                    html.Span(" Running simulation...", style={"margin-left": "10px"})
                ], className="status-loading")
                
                # Run simulation
                shock_pct = shock_magnitude / 100.0
                results = self.propagator.simulate_shock_propagation(
                    seed_ticker=seed_ticker,
                    shock_magnitude=shock_pct,
                    damping_factor=damping_factor
                )
                
                if results.empty:
                    return {}, html.Div("Simulation failed - no results", className="status-error")
                
                # Convert to JSON serializable format
                results_dict = results.to_dict('records')
                
                success_status = html.Div([
                    html.I(className="fas fa-check-circle"),
                    html.Span(f" Simulation completed - {len(results)} stocks analyzed", 
                             style={"margin-left": "10px"})
                ], className="status-success")
                
                return results_dict, success_status
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                error_status = html.Div([
                    html.I(className="fas fa-exclamation-triangle"),
                    html.Span(f" Error: {str(e)}", style={"margin-left": "10px"})
                ], className="status-error")
                return {}, error_status
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('visualization-tabs', 'value'),
             Input('simulation-results-store', 'data')]
        )
        def update_tab_content(active_tab, simulation_results):
            """Update content based on active tab."""
            if not simulation_results:
                return html.Div("Run a simulation to see results", className="no-data-message")
            
            df_results = pd.DataFrame(simulation_results)
            
            if active_tab == 'network-view':
                return self._create_network_view(df_results)
            elif active_tab == 'impact-analysis':
                return self._create_impact_analysis(df_results)
            elif active_tab == 'time-series':
                return self._create_time_series_view(df_results)
            elif active_tab == 'sector-analysis':
                return self._create_sector_analysis(df_results)
            
            return html.Div("Tab content not implemented")
    
    def _create_network_view(self, df_results: pd.DataFrame) -> html.Div:
        """Create network visualization view."""
        try:
            # Create network graph
            network_fig = self._create_network_graph(df_results)
            
            return html.Div([
                html.H4("Network Impact Visualization"),
                dcc.Graph(figure=network_fig, id="network-graph"),
                html.P("Node size represents impact magnitude. Edge thickness represents relationship strength.",
                      className="chart-caption")
            ])
            
        except Exception as e:
            logger.error(f"Error creating network view: {e}")
            return html.Div(f"Error creating network view: {e}", className="error-message")
    
    def _create_impact_analysis(self, df_results: pd.DataFrame) -> html.Div:
        """Create impact analysis view."""
        try:
            # Top impacted stocks table
            top_impacted = df_results.nlargest(20, 'final_impact', keep='all')
            
            # Impact distribution chart
            impact_fig = px.bar(
                top_impacted,
                x='ticker',
                y='final_impact',
                title="Top 20 Most Impacted Stocks",
                labels={'final_impact': 'Impact (%)', 'ticker': 'Ticker'}
            )
            impact_fig.update_layout(xaxis_tickangle=-45)
            
            # Summary statistics
            total_impact = df_results['final_impact'].abs().sum()
            affected_stocks = (df_results['final_impact'].abs() > 0.001).sum()
            
            return html.Div([
                # Summary cards
                html.Div([
                    html.Div([
                        html.H4(f"{total_impact:.4f}"),
                        html.P("Total Impact")
                    ], className="summary-card"),
                    
                    html.Div([
                        html.H4(f"{affected_stocks}"),
                        html.P("Affected Stocks")
                    ], className="summary-card"),
                    
                    html.Div([
                        html.H4(f"{df_results['total_iterations'].iloc[0]}"),
                        html.P("Iterations")
                    ], className="summary-card")
                ], className="summary-cards"),
                
                # Impact chart
                dcc.Graph(figure=impact_fig),
                
                # Detailed table
                html.H4("Detailed Impact Analysis"),
                dash_table.DataTable(
                    data=top_impacted[['ticker', 'final_impact', 'cumulative_impact', 
                                     'max_impact', 'iterations_to_peak']].to_dict('records'),
                    columns=[
                        {'name': 'Ticker', 'id': 'ticker'},
                        {'name': 'Final Impact (%)', 'id': 'final_impact', 'type': 'numeric', 'format': FormatTemplate.percentage(4)},
                        {'name': 'Cumulative Impact (%)', 'id': 'cumulative_impact', 'type': 'numeric', 'format': FormatTemplate.percentage(4)},
                        {'name': 'Max Impact (%)', 'id': 'max_impact', 'type': 'numeric', 'format': FormatTemplate.percentage(4)},
                        {'name': 'Iterations to Peak', 'id': 'iterations_to_peak', 'type': 'numeric'}
                    ],
                    sort_action="native",
                    page_size=20,
                    style_cell={'textAlign': 'left'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{final_impact} > 0'},
                            'backgroundColor': '#d4edda',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{final_impact} < 0'},
                            'backgroundColor': '#f8d7da',
                            'color': 'black',
                        }
                    ]
                )
            ])
            
        except Exception as e:
            logger.error(f"Error creating impact analysis: {e}")
            return html.Div(f"Error creating impact analysis: {e}", className="error-message")
    
    def _create_time_series_view(self, df_results: pd.DataFrame) -> html.Div:
        """Create time series propagation view (placeholder)."""
        return html.Div([
            html.H4("Time Series Propagation"),
            html.P("Time series visualization would show how the shock propagates over iterations."),
            html.P("This requires storing iteration history from the simulation.")
        ])
    
    def _create_sector_analysis(self, df_results: pd.DataFrame) -> html.Div:
        """Create sector-based analysis view."""
        try:
            # Get sector information
            tickers = df_results['ticker'].tolist()
            sector_query = f"""
            SELECT ticker, sector, market_cap
            FROM companies 
            WHERE ticker = ANY(ARRAY{tickers})
            """
            
            sector_df = pg_manager.read_dataframe(sector_query)
            
            # Merge with results
            df_with_sectors = df_results.merge(sector_df, on='ticker', how='left')
            
            # Sector impact aggregation
            sector_impact = df_with_sectors.groupby('sector').agg({
                'final_impact': ['sum', 'mean', 'count'],
                'market_cap': 'sum'
            }).round(6)
            
            sector_impact.columns = ['total_impact', 'avg_impact', 'stock_count', 'total_market_cap']
            sector_impact = sector_impact.reset_index()
            
            # Create sector chart
            sector_fig = px.bar(
                sector_impact,
                x='sector',
                y='total_impact',
                title="Impact by Sector"
            )
            sector_fig.update_layout(xaxis_tickangle=-45)
            
            return html.Div([
                html.H4("Sector Impact Analysis"),
                dcc.Graph(figure=sector_fig),
                
                html.H5("Sector Summary"),
                dash_table.DataTable(
                    data=sector_impact.to_dict('records'),
                    columns=[
                        {'name': 'Sector', 'id': 'sector'},
                        {'name': 'Total Impact', 'id': 'total_impact', 'type': 'numeric'},
                        {'name': 'Average Impact', 'id': 'avg_impact', 'type': 'numeric'},
                        {'name': 'Stock Count', 'id': 'stock_count', 'type': 'numeric'},
                        {'name': 'Total Market Cap', 'id': 'total_market_cap', 'type': 'numeric'}
                    ],
                    sort_action="native"
                )
            ])
            
        except Exception as e:
            logger.error(f"Error creating sector analysis: {e}")
            return html.Div(f"Error creating sector analysis: {e}", className="error-message")
    
    def _create_network_graph(self, df_results: pd.DataFrame) -> go.Figure:
        """Create network graph visualization."""
        # This is a simplified network visualization
        # In production, you'd use more sophisticated graph layout algorithms
        
        # Create scatter plot as placeholder for network
        fig = go.Figure()
        
        # Add nodes (stocks)
        fig.add_trace(go.Scatter(
            x=np.random.randn(len(df_results)),  # Random layout for demo
            y=np.random.randn(len(df_results)),
            mode='markers+text',
            marker=dict(
                size=np.abs(df_results['final_impact']) * 1000 + 10,
                color=df_results['final_impact'],
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Impact")
            ),
            text=df_results['ticker'],
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>Impact: %{marker.color:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Stock Dependency Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def run_server(self, debug: bool = False, port: int = 8050):
        """Run the Dash server."""
        self.app.run_server(debug=debug, port=port)


# CSS styling
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
]

# Custom CSS (would be in assets folder in production)
custom_css = """
.app-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    text-align: center;
}

.header-title {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 300;
}

.header-subtitle {
    margin: 0.5rem 0 0;
    font-size: 1.2rem;
    opacity: 0.9;
}

.main-content {
    display: flex;
    min-height: 80vh;
}

.control-panel {
    width: 300px;
    background: #f8f9fa;
    padding: 2rem;
    border-right: 1px solid #dee2e6;
}

.visualization-area {
    flex: 1;
    padding: 2rem;
}

.control-group {
    margin-bottom: 1.5rem;
}

.btn-primary {
    background: #667eea;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.25rem;
    cursor: pointer;
    font-size: 1rem;
    width: 100%;
}

.btn-primary:hover {
    background: #5a6fd8;
}

.summary-cards {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}

.summary-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
    flex: 1;
}

.summary-card h4 {
    margin: 0;
    font-size: 2rem;
    color: #667eea;
}

.summary-card p {
    margin: 0.5rem 0 0;
    color: #6c757d;
}

.status-loading {
    color: #ffc107;
}

.status-success {
    color: #28a745;
}

.status-error {
    color: #dc3545;
}

.no-data-message {
    text-align: center;
    color: #6c757d;
    font-size: 1.1rem;
    margin-top: 3rem;
}

.error-message {
    color: #dc3545;
    background: #f8d7da;
    padding: 1rem;
    border-radius: 0.25rem;
    border: 1px solid #f5c6cb;
}
"""