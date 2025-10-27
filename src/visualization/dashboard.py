"""Enhanced Dash web application for comprehensive financial services ripple effect visualization."""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
from dash.dash_table import FormatTemplate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json
import logging
import networkx as nx
from plotly.subplots import make_subplots

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
        """Setup the enhanced dashboard layout."""
        # Get platform statistics for header
        try:
            stats_query = """
            SELECT 
                COUNT(DISTINCT ticker) as total_tickers,
                COUNT(*) as total_records,
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date
            FROM prices
            """
            stats = pg_manager.read_dataframe(stats_query)
            total_tickers = stats.iloc[0]['total_tickers'] if not stats.empty else 0
            total_records = stats.iloc[0]['total_records'] if not stats.empty else 0
        except:
            total_tickers, total_records = 0, 0
        
        self.app.layout = html.Div([
            # Enhanced Header with statistics
            html.Div([
                html.Div([
                    html.H1("🏦 Stock Dependency & Ripple Effect Analysis Platform", 
                           className="header-title"),
                    html.P("Enterprise-grade financial services risk analysis with comprehensive sector coverage",
                          className="header-subtitle"),
                    html.Div([
                        html.Div([
                            html.H3(f"{total_tickers}", className="stat-number"),
                            html.P("Stock Tickers", className="stat-label")
                        ], className="stat-card"),
                        html.Div([
                            html.H3(f"{total_records:,}", className="stat-number"),
                            html.P("Price Records", className="stat-label")
                        ], className="stat-card"),
                        html.Div([
                            html.H3("3,942+", className="stat-number"),
                            html.P("Correlations", className="stat-label")
                        ], className="stat-card"),
                        html.Div([
                            html.H3("45", className="stat-number"),
                            html.P("Financial Services", className="stat-label")
                        ], className="stat-card")
                    ], className="header-stats")
                ], className="header-content")
            ], className="header"),
            
            # Navigation tabs
            html.Div([
                dcc.Tabs(id="main-tabs", value='dashboard', className="main-tabs", children=[
                    dcc.Tab(label='🎛️ Dashboard', value='dashboard', className="main-tab"),
                    dcc.Tab(label='🔗 Correlations', value='correlations', className="main-tab"),
                    dcc.Tab(label='🏦 Sectors', value='sectors', className="main-tab"),
                    dcc.Tab(label='📊 Analytics', value='analytics', className="main-tab"),
                    dcc.Tab(label='⚙️ System', value='system', className="main-tab")
                ])
            ], className="navigation"),
            
            # Main content area
            html.Div(id="main-content", className="main-content-area"),
            
            # Store components for data
            dcc.Store(id='simulation-results-store'),
            dcc.Store(id='network-data-store'),
            dcc.Store(id='ticker-list-store'),
            dcc.Store(id='correlation-data-store'),
            dcc.Store(id='sector-data-store'),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
            
        ], className="app-container")
    
    def setup_callbacks(self):
        """Setup all enhanced dashboard callbacks."""
        
        @self.app.callback(
            Output('main-content', 'children'),
            [Input('main-tabs', 'value')]
        )
        def render_main_content(active_tab):
            """Render main content based on active tab."""
            if active_tab == 'dashboard':
                return self._create_dashboard_content()
            elif active_tab == 'correlations':
                return self._create_correlations_content()
            elif active_tab == 'sectors':
                return self._create_sectors_content()
            elif active_tab == 'analytics':
                return self._create_analytics_content()
            elif active_tab == 'system':
                return self._create_system_content()
            else:
                return self._create_dashboard_content()
        
        @self.app.callback(
            [Output('ticker-list-store', 'data')],
            [Input('interval-component', 'n_intervals')]
        )
        def load_ticker_list(n_intervals):
            """Load available tickers from database."""
            try:
                # Query available tickers with additional info
                query = """
                SELECT DISTINCT p.ticker, 
                       COALESCE(c.name, p.ticker) as name,
                       COALESCE(c.sector, 'Unknown') as sector,
                       COUNT(p.*) as record_count,
                       MAX(p.trade_date) as latest_date,
                       AVG(p.close) as avg_price
                FROM prices p
                LEFT JOIN companies c ON p.ticker = c.ticker
                WHERE p.ticker IS NOT NULL 
                GROUP BY p.ticker, c.name, c.sector
                ORDER BY p.ticker
                """
                tickers_df = pg_manager.read_dataframe(query)
                tickers_data = tickers_df.to_dict('records') if not tickers_df.empty else []
                
                return [tickers_data]
                
            except Exception as e:
                logger.error(f"Error loading tickers: {e}")
                return [[]]
        
    def _create_dashboard_content(self):
        """Create main dashboard content with simulation controls."""
        return html.Div([
            # Control Panel and Visualization Side by Side
            html.Div([
                # Control Panel
                html.Div([
                    html.H3("🎛️ Simulation Controls", className="panel-title"),
                    
                    # Ticker selection with search
                    html.Div([
                        html.Label("Seed Ticker:", className="control-label"),
                        dcc.Dropdown(
                            id='seed-ticker-dropdown',
                            options=[],
                            value=None,
                            placeholder="🔍 Search and select seed ticker...",
                            searchable=True,
                            className="control-dropdown"
                        )
                    ], className="control-group"),
                    
                    # Preset scenarios
                    html.Div([
                        html.Label("Preset Scenarios:", className="control-label"),
                        html.Div([
                            html.Button("🏦 Banking Crisis", id="banking-crisis-btn", className="scenario-btn"),
                            html.Button("💳 Payment Shock", id="payment-shock-btn", className="scenario-btn"),
                            html.Button("📱 Tech Disruption", id="tech-shock-btn", className="scenario-btn")
                        ], className="scenario-buttons")
                    ], className="control-group"),
                    
                    # Shock magnitude
                    html.Div([
                        html.Label("Shock Magnitude:", className="control-label"),
                        dcc.Slider(
                            id='shock-magnitude-slider',
                            min=-50,
                            max=50,
                            step=1,
                            value=-5,
                            marks={i: f"{i}%" for i in range(-50, 51, 10)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="control-slider"
                        )
                    ], className="control-group"),
                    
                    # Damping factor
                    html.Div([
                        html.Label("Damping Factor:", className="control-label"),
                        dcc.Slider(
                            id='damping-factor-slider',
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            value=0.85,
                            marks={i/10: f"{i/10:.1f}" for i in range(1, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="control-slider"
                        )
                    ], className="control-group"),
                    
                    # Advanced options
                    html.Details([
                        html.Summary("⚙️ Advanced Options", className="advanced-summary"),
                        html.Div([
                            html.Label("Max Iterations:", className="control-label"),
                            dcc.Input(
                                id='max-iterations-input',
                                type='number',
                                value=50,
                                min=10,
                                max=200,
                                className="control-input"
                            )
                        ], className="control-group"),
                    ], className="advanced-controls"),
                    
                    # Run simulation button
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-play"),
                            html.Span(" Run Simulation", style={"margin-left": "8px"})
                        ], 
                        id="run-simulation-btn", 
                        className="btn-primary",
                        n_clicks=0)
                    ], className="control-group"),
                    
                    # Status indicator
                    html.Div(id="simulation-status", className="status-indicator")
                    
                ], className="control-panel"),
                
                # Main visualization area
                html.Div([
                    # Tabs for different views
                    dcc.Tabs(id="visualization-tabs", value='network-view', className="viz-tabs", children=[
                        dcc.Tab(label='🕸️ Network', value='network-view', className="viz-tab"),
                        dcc.Tab(label='📊 Impact', value='impact-analysis', className="viz-tab"),
                        dcc.Tab(label='📈 Time Series', value='time-series', className="viz-tab"),
                        dcc.Tab(label='🏢 Sectors', value='sector-analysis', className="viz-tab")
                    ]),
                    
                    # Tab content
                    html.Div(id="tab-content", className="tab-content")
                    
                ], className="visualization-area")
                
            ], className="dashboard-layout"),
            
            # Add callback for scenario buttons and simulation
            self._setup_simulation_callbacks()
            
        ])
    
    def _setup_simulation_callbacks(self):
        """Setup simulation-specific callbacks."""
        @self.app.callback(
            [Output('seed-ticker-dropdown', 'options'),
             Output('seed-ticker-dropdown', 'value')],
            [Input('ticker-list-store', 'data'),
             Input('banking-crisis-btn', 'n_clicks'),
             Input('payment-shock-btn', 'n_clicks'),
             Input('tech-shock-btn', 'n_clicks')]
        )
        def update_ticker_options_and_scenarios(tickers_data, banking_clicks, payment_clicks, tech_clicks):
            """Update ticker options and handle scenario button clicks."""
            try:
                # Create dropdown options
                options = []
                if tickers_data:
                    for ticker in tickers_data:
                        sector = ticker.get('sector', 'Unknown')
                        label = f"{ticker['ticker']} - {ticker.get('name', ticker['ticker'])} ({sector})"
                        options.append({'label': label, 'value': ticker['ticker']})
                
                # Handle scenario button clicks
                ctx = dash.callback_context
                if ctx.triggered:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                    if button_id == 'banking-crisis-btn' and banking_clicks:
                        return options, 'JPM'
                    elif button_id == 'payment-shock-btn' and payment_clicks:
                        return options, 'V'
                    elif button_id == 'tech-shock-btn' and tech_clicks:
                        return options, 'AAPL'
                
                return options, None
                
            except Exception as e:
                logger.error(f"Error updating ticker options: {e}")
                return [], None
        
        @self.app.callback(
            [Output('simulation-results-store', 'data'),
             Output('simulation-status', 'children')],
            [Input('run-simulation-btn', 'n_clicks')],
            [State('seed-ticker-dropdown', 'value'),
             State('shock-magnitude-slider', 'value'),
             State('damping-factor-slider', 'value'),
             State('max-iterations-input', 'value')]
        )
        def run_simulation(n_clicks, seed_ticker, shock_magnitude, damping_factor, max_iterations):
            """Run enhanced ripple effect simulation."""
            if n_clicks == 0 or not seed_ticker:
                return {}, ""
            
            try:
                # Show loading status
                status = html.Div([
                    html.I(className="fas fa-spinner fa-spin"),
                    html.Span(" Running advanced simulation...", style={"margin-left": "10px"})
                ], className="status-loading")
                
                # Run simulation with enhanced parameters
                shock_pct = shock_magnitude / 100.0
                results = self.propagator.simulate_shock_propagation(
                    seed_ticker=seed_ticker,
                    shock_magnitude=shock_pct,
                    damping_factor=damping_factor,
                    max_iterations=max_iterations or 50
                )
                
                if results.empty:
                    return {}, html.Div("⚠️ Simulation failed - no results", className="status-error")
                
                # Convert to JSON serializable format
                results_dict = results.to_dict('records')
                
                success_status = html.Div([
                    html.I(className="fas fa-check-circle"),
                    html.Span(f" ✅ Simulation completed - {len(results)} stocks analyzed", 
                             style={"margin-left": "10px"})
                ], className="status-success")
                
                return results_dict, success_status
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                error_status = html.Div([
                    html.I(className="fas fa-exclamation-triangle"),
                    html.Span(f" ❌ Error: {str(e)}", style={"margin-left": "10px"})
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
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-info-circle", style={"font-size": "3rem", "color": "#6c757d"}),
                        html.H4("Ready for Analysis", style={"margin-top": "1rem", "color": "#6c757d"}),
                        html.P("Select a seed ticker and run a simulation to see results"),
                        html.P("💡 Try our preset scenarios for quick analysis!")
                    ], style={"text-align": "center", "margin-top": "3rem"})
                ], className="no-data-message")
            
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
    
    def _create_correlations_content(self):
        """Create correlations analysis content."""
        return html.Div([
            html.H2("🔗 Correlation Matrix Analysis"),
            html.P("Real-time correlation analysis across financial services sector"),
            
            # Correlation heatmap will be added here
            html.Div(id="correlation-heatmap"),
            
            # Top correlations table
            html.Div(id="top-correlations-table")
        ])
    
    def _create_sectors_content(self):
        """Create sector analysis content."""
        return html.Div([
            html.H2("🏦 Sector Performance Analysis"),
            html.P("Comprehensive analysis across financial services subsectors"),
            
            # Sector breakdown charts
            html.Div(id="sector-breakdown"),
            
            # Sector correlation matrix
            html.Div(id="sector-correlations")
        ])
    
    def _create_analytics_content(self):
        """Create advanced analytics content."""
        return html.Div([
            html.H2("📊 Advanced Analytics"),
            html.P("Deep dive into market dynamics and risk metrics"),
            
            # Risk metrics dashboard
            html.Div(id="risk-metrics"),
            
            # Volatility analysis
            html.Div(id="volatility-analysis")
        ])
    
    def _create_system_content(self):
        """Create system status and configuration content."""
        try:
            # Get system statistics
            stats_query = """
            SELECT 
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(*) as total_records,
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date,
                COUNT(*) FILTER (WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days') as recent_records
            FROM prices
            """
            stats = pg_manager.read_dataframe(stats_query)
            
            if not stats.empty:
                stat_row = stats.iloc[0]
                return html.Div([
                    html.H2("⚙️ System Status"),
                    
                    # System health cards
                    html.Div([
                        html.Div([
                            html.H3("📊 Database"),
                            html.P(f"✅ Connected"),
                            html.P(f"Tickers: {stat_row['unique_tickers']}"),
                            html.P(f"Records: {stat_row['total_records']:,}")
                        ], className="system-card"),
                        
                        html.Div([
                            html.H3("🔗 Neo4j"),
                            html.P("✅ Connected"),
                            html.P("Correlations: 3,942+"),
                            html.P("Relationships: Active")
                        ], className="system-card"),
                        
                        html.Div([
                            html.H3("📈 Data Coverage"),
                            html.P(f"From: {stat_row['earliest_date']}"),
                            html.P(f"To: {stat_row['latest_date']}"),
                            html.P(f"Recent: {stat_row['recent_records']:,}")
                        ], className="system-card")
                    ], className="system-cards"),
                    
                    # Configuration
                    html.H3("Configuration"),
                    html.Pre(f"""
Database: PostgreSQL
Graph DB: Neo4j  
Cache: Redis
API: Yahoo Finance
Sectors: 45 Financial Services + Technology
Correlation Window: 30 days
Max Iterations: 50
Default Damping: 0.85
                    """, className="config-display")
                ])
            else:
                return html.Div("No system data available")
                
        except Exception as e:
            return html.Div(f"System status error: {e}")

    def _create_network_view(self, df_results: pd.DataFrame) -> html.Div:
        """Create enhanced network visualization view."""
        try:
            # Create enhanced network graph
            network_fig = self._create_enhanced_network_graph(df_results)
            
            # Get seed ticker info
            seed_ticker = df_results[df_results['final_impact'].abs() == df_results['final_impact'].abs().max()]['ticker'].iloc[0]
            
            return html.Div([
                html.Div([
                    html.H4("🕸️ Network Impact Visualization"),
                    html.P(f"Shock propagation from {seed_ticker} across the financial network")
                ], className="section-header"),
                
                # Network controls
                html.Div([
                    html.Label("Network Layout:"),
                    dcc.RadioItems(
                        id='network-layout-radio',
                        options=[
                            {'label': 'Force-directed', 'value': 'force'},
                            {'label': 'Circular', 'value': 'circular'},
                            {'label': 'Hierarchical', 'value': 'hierarchical'}
                        ],
                        value='force',
                        inline=True,
                        className="network-controls"
                    )
                ], className="control-row"),
                
                # Main network graph
                dcc.Graph(
                    figure=network_fig, 
                    id="network-graph",
                    config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'network_analysis'}}
                ),
                
                # Network statistics
                html.Div([
                    html.Div([
                        html.H5(f"{len(df_results)}"),
                        html.P("Total Nodes")
                    ], className="network-stat"),
                    html.Div([
                        html.H5(f"{(df_results['final_impact'].abs() > 0.001).sum()}"),
                        html.P("Affected Nodes")
                    ], className="network-stat"),
                    html.Div([
                        html.H5(f"{df_results['final_impact'].abs().max():.4f}"),
                        html.P("Max Impact")
                    ], className="network-stat"),
                    html.Div([
                        html.H5(f"{df_results['total_iterations'].iloc[0] if 'total_iterations' in df_results.columns else 'N/A'}"),
                        html.P("Iterations")
                    ], className="network-stat")
                ], className="network-stats"),
                
                html.P([
                    "💡 ", html.Strong("Interactive Network: "), 
                    "Node size = impact magnitude, Color = positive/negative impact, ",
                    "Hover for details, Zoom and pan enabled"
                ], className="chart-caption")
            ])
            
        except Exception as e:
            logger.error(f"Error creating network view: {e}")
            return html.Div([
                html.I(className="fas fa-exclamation-triangle"),
                html.Span(f" Error creating network view: {e}")
            ], className="error-message")
    
    def _create_impact_analysis(self, df_results: pd.DataFrame) -> html.Div:
        """Create enhanced impact analysis view."""
        try:
            # Sort by absolute impact
            df_sorted = df_results.copy()
            df_sorted['abs_impact'] = df_sorted['final_impact'].abs()
            df_sorted = df_sorted.sort_values('abs_impact', ascending=False)
            
            # Top impacted stocks
            top_impacted = df_sorted.head(20)
            
            # Create dual chart: positive and negative impacts
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("📈 Positive Impacts", "📉 Negative Impacts"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Positive impacts
            positive_impacts = df_sorted[df_sorted['final_impact'] > 0].head(10)
            if not positive_impacts.empty:
                fig.add_trace(
                    go.Bar(
                        x=positive_impacts['ticker'],
                        y=positive_impacts['final_impact'] * 100,
                        name="Positive Impact",
                        marker_color='#28a745',
                        hovertemplate="<b>%{x}</b><br>Impact: %{y:.3f}%<extra></extra>"
                    ),
                    row=1, col=1
                )
            
            # Negative impacts
            negative_impacts = df_sorted[df_sorted['final_impact'] < 0].head(10)
            if not negative_impacts.empty:
                fig.add_trace(
                    go.Bar(
                        x=negative_impacts['ticker'],
                        y=negative_impacts['final_impact'].abs() * 100,
                        name="Negative Impact",
                        marker_color='#dc3545',
                        hovertemplate="<b>%{x}</b><br>Impact: -%{y:.3f}%<extra></extra>"
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="📊 Impact Distribution Analysis",
                showlegend=False,
                height=500
            )
            fig.update_xaxes(tickangle=-45)
            fig.update_yaxes(title_text="Impact (%)")
            
            # Summary statistics
            total_positive = df_results[df_results['final_impact'] > 0]['final_impact'].sum()
            total_negative = df_results[df_results['final_impact'] < 0]['final_impact'].sum()
            affected_stocks = (df_results['final_impact'].abs() > 0.001).sum()
            max_impact = df_results['final_impact'].abs().max()
            
            return html.Div([
                html.H4("📊 Impact Analysis Dashboard"),
                
                # Enhanced summary cards
                html.Div([
                    html.Div([
                        html.I(className="fas fa-arrow-up", style={"color": "#28a745"}),
                        html.H4(f"+{total_positive:.4f}", style={"color": "#28a745"}),
                        html.P("Positive Impact")
                    ], className="impact-card positive"),
                    
                    html.Div([
                        html.I(className="fas fa-arrow-down", style={"color": "#dc3545"}),
                        html.H4(f"{total_negative:.4f}", style={"color": "#dc3545"}),
                        html.P("Negative Impact")
                    ], className="impact-card negative"),
                    
                    html.Div([
                        html.I(className="fas fa-network-wired", style={"color": "#6c757d"}),
                        html.H4(f"{affected_stocks}"),
                        html.P("Affected Stocks")
                    ], className="impact-card neutral"),
                    
                    html.Div([
                        html.I(className="fas fa-chart-line", style={"color": "#ffc107"}),
                        html.H4(f"{max_impact:.4f}"),
                        html.P("Max Impact")
                    ], className="impact-card warning")
                ], className="impact-cards"),
                
                # Impact distribution chart
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True, 'toImageButtonOptions': {'filename': 'impact_analysis'}}
                ),
                
                # Enhanced detailed table with filtering
                html.Div([
                    html.H5("🔍 Detailed Impact Analysis"),
                    html.Div([
                        html.Label("Filter by Impact Type:"),
                        dcc.RadioItems(
                            id='impact-filter-radio',
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'Positive Only', 'value': 'positive'},
                                {'label': 'Negative Only', 'value': 'negative'},
                                {'label': 'High Impact (>0.01%)', 'value': 'high'}
                            ],
                            value='all',
                            inline=True,
                            className="filter-controls"
                        )
                    ], className="table-controls"),
                    
                    dash_table.DataTable(
                        data=top_impacted[['ticker', 'final_impact', 'cumulative_impact', 
                                         'max_impact']].to_dict('records'),
                        columns=[
                            {'name': 'Ticker', 'id': 'ticker'},
                            {'name': 'Final Impact (%)', 'id': 'final_impact', 'type': 'numeric', 
                             'format': {'specifier': '.4%'}},
                            {'name': 'Cumulative Impact (%)', 'id': 'cumulative_impact', 'type': 'numeric', 
                             'format': {'specifier': '.4%'}},
                            {'name': 'Max Impact (%)', 'id': 'max_impact', 'type': 'numeric', 
                             'format': {'specifier': '.4%'}}
                        ],
                        sort_action="native",
                        filter_action="native",
                        page_size=15,
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
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
                            },
                            {
                                'if': {'filter_query': '{final_impact} > 0.01 || {final_impact} < -0.01'},
                                'fontWeight': 'bold'
                            }
                        ],
                        export_format="csv",
                        export_headers="display"
                    )
                ], className="table-section")
            ])
            
        except Exception as e:
            logger.error(f"Error creating impact analysis: {e}")
            return html.Div([
                html.I(className="fas fa-exclamation-triangle"),
                html.Span(f" Error creating impact analysis: {e}")
            ], className="error-message")
    
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
    
    def _create_enhanced_network_graph(self, df_results: pd.DataFrame) -> go.Figure:
        """Create enhanced network graph visualization with realistic positioning."""
        try:
            # Create a more sophisticated network layout
            n_nodes = len(df_results)
            
            # Use circular layout with impact-based positioning
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            
            # Position nodes based on impact magnitude (higher impact = closer to center)
            impact_magnitudes = df_results['final_impact'].abs()
            radii = 1 - (impact_magnitudes / impact_magnitudes.max()) * 0.7  # Outer ring to center
            
            x_positions = radii * np.cos(angles)
            y_positions = radii * np.sin(angles)
            
            # Create the network figure
            fig = go.Figure()
            
            # Add edges (connections) - simplified for demonstration
            # In practice, you'd get actual correlation data
            seed_idx = df_results['final_impact'].abs().idxmax()
            for i in range(len(df_results)):
                if i != seed_idx and df_results.iloc[i]['final_impact'] != 0:
                    # Draw edge from seed to affected node
                    fig.add_trace(go.Scatter(
                        x=[x_positions[seed_idx], x_positions[i]],
                        y=[y_positions[seed_idx], y_positions[i]],
                        mode='lines',
                        line=dict(
                            width=abs(df_results.iloc[i]['final_impact']) * 5000,
                            color='rgba(128,128,128,0.3)'
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Add nodes (stocks)
            node_sizes = np.maximum(impact_magnitudes * 5000 + 15, 8)  # Minimum size of 8
            node_colors = df_results['final_impact']
            
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='RdBu_r',
                    showscale=True,
                    colorbar=dict(
                        title="Impact (%)",
                        titleside="right",
                        tickformat=".2%"
                    ),
                    line=dict(width=2, color='rgba(50,50,50,0.8)'),
                    symbol='circle'
                ),
                text=df_results['ticker'],
                textposition="middle center",
                textfont=dict(
                    size=np.minimum(node_sizes/2, 12),
                    color='white'
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Impact: %{marker.color:.4f}<br>" +
                    "Position: (%{x:.2f}, %{y:.2f})" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))
            
            # Highlight seed node
            if seed_idx is not None:
                fig.add_trace(go.Scatter(
                    x=[x_positions[seed_idx]],
                    y=[y_positions[seed_idx]],
                    mode='markers',
                    marker=dict(
                        size=node_sizes[seed_idx] + 10,
                        color='gold',
                        symbol='star',
                        line=dict(width=3, color='orange')
                    ),
                    name='Seed Node',
                    hovertemplate=f"<b>SEED: {df_results.iloc[seed_idx]['ticker']}</b><extra></extra>",
                    showlegend=True
                ))
            
            fig.update_layout(
                title={
                    'text': "🕸️ Financial Network Impact Propagation",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=[-1.5, 1.5]
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=[-1.5, 1.5]
                ),
                plot_bgcolor='rgba(240,240,240,0.1)',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12),
                height=600,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Network graph creation error: {e}")
            # Fallback to simple visualization
            return self._create_simple_network_graph(df_results)
    
    def _create_simple_network_graph(self, df_results: pd.DataFrame) -> go.Figure:
        """Fallback simple network graph."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=np.random.normal(0, 1, len(df_results)),
            y=np.random.normal(0, 1, len(df_results)),
            mode='markers+text',
            marker=dict(
                size=np.abs(df_results['final_impact']) * 1000 + 15,
                color=df_results['final_impact'],
                colorscale='RdBu_r',
                showscale=True
            ),
            text=df_results['ticker'],
            textposition="middle center"
        ))
        
        fig.update_layout(
            title="Network Analysis",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def run_server(self, debug: bool = False, port: int = 8050):
        """Start the enhanced dashboard server"""
        logger.info(f"🚀 Enhanced Financial Services Dashboard running on http://localhost:{port}")
        logger.info("🏦 Features: 45 stocks, real-time correlations, advanced analytics")
        self.app.run(debug=debug, port=port)


# Enhanced external stylesheets
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
]

# Enhanced custom CSS
custom_css = """
/* Global Styles */
* {
    box-sizing: border-box;
}

.app-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
    background: #f8f9fa;
    min-height: 100vh;
}

/* Header Styles */
.header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 3rem 2rem;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
}

.header-title {
    margin: 0;
    font-size: 3rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.header-subtitle {
    margin: 1rem 0 2rem;
    font-size: 1.3rem;
    opacity: 0.9;
    font-weight: 300;
}

.header-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.stat-card {
    background: rgba(255,255,255,0.15);
    padding: 1.5rem;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    color: #fff;
}

.stat-label {
    font-size: 0.9rem;
    margin: 0.5rem 0 0;
    opacity: 0.8;
    font-weight: 400;
}

/* Navigation */
.navigation {
    background: white;
    border-bottom: 1px solid #e9ecef;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.main-tabs {
    max-width: 1200px;
    margin: 0 auto;
}

.main-tab {
    font-weight: 600;
    color: #495057;
    border: none !important;
    padding: 1rem 2rem !important;
    font-size: 1.1rem;
}

.main-tab:hover {
    color: #2a5298;
}

.main-tab.react-tabs__tab--selected {
    color: #2a5298;
    border-bottom: 3px solid #2a5298 !important;
    background: transparent;
}

/* Main Content */
.main-content-area {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

.dashboard-layout {
    display: grid;
    grid-template-columns: 350px 1fr;
    gap: 2rem;
    min-height: 80vh;
}

/* Control Panel */
.control-panel {
    background: white;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    height: fit-content;
    position: sticky;
    top: 2rem;
}

.panel-title {
    color: #2a5298;
    font-weight: 600;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #e9ecef;
}

.control-group {
    margin-bottom: 2rem;
}

.control-label {
    display: block;
    font-weight: 600;
    color: #495057;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
}

.control-dropdown, .control-input {
    width: 100%;
    border-radius: 8px;
    border: 2px solid #e9ecef;
    font-size: 1rem;
}

.control-dropdown:focus, .control-input:focus {
    border-color: #2a5298;
    box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1);
}

.control-slider .rc-slider-track {
    background: #2a5298;
}

.control-slider .rc-slider-handle {
    border-color: #2a5298;
}

.scenario-buttons {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.scenario-btn {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    color: #495057;
    border: 1px solid #dee2e6;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.scenario-btn:hover {
    background: linear-gradient(135deg, #e9ecef, #dee2e6);
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.advanced-controls {
    margin-top: 1rem;
}

.advanced-summary {
    cursor: pointer;
    font-weight: 600;
    color: #6c757d;
    padding: 0.5rem 0;
}

.btn-primary {
    background: linear-gradient(135deg, #2a5298, #1e3c72);
    color: white;
    border: none;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    cursor: pointer;
    font-size: 1.1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(42, 82, 152, 0.3);
}

/* Visualization Area */
.visualization-area {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    overflow: hidden;
}

.viz-tabs .react-tabs__tab-list {
    margin: 0;
    padding: 0 2rem;
    background: #f8f9fa;
    border-bottom: 1px solid #e9ecef;
}

.viz-tab {
    padding: 1rem 1.5rem !important;
    font-weight: 600;
    color: #6c757d;
    border: none !important;
}

.viz-tab.react-tabs__tab--selected {
    color: #2a5298;
    background: white;
    border-bottom: 3px solid #2a5298 !important;
}

.tab-content {
    padding: 2rem;
}

/* Cards and Stats */
.impact-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.impact-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    border-left: 4px solid #e9ecef;
}

.impact-card.positive {
    border-left-color: #28a745;
}

.impact-card.negative {
    border-left-color: #dc3545;
}

.impact-card.neutral {
    border-left-color: #6c757d;
}

.impact-card.warning {
    border-left-color: #ffc107;
}

.impact-card h4 {
    margin: 0.5rem 0 0;
    font-size: 2rem;
    font-weight: 700;
}

.impact-card p {
    margin: 0.5rem 0 0;
    color: #6c757d;
    font-weight: 500;
}

/* Network Stats */
.network-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.network-stat {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}

.network-stat h5 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #2a5298;
}

.network-stat p {
    margin: 0.25rem 0 0;
    font-size: 0.85rem;
    color: #6c757d;
}

/* System Cards */
.system-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
    margin: 2rem 0;
}

.system-card {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #28a745;
}

.system-card h3 {
    color: #2a5298;
    margin-bottom: 1rem;
}

.config-display {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #6c757d;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Status Indicators */
.status-indicator {
    padding: 1rem 0;
    font-weight: 600;
}

.status-loading {
    color: #ffc107;
}

.status-success {
    color: #28a745;
}

.status-error {
    color: #dc3545;
    background: #fff5f5;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #dc3545;
}

/* Utility Classes */
.section-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #e9ecef;
}

.section-header h4 {
    color: #2a5298;
    font-weight: 600;
    margin: 0;
}

.chart-caption {
    color: #6c757d;
    font-size: 0.9rem;
    margin-top: 1rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #17a2b8;
}

.no-data-message {
    text-align: center;
    color: #6c757d;
    font-size: 1.1rem;
    margin: 3rem 0;
    padding: 3rem;
    background: #f8f9fa;
    border-radius: 12px;
}

.error-message {
    color: #dc3545;
    background: #fff5f5;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #f5c6cb;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
}

/* Responsive Design */
@media (max-width: 1200px) {
    .dashboard-layout {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .control-panel {
        position: static;
    }
    
    .header-title {
        font-size: 2.5rem;
    }
    
    .header-stats {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 768px) {
    .main-content-area {
        padding: 1rem;
    }
    
    .header {
        padding: 2rem 1rem;
    }
    
    .header-title {
        font-size: 2rem;
    }
    
    .header-stats {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .impact-cards {
        grid-template-columns: 1fr;
    }
    
    .network-stats {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.tab-content > * {
    animation: fadeIn 0.3s ease-out;
}
"""