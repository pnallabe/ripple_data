# 🧪 Advanced Simulation Module

The **Enhanced Simulation Module** provides comprehensive stress testing and scenario analysis capabilities for financial risk assessment and ripple effect modeling.

## Features

### 🎛️ Simulation Types

1. **Matrix Propagation** - Traditional network-based shock propagation
2. **Monte Carlo Analysis** - Uncertainty quantification with confidence intervals  
3. **Stress Testing** - Multi-level stress scenarios with sensitivity analysis
4. **Scenario Analysis** - Cross-scenario impact comparison
5. **Systemic Risk Assessment** - Comprehensive systemic importance ranking

### 📊 Advanced Analytics

- **Impact Attribution** - Portfolio-level impact analysis with market cap weighting
- **Risk Metrics** - VaR, CVaR, concentration indices, diversification ratios
- **Network Analysis** - Centrality metrics, community detection, contagion pathways
- **Convergence Analysis** - Simulation stability and reliability metrics

### 🎨 Interactive Visualizations

- **Impact Waterfall** - Cascading effect visualization
- **Distribution Analysis** - Statistical impact distribution with confidence bands
- **Network Heatmaps** - Risk intensity mapping
- **Sector Analysis** - Cross-sector impact breakdown
- **Monte Carlo Confidence** - Uncertainty bands and risk corridors
- **Scenario Comparison** - Multi-scenario risk ranking

## Usage

### 1. Dashboard Interface

Navigate to the **🧪 Simulation** tab in the dashboard for:
- Interactive parameter configuration
- Real-time simulation execution
- Multiple visualization options
- Scenario library management

### 2. Programmatic API

```python
from src.simulation import SimulationEngine, ScenarioManager
from src.simulation.engine import SimulationConfig, SimulationType

# Initialize components
engine = SimulationEngine()
scenario_manager = ScenarioManager()

# Create simulation configuration
config = SimulationConfig(
    simulation_type=SimulationType.MONTE_CARLO,
    seed_ticker="JPM",
    shock_magnitude=-0.15,
    damping_factor=0.80,
    monte_carlo_runs=2000
)

# Run simulation
results = engine.run_simulation(config)

# Analyze results
print(f"Impact on {len(results.results_df)} stocks")
print(f"Total absolute impact: {results.statistics['total_absolute_impact']:.2%}")
```

### 3. Scenario Management

```python
from src.simulation.scenarios import Scenario, ScenarioManager

# Create custom scenario
scenario = Scenario(
    name="Custom Bank Crisis",
    description="Severe banking sector stress test",
    seed_ticker="JPM",
    shock_magnitude=-0.20,
    simulation_type=SimulationType.STRESS_TEST,
    damping_factor=0.75
)

# Save and manage scenarios
manager = ScenarioManager()
scenario_id = manager.create_scenario(scenario)

# Load predefined scenarios
predefined = manager.create_predefined_scenarios()
```

## Predefined Scenarios

### 🏦 Financial Crisis Scenarios

1. **Major Bank Crisis** - Severe banking sector stress (JPM, -15%, Stress Test)
2. **Insurance Sector Crisis** - Insurance company failure (AIG, -18%, Stress Test)  
3. **Payment System Disruption** - Critical payment infrastructure failure (V, -14%, Matrix)

### 🔬 Technology Scenarios

4. **Technology Sector Shock** - Major tech company crisis (MSFT, -12%, Matrix)

### 📈 Market Scenarios

5. **Moderate Market Correction** - Typical market correction (SPY, -8%, Monte Carlo)
6. **Systemic Risk Assessment** - Comprehensive risk analysis (JPM, -5%, Systemic)

## Configuration Options

### Simulation Parameters

- **Shock Magnitude**: -50% to -1% (impact severity)
- **Damping Factor**: 0.1 to 0.99 (propagation decay)
- **Max Iterations**: 1 to 1000 (convergence limit)
- **Monte Carlo Runs**: 100 to 10,000 (uncertainty samples)

### Advanced Settings

- **Include/Exclude Tickers**: Custom stock universe
- **Min Correlation**: Network edge filtering  
- **Convergence Threshold**: Precision control
- **Time Horizon**: Forward-looking analysis period

## Results Analysis

### Key Metrics

- **VaR/CVaR**: Value-at-Risk and Conditional Value-at-Risk
- **Systemic Risk Score**: Network-based importance ranking
- **Concentration Index**: Risk concentration measurement
- **Diversification Ratio**: Portfolio diversification effectiveness

### Output Formats

- **Interactive Dashboards**: Real-time visualization
- **JSON Export**: Programmatic access
- **CSV Export**: Spreadsheet analysis
- **Excel Reports**: Multi-sheet comprehensive reports

## Performance

- **Matrix Propagation**: ~0.02s for 45 stocks
- **Monte Carlo (1000 runs)**: ~2-5s for 45 stocks  
- **Stress Testing**: ~0.1s for multiple scenarios
- **Systemic Risk Analysis**: ~10-30s for full assessment

## Integration

### Database Requirements

- **PostgreSQL**: Price data and company information
- **Neo4j**: Correlation network and relationships

### Dependencies

- **Core**: numpy, pandas, networkx, scipy
- **Visualization**: plotly, dash
- **Database**: psycopg2, neo4j-driver

## Testing

Run comprehensive tests:

```bash
python test_simulation.py
```

Expected output:
- ✅ All components initialized successfully
- ✅ Configuration created and validated
- ✅ Scenario management functional
- ✅ Predefined scenarios loaded
- ✅ Visualization components ready
- ✅ Dashboard integration confirmed

## Dashboard Access

1. Start the enhanced dashboard:
   ```bash
   python main.py --mode dashboard --port 8055
   ```

2. Navigate to: [http://localhost:8055](http://localhost:8055)

3. Click the **🧪 Simulation** tab

4. Configure parameters and run simulations

## Example Workflow

1. **Select Simulation Type**: Choose Monte Carlo for uncertainty analysis
2. **Configure Parameters**: Set JPM as seed, -10% shock, 0.80 damping
3. **Set Monte Carlo Runs**: 2000 for high precision
4. **Run Simulation**: Execute and monitor progress
5. **Analyze Results**: Review impact waterfall and confidence intervals
6. **Export Data**: Save results for further analysis
7. **Save Scenario**: Store configuration for future use

## Advanced Features

### Batch Processing

```python
# Run multiple scenarios
configs = [config1, config2, config3]
results_list = engine.run_batch_simulations(configs, max_workers=4)
```

### Custom Visualization

```python
from src.simulation.visualization import SimulationVisualizer

visualizer = SimulationVisualizer()
dashboard = visualizer.create_risk_dashboard(results)
```

### Portfolio Analysis

```python
from src.simulation.analysis import ResultsAnalyzer

analyzer = ResultsAnalyzer()
portfolio_impact = analyzer.calculate_portfolio_impact(
    results, 
    portfolio_weights={'JPM': 0.3, 'BAC': 0.2, 'C': 0.15, ...}
)
```

## Troubleshooting

### Common Issues

1. **"Simulation modules not available"**
   - Ensure all dependencies are installed
   - Check Python path and virtual environment

2. **"No results returned"**
   - Verify database connectivity
   - Check if seed ticker exists in network

3. **"Convergence not achieved"**
   - Increase max_iterations
   - Adjust convergence_threshold
   - Check network structure

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] **Real-time Streaming**: Live market data integration
- [ ] **Machine Learning**: AI-enhanced risk prediction
- [ ] **Regulatory Compliance**: Basel III/CCAR scenario templates
- [ ] **Multi-Asset Classes**: Beyond equities (bonds, derivatives)
- [ ] **International Markets**: Global systemic risk analysis

---

The Enhanced Simulation Module transforms the Stock Ripple Platform into a comprehensive financial risk laboratory, enabling sophisticated stress testing and scenario analysis with enterprise-grade visualizations and analytics.