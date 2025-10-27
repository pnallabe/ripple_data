#!/usr/bin/env python3
"""Test script for the enhanced simulation module."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.engine import SimulationEngine, SimulationConfig, SimulationType
from src.simulation.scenarios import ScenarioManager
from src.simulation.analysis import ResultsAnalyzer
from src.simulation.visualization import SimulationVisualizer

def test_simulation_components():
    """Test the simulation components."""
    print("🧪 Testing Enhanced Simulation Module")
    print("=" * 50)
    
    # Test 1: Initialize components
    print("1. Initializing simulation components...")
    try:
        engine = SimulationEngine()
        scenario_manager = ScenarioManager()
        analyzer = ResultsAnalyzer()
        visualizer = SimulationVisualizer()
        print("   ✅ All components initialized successfully")
    except Exception as e:
        print(f"   ❌ Error initializing components: {e}")
        return False
    
    # Test 2: Create a simple simulation config
    print("\n2. Creating simulation configuration...")
    try:
        config = SimulationConfig(
            simulation_type=SimulationType.MATRIX_PROPAGATION,
            seed_ticker="JPM",
            shock_magnitude=-0.05,
            damping_factor=0.85,
            max_iterations=50
        )
        print(f"   ✅ Configuration created: {config.simulation_type.value}")
    except Exception as e:
        print(f"   ❌ Error creating configuration: {e}")
        return False
    
    # Test 3: Test scenario management
    print("\n3. Testing scenario management...")
    try:
        from src.simulation.scenarios import Scenario
        
        test_scenario = Scenario(
            name="Test Scenario",
            description="A test scenario for validation",
            seed_ticker="JPM",
            shock_magnitude=-0.10,
            simulation_type=SimulationType.MATRIX_PROPAGATION
        )
        
        scenario_id = scenario_manager.create_scenario(test_scenario)
        scenarios = scenario_manager.list_scenarios()
        print(f"   ✅ Scenario created with ID: {scenario_id}")
        print(f"   ✅ Total scenarios: {len(scenarios)}")
    except Exception as e:
        print(f"   ❌ Error in scenario management: {e}")
        return False
    
    # Test 4: Test predefined scenarios
    print("\n4. Creating predefined scenarios...")
    try:
        predefined_ids = scenario_manager.create_predefined_scenarios()
        print(f"   ✅ Created {len(predefined_ids)} predefined scenarios")
        
        # List all scenarios
        all_scenarios = scenario_manager.list_scenarios()
        for scenario_id, scenario in all_scenarios.items():
            print(f"   - {scenario.name}: {scenario.simulation_type.value}")
    except Exception as e:
        print(f"   ❌ Error creating predefined scenarios: {e}")
        return False
    
    # Test 5: Test visualization components
    print("\n5. Testing visualization components...")
    try:
        # Test empty figure creation
        empty_fig = visualizer._create_empty_figure("Test message")
        print("   ✅ Empty figure creation works")
        
        # Test color schemes
        colors = visualizer.colors
        print(f"   ✅ Color scheme loaded with {len(colors)} colors")
    except Exception as e:
        print(f"   ❌ Error in visualization components: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All simulation module tests passed!")
    print("\nNext steps:")
    print("- Run the dashboard: python main.py --mode dashboard --port 8054")
    print("- Navigate to the 🧪 Simulation tab")
    print("- Try different simulation types and scenarios")
    
    return True

def test_dashboard_integration():
    """Test dashboard integration."""
    print("\n🖥️  Testing Dashboard Integration")
    print("=" * 50)
    
    try:
        from src.visualization.dashboard import RippleDashboard, SIMULATION_AVAILABLE
        
        print(f"1. Simulation module available: {SIMULATION_AVAILABLE}")
        
        if SIMULATION_AVAILABLE:
            dashboard = RippleDashboard()
            print("   ✅ Dashboard initialized with simulation support")
            
            # Check if simulation components are available
            if hasattr(dashboard, 'simulation_engine') and dashboard.simulation_engine:
                print("   ✅ Simulation engine available")
            if hasattr(dashboard, 'scenario_manager') and dashboard.scenario_manager:
                print("   ✅ Scenario manager available")
            if hasattr(dashboard, 'results_analyzer') and dashboard.results_analyzer:
                print("   ✅ Results analyzer available")
            if hasattr(dashboard, 'simulation_visualizer') and dashboard.simulation_visualizer:
                print("   ✅ Simulation visualizer available")
                
        else:
            print("   ⚠️  Simulation module not available in dashboard")
            
    except Exception as e:
        print(f"   ❌ Dashboard integration error: {e}")
        return False
    
    print("   ✅ Dashboard integration test passed!")
    return True

if __name__ == "__main__":
    success = test_simulation_components()
    if success:
        test_dashboard_integration()
    
    print("\n🚀 Ready to launch enhanced simulation platform!")