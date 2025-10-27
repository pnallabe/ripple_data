"""Scenario management for simulation analysis."""

import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .engine import SimulationConfig, SimulationType
from src.database import pg_manager, neo4j_manager

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Scenario definition for simulation."""
    name: str
    description: str
    seed_ticker: str
    shock_magnitude: float
    simulation_type: SimulationType = SimulationType.MATRIX_PROPAGATION
    damping_factor: float = 0.85
    max_iterations: int = 100
    include_tickers: Optional[List[str]] = None
    exclude_tickers: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_date: Optional[datetime] = None
    
    def to_simulation_config(self) -> SimulationConfig:
        """Convert scenario to simulation configuration."""
        return SimulationConfig(
            simulation_type=self.simulation_type,
            seed_ticker=self.seed_ticker,
            shock_magnitude=self.shock_magnitude,
            damping_factor=self.damping_factor,
            max_iterations=self.max_iterations,
            include_tickers=self.include_tickers,
            exclude_tickers=self.exclude_tickers,
            metadata=self.metadata
        )


class ScenarioManager:
    """Manager for creating and managing simulation scenarios."""
    
    def __init__(self, scenarios_file: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.ScenarioManager")
        self.scenarios_file = scenarios_file or "config/scenarios.json"
        self.scenarios = {}
        self._load_scenarios()
    
    def create_scenario(self, scenario: Scenario) -> str:
        """Create and save a new scenario."""
        try:
            scenario.created_date = datetime.now()
            scenario_id = self._generate_scenario_id(scenario.name)
            
            self.scenarios[scenario_id] = scenario
            self._save_scenarios()
            
            self.logger.info(f"Created scenario: {scenario_id} - {scenario.name}")
            return scenario_id
            
        except Exception as e:
            self.logger.error(f"Error creating scenario: {e}")
            raise
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID."""
        return self.scenarios.get(scenario_id)
    
    def list_scenarios(self) -> Dict[str, Scenario]:
        """List all scenarios."""
        return self.scenarios.copy()
    
    def update_scenario(self, scenario_id: str, scenario: Scenario) -> bool:
        """Update existing scenario."""
        try:
            if scenario_id not in self.scenarios:
                return False
            
            # Preserve creation date
            if self.scenarios[scenario_id].created_date:
                scenario.created_date = self.scenarios[scenario_id].created_date
            
            self.scenarios[scenario_id] = scenario
            self._save_scenarios()
            
            self.logger.info(f"Updated scenario: {scenario_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating scenario {scenario_id}: {e}")
            return False
    
    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete scenario."""
        try:
            if scenario_id in self.scenarios:
                del self.scenarios[scenario_id]
                self._save_scenarios()
                self.logger.info(f"Deleted scenario: {scenario_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting scenario {scenario_id}: {e}")
            return False
    
    def create_predefined_scenarios(self) -> List[str]:
        """Create a set of predefined scenarios for common analysis."""
        try:
            # Get available tickers
            query = "MATCH (c:Company) RETURN c.ticker AS ticker, c.sector AS sector ORDER BY c.ticker"
            companies = pd.DataFrame(neo4j_manager.execute_query(query))
            
            if companies.empty:
                self.logger.warning("No companies found for scenario creation")
                return []
            
            scenario_ids = []
            
            # 1. Major Bank Crisis Scenario
            bank_tickers = companies[companies['sector'].str.contains('Bank|Finance', case=False, na=False)]['ticker'].tolist()
            if bank_tickers:
                scenario = Scenario(
                    name="Major Bank Crisis",
                    description="Simulates a severe crisis starting from a major bank, representing systemic banking sector stress",
                    seed_ticker=bank_tickers[0],  # Use first major bank
                    shock_magnitude=-0.15,
                    simulation_type=SimulationType.MATRIX_PROPAGATION,
                    damping_factor=0.80,
                    metadata={'category': 'financial_crisis', 'severity': 'high'}
                )
                scenario_ids.append(self.create_scenario(scenario))
            
            # 2. Technology Sector Shock
            tech_tickers = companies[companies['sector'].str.contains('Tech|Software', case=False, na=False)]['ticker'].tolist()
            if tech_tickers:
                scenario = Scenario(
                    name="Technology Sector Shock",
                    description="Major technology company crisis affecting digital infrastructure and innovation sectors",
                    seed_ticker=tech_tickers[0],
                    shock_magnitude=-0.12,
                    simulation_type=SimulationType.STRESS_TEST,
                    damping_factor=0.75,
                    metadata={'category': 'sector_crisis', 'sector': 'technology'}
                )
                scenario_ids.append(self.create_scenario(scenario))
            
            # 3. Moderate Market Correction
            scenario = Scenario(
                name="Moderate Market Correction",
                description="A typical market correction scenario with moderate shock propagation",
                seed_ticker=companies.iloc[0]['ticker'],  # Use first available ticker
                shock_magnitude=-0.08,
                simulation_type=SimulationType.MONTE_CARLO,
                damping_factor=0.85,
                metadata={'category': 'market_correction', 'severity': 'moderate', 'monte_carlo_runs': 500}
            )
            scenario_ids.append(self.create_scenario(scenario))
            
            # 4. Systemic Risk Assessment
            scenario = Scenario(
                name="Systemic Risk Assessment",
                description="Comprehensive analysis of systemic risk across all institutions",
                seed_ticker=companies.iloc[0]['ticker'],
                shock_magnitude=-0.05,
                simulation_type=SimulationType.SYSTEMIC_RISK,
                damping_factor=0.85,
                metadata={'category': 'risk_assessment', 'comprehensive': True}
            )
            scenario_ids.append(self.create_scenario(scenario))
            
            # 5. Cross-Sector Contagion
            scenario = Scenario(
                name="Cross-Sector Contagion",
                description="Analysis of contagion effects across multiple sectors and institutions",
                seed_ticker=companies.iloc[0]['ticker'],
                shock_magnitude=-0.10,
                simulation_type=SimulationType.SCENARIO_ANALYSIS,
                damping_factor=0.82,
                metadata={'category': 'contagion_analysis', 'cross_sector': True}
            )
            scenario_ids.append(self.create_scenario(scenario))
            
            # 6. Insurance Sector Crisis
            insurance_tickers = companies[companies['sector'].str.contains('Insurance', case=False, na=False)]['ticker'].tolist()
            if insurance_tickers:
                scenario = Scenario(
                    name="Insurance Sector Crisis",
                    description="Major insurance company failure affecting risk transfer markets",
                    seed_ticker=insurance_tickers[0],
                    shock_magnitude=-0.18,
                    simulation_type=SimulationType.STRESS_TEST,
                    damping_factor=0.78,
                    metadata={'category': 'sector_crisis', 'sector': 'insurance'}
                )
                scenario_ids.append(self.create_scenario(scenario))
            
            # 7. Payment System Disruption
            payment_tickers = companies[companies['sector'].str.contains('Payment|Card', case=False, na=False)]['ticker'].tolist()
            if payment_tickers:
                scenario = Scenario(
                    name="Payment System Disruption",
                    description="Critical payment infrastructure failure affecting transaction processing",
                    seed_ticker=payment_tickers[0],
                    shock_magnitude=-0.14,
                    simulation_type=SimulationType.MATRIX_PROPAGATION,
                    damping_factor=0.70,
                    metadata={'category': 'infrastructure_crisis', 'critical_system': True}
                )
                scenario_ids.append(self.create_scenario(scenario))
            
            self.logger.info(f"Created {len(scenario_ids)} predefined scenarios")
            return scenario_ids
            
        except Exception as e:
            self.logger.error(f"Error creating predefined scenarios: {e}")
            return []
    
    def create_scenario_from_template(self, template_name: str, 
                                    seed_ticker: str,
                                    **kwargs) -> Optional[str]:
        """Create scenario from predefined template."""
        templates = {
            'mild_shock': {
                'shock_magnitude': -0.03,
                'damping_factor': 0.90,
                'simulation_type': SimulationType.MATRIX_PROPAGATION,
                'description': 'Mild market shock with limited propagation'
            },
            'moderate_shock': {
                'shock_magnitude': -0.08,
                'damping_factor': 0.85,
                'simulation_type': SimulationType.MATRIX_PROPAGATION,
                'description': 'Moderate shock with typical market response'
            },
            'severe_shock': {
                'shock_magnitude': -0.15,
                'damping_factor': 0.75,
                'simulation_type': SimulationType.STRESS_TEST,
                'description': 'Severe shock testing systemic resilience'
            },
            'extreme_shock': {
                'shock_magnitude': -0.25,
                'damping_factor': 0.65,
                'simulation_type': SimulationType.MONTE_CARLO,
                'description': 'Extreme scenario for worst-case analysis',
                'metadata': {'monte_carlo_runs': 1000}
            },
            'uncertainty_analysis': {
                'shock_magnitude': -0.10,
                'damping_factor': 0.80,
                'simulation_type': SimulationType.MONTE_CARLO,
                'description': 'Monte Carlo analysis with uncertainty quantification',
                'metadata': {'monte_carlo_runs': 2000}
            }
        }
        
        if template_name not in templates:
            self.logger.error(f"Unknown template: {template_name}")
            return None
        
        try:
            template = templates[template_name]
            
            # Override template values with provided kwargs
            template.update(kwargs)
            
            scenario = Scenario(
                name=f"{template_name.title().replace('_', ' ')} - {seed_ticker}",
                description=template['description'],
                seed_ticker=seed_ticker,
                shock_magnitude=template['shock_magnitude'],
                simulation_type=template['simulation_type'],
                damping_factor=template['damping_factor'],
                max_iterations=template.get('max_iterations', 100),
                metadata={
                    'template': template_name,
                    'custom_parameters': kwargs
                }
            )
            
            return self.create_scenario(scenario)
            
        except Exception as e:
            self.logger.error(f"Error creating scenario from template {template_name}: {e}")
            return None
    
    def get_scenarios_by_category(self, category: str) -> Dict[str, Scenario]:
        """Get scenarios by metadata category."""
        filtered_scenarios = {}
        
        for scenario_id, scenario in self.scenarios.items():
            if scenario.metadata and scenario.metadata.get('category') == category:
                filtered_scenarios[scenario_id] = scenario
        
        return filtered_scenarios
    
    def get_scenarios_by_ticker(self, ticker: str) -> Dict[str, Scenario]:
        """Get scenarios for specific ticker."""
        filtered_scenarios = {}
        
        for scenario_id, scenario in self.scenarios.items():
            if scenario.seed_ticker == ticker:
                filtered_scenarios[scenario_id] = scenario
        
        return filtered_scenarios
    
    def validate_scenario(self, scenario: Scenario) -> List[str]:
        """Validate scenario configuration."""
        errors = []
        
        try:
            # Check if seed ticker exists
            query = "MATCH (c:Company {ticker: $ticker}) RETURN c.ticker"
            result = neo4j_manager.execute_query(query, {'ticker': scenario.seed_ticker})
            
            if not result:
                errors.append(f"Seed ticker '{scenario.seed_ticker}' not found in database")
            
            # Validate shock magnitude
            if abs(scenario.shock_magnitude) > 0.5:
                errors.append(f"Shock magnitude {scenario.shock_magnitude} seems unrealistic (>50%)")
            
            # Validate damping factor
            if not (0.1 <= scenario.damping_factor <= 0.99):
                errors.append(f"Damping factor {scenario.damping_factor} should be between 0.1 and 0.99")
            
            # Validate max iterations
            if scenario.max_iterations <= 0 or scenario.max_iterations > 1000:
                errors.append(f"Max iterations {scenario.max_iterations} should be between 1 and 1000")
            
            # Check include/exclude ticker conflicts
            if scenario.include_tickers and scenario.exclude_tickers:
                overlap = set(scenario.include_tickers) & set(scenario.exclude_tickers)
                if overlap:
                    errors.append(f"Tickers appear in both include and exclude lists: {overlap}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def export_scenarios(self, output_file: str, scenario_ids: Optional[List[str]] = None) -> bool:
        """Export scenarios to file."""
        try:
            scenarios_to_export = {}
            
            if scenario_ids:
                for scenario_id in scenario_ids:
                    if scenario_id in self.scenarios:
                        scenarios_to_export[scenario_id] = self.scenarios[scenario_id]
            else:
                scenarios_to_export = self.scenarios
            
            # Convert to serializable format
            export_data = {}
            for scenario_id, scenario in scenarios_to_export.items():
                scenario_dict = asdict(scenario)
                # Convert datetime to string
                if scenario_dict['created_date']:
                    scenario_dict['created_date'] = scenario_dict['created_date'].isoformat()
                # Convert enum to string
                scenario_dict['simulation_type'] = scenario.simulation_type.value
                export_data[scenario_id] = scenario_dict
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(export_data)} scenarios to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting scenarios: {e}")
            return False
    
    def import_scenarios(self, input_file: str, overwrite: bool = False) -> int:
        """Import scenarios from file."""
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for scenario_id, scenario_dict in import_data.items():
                # Skip if exists and not overwriting
                if scenario_id in self.scenarios and not overwrite:
                    continue
                
                # Convert back from serialized format
                if scenario_dict['created_date']:
                    scenario_dict['created_date'] = datetime.fromisoformat(scenario_dict['created_date'])
                
                scenario_dict['simulation_type'] = SimulationType(scenario_dict['simulation_type'])
                
                scenario = Scenario(**scenario_dict)
                self.scenarios[scenario_id] = scenario
                imported_count += 1
            
            if imported_count > 0:
                self._save_scenarios()
            
            self.logger.info(f"Imported {imported_count} scenarios from {input_file}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Error importing scenarios: {e}")
            return 0
    
    def _load_scenarios(self):
        """Load scenarios from file."""
        try:
            if Path(self.scenarios_file).exists():
                with open(self.scenarios_file, 'r') as f:
                    data = json.load(f)
                
                for scenario_id, scenario_dict in data.items():
                    # Convert datetime string back to datetime
                    if scenario_dict.get('created_date'):
                        scenario_dict['created_date'] = datetime.fromisoformat(scenario_dict['created_date'])
                    
                    # Convert simulation type string back to enum
                    if isinstance(scenario_dict.get('simulation_type'), str):
                        scenario_dict['simulation_type'] = SimulationType(scenario_dict['simulation_type'])
                    
                    self.scenarios[scenario_id] = Scenario(**scenario_dict)
                
                self.logger.info(f"Loaded {len(self.scenarios)} scenarios from {self.scenarios_file}")
            else:
                self.logger.info("No existing scenarios file found")
                
        except Exception as e:
            self.logger.error(f"Error loading scenarios: {e}")
            self.scenarios = {}
    
    def _save_scenarios(self):
        """Save scenarios to file."""
        try:
            # Ensure directory exists
            Path(self.scenarios_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {}
            for scenario_id, scenario in self.scenarios.items():
                scenario_dict = asdict(scenario)
                # Convert datetime to string
                if scenario_dict['created_date']:
                    scenario_dict['created_date'] = scenario_dict['created_date'].isoformat()
                # Convert enum to string
                scenario_dict['simulation_type'] = scenario.simulation_type.value
                data[scenario_id] = scenario_dict
            
            with open(self.scenarios_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving scenarios: {e}")
    
    def _generate_scenario_id(self, name: str) -> str:
        """Generate unique scenario ID from name."""
        # Create base ID from name
        base_id = name.lower().replace(' ', '_').replace('-', '_')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')
        
        # Ensure uniqueness
        scenario_id = base_id
        counter = 1
        while scenario_id in self.scenarios:
            scenario_id = f"{base_id}_{counter}"
            counter += 1
        
        return scenario_id