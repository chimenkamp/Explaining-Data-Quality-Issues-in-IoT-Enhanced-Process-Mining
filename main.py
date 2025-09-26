# main.py
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.explainability.insights import InsightGenerator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create synthetic IoT environment
    logger.info("Creating synthetic IoT environment...")
    env = IoTEnvironment(
        name="Manufacturing Line",
        duration_hours=24,
        num_machines=3
    )
    
    # Add sensors with various quality issues
    env.add_welding_station()
    env.add_inspection_station() 
    env.add_packaging_station()
    
    # Generate data with quality issues
    logger.info("Generating synthetic data with quality issues...")
    raw_data = env.generate_data()
    
    # Initialize and run pipeline
    logger.info("Running IoT data quality pipeline...")
    pipeline = PipelineManager()
    results = pipeline.run(raw_data, env)
    
    # Generate explainable insights
    logger.info("Generating explainable insights...")
    insight_generator = InsightGenerator()
    insights = insight_generator.generate_insights(results)
    
    # Display results
    print("\n" + "="*60)
    print("IoT DATA QUALITY ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nDetected Quality Issues: {len(results['quality_issues'])}")
    for issue in results['quality_issues']:
        print(f"  - {issue['type']}: {issue['description']}")
    
    print(f"\nProcess Model Fitness: {results['process_model']['fitness']:.3f}")
    print(f"Model Complexity: {results['process_model']['complexity']:.3f}")
    
    print(f"\nTop Insights:")
    for insight in insights[:3]:
        print(f"  - {insight['message']}")
        print(f"    Confidence: {insight['confidence']:.3f}")
        print(f"    Actionable: {insight['actionable']}")
        print()

if __name__ == "__main__":
    main()

# src/__init__.py
"""
IoT Data Quality Pipeline
A system for propagating and interpreting IoT data quality issues
"""
__version__ = "1.0.0"

# src/config/settings.py
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class QualityIssueConfig:
    """Configuration for different types of quality issues"""
    
    # C1: Inadequate Sampling Rate
    inadequate_sampling_rate: Dict[str, Any] = None
    
    # C2: Poor Sensor Placement  
    poor_sensor_placement: Dict[str, Any] = None
    
    # C3: Sensor Noise & Outliers
    sensor_noise: Dict[str, Any] = None
    
    # C4: Sensor Range Too Small
    sensor_range_too_small: Dict[str, Any] = None
    
    # C5: High Data Volume/Velocity
    high_data_volume: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.inadequate_sampling_rate is None:
            self.inadequate_sampling_rate = {
                'probability': 0.15,
                'min_sampling_rate': 0.1,  # Hz
                'max_sampling_rate': 0.5   # Hz
            }
            
        if self.poor_sensor_placement is None:
            self.poor_sensor_placement = {
                'probability': 0.10,
                'overlap_factor': 0.3,
                'inconsistency_factor': 0.4
            }
            
        if self.sensor_noise is None:
            self.sensor_noise = {
                'probability': 0.25,
                'noise_std': 0.1,
                'outlier_probability': 0.05,
                'outlier_magnitude': 3.0
            }
            
        if self.sensor_range_too_small is None:
            self.sensor_range_too_small = {
                'probability': 0.12,
                'blind_spot_fraction': 0.2,
                'range_reduction_factor': 0.6
            }
            
        if self.high_data_volume is None:
            self.high_data_volume = {
                'probability': 0.08,
                'drop_probability': 0.15,
                'timestamp_drift_std': 2.0  # seconds
            }

@dataclass 
class PipelineConfig:
    """Configuration for pipeline stages"""
    
    # Event abstraction parameters
    event_window_size: int = 10
    min_event_duration: float = 1.0
    
    # Case correlation parameters
    case_timeout: float = 300.0  # 5 minutes
    min_case_events: int = 2
    
    # Process mining parameters
    noise_threshold: float = 0.05
    dependency_threshold: float = 0.1
    
    # Probabilistic reasoning
    prior_probabilities: Dict[str, float] = None
    
    def __post_init__(self):
        if self.prior_probabilities is None:
            self.prior_probabilities = {
                'C1_inadequate_sampling': 0.2,
                'C2_poor_placement': 0.15,
                'C3_sensor_noise': 0.3,
                'C4_range_too_small': 0.2,
                'C5_high_volume': 0.15
            }

# Global configuration instances
QUALITY_CONFIG = QualityIssueConfig()
PIPELINE_CONFIG = PipelineConfig()