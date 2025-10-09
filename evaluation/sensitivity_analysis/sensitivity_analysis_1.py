"""
Sensitivity Analysis of IoT Data Quality Pipeline
Part 1: Setup and Single-Parameter Analysis

This notebook performs comprehensive sensitivity analysis for scientific evaluation
of the IoT data quality detection and conformance checking pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import product
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

import sys
sys.path.append('../../../..')

from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.config.settings import QUALITY_CONFIG, QUALITY_THRESHOLDS

# ============================================================================
# 1. EXPERIMENTAL DESIGN
# ============================================================================

class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis framework for IoT quality pipeline.
    
    Analyzes:
    - Single-parameter sensitivity
    - Multi-parameter interactions
    - Threshold robustness
    - Statistical significance
    """
    
    def __init__(self, base_config: Dict[str, Any] = None):
        self.base_config = base_config or self._get_default_config()
        self.results = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default experimental configuration."""
        return {
            'duration_hours': 1,
            'num_cases': 15,
            'conformance_threshold': 0.7,
            'noise_threshold': 0.2,
            'num_replications': 5
        }
    
    def run_single_parameter_sweep(
        self, 
        parameter_name: str,
        values: List[float],
        metric_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Perform single-parameter sensitivity sweep.
        
        :param parameter_name: Name of parameter to vary
        :param values: List of values to test
        :param metric_names: Metrics to record
        :return: DataFrame with results
        """
        if metric_names is None:
            metric_names = [
                'fitness', 'precision', 'num_quality_issues',
                'detection_rate', 'avg_confidence', 'conformance_triggered'
            ]
        
        results = []
        
        for value in values:
            print(f"Testing {parameter_name} = {value:.3f}")
            
            # Run multiple replications for statistical power
            for rep in range(self.base_config['num_replications']):
                config = self.base_config.copy()
                metrics = self._run_experiment(parameter_name, value, config, rep)
                
                result = {
                    'parameter': parameter_name,
                    'value': value,
                    'replication': rep,
                    **metrics
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def _run_experiment(
        self, 
        param_name: str, 
        param_value: float,
        config: Dict[str, Any],
        seed: int
    ) -> Dict[str, float]:
        """
        Run single experimental configuration.
        
        :param param_name: Parameter being varied
        :param param_value: Value for this run
        :param config: Base configuration
        :param seed: Random seed
        :return: Dictionary of metrics
        """
        np.random.seed(seed)
        
        # Apply parameter change
        self._apply_parameter(param_name, param_value)
        
        # Generate environment and run pipeline
        try:
            env = IoTEnvironment(
                name=f"Sensitivity_{param_name}",
                duration_hours=config['duration_hours'],
                num_cases=config['num_cases']
            )
            
            data = env.generate_data()
            
            pipeline = PipelineManager(
                conformance_threshold=config['conformance_threshold']
            )
            
            results = pipeline.run(data, env)
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            metrics = self._get_empty_metrics()
        
        return metrics
    
    def _apply_parameter(self, param_name: str, value: float):
        """Apply parameter value to configuration."""
        if param_name == 'c1_probability':
            QUALITY_CONFIG.inadequate_sampling_rate['probability'] = value
        elif param_name == 'c3_probability':
            QUALITY_CONFIG.sensor_noise['probability'] = value
        elif param_name == 'sampling_rate_min':
            QUALITY_CONFIG.inadequate_sampling_rate['min_sampling_rate'] = value
        elif param_name == 'noise_std':
            QUALITY_CONFIG.sensor_noise['noise_std'] = value
        elif param_name == 'conformance_threshold':
            # Handled in pipeline initialization
            pass
        elif param_name == 'fitness_threshold':
            QUALITY_THRESHOLDS.process_mining['fitness_threshold'] = value
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from pipeline results."""
        quality_issues = results.get('quality_issues', [])
        process_model = results.get('process_model', {})
        metrics_dict = process_model.get('metrics', {})
        quality_analysis = process_model.get('quality_analysis', {})
        
        return {
            'fitness': metrics_dict.get('fitness', 0.0),
            'precision': metrics_dict.get('precision', 0.0),
            'quality_weighted_fitness': metrics_dict.get('quality_weighted_fitness', 0.0),
            'num_quality_issues': len(quality_issues),
            'high_severity_issues': len([qi for qi in quality_issues 
                                         if qi.get('severity') == 'high']),
            'avg_confidence': np.mean([qi.get('confidence', 0) 
                                      for qi in quality_issues]) if quality_issues else 0.0,
            'conformance_triggered': int(quality_analysis.get('has_conformance_issues', False)),
            'num_conformance_issues': quality_analysis.get('issue_count', 0),
            'num_backtrack_results': len(quality_analysis.get('backtracking_results', []))
        }
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics for failed experiments."""
        return {
            'fitness': 0.0,
            'precision': 0.0,
            'quality_weighted_fitness': 0.0,
            'num_quality_issues': 0,
            'high_severity_issues': 0,
            'avg_confidence': 0.0,
            'conformance_triggered': 0,
            'num_conformance_issues': 0,
            'num_backtrack_results': 0
        }


# ============================================================================
# 2. SINGLE-PARAMETER SENSITIVITY ANALYSIS
# ============================================================================

def analyze_quality_issue_probability_sensitivity():
    """
    Analyze sensitivity to quality issue probability (C1).
    
    Research Question: How does the probability of inadequate sampling
    affect detection accuracy and model quality?
    """
    print("=" * 70)
    print("EXPERIMENT 1: Quality Issue Probability Sensitivity (C1)")
    print("=" * 70)
    
    analyzer = SensitivityAnalyzer()
    
    # Test range: 0.0 to 0.8 (0% to 80% probability)
    probabilities = np.linspace(0.0, 0.8, 9)
    
    results_df = analyzer.run_single_parameter_sweep(
        parameter_name='c1_probability',
        values=probabilities
    )
    
    # Statistical analysis
    print("\nStatistical Summary:")
    print(results_df.groupby('value')[['fitness', 'precision', 'num_quality_issues']].agg(
        ['mean', 'std', 'min', 'max']
    ))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Fitness vs Probability
    ax1 = axes[0, 0]
    sns.lineplot(data=results_df, x='value', y='fitness', 
                errorbar='sd', marker='o', ax=ax1)
    ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, 
                label='Threshold')
    ax1.set_xlabel('C1 Issue Probability')
    ax1.set_ylabel('Model Fitness')
    ax1.set_title('Impact on Model Fitness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision vs Probability
    ax2 = axes[0, 1]
    sns.lineplot(data=results_df, x='value', y='precision',
                errorbar='sd', marker='o', ax=ax2)
    ax2.set_xlabel('C1 Issue Probability')
    ax2.set_ylabel('Model Precision')
    ax2.set_title('Impact on Model Precision')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection Rate
    ax3 = axes[1, 0]
    sns.lineplot(data=results_df, x='value', y='num_quality_issues',
                errorbar='sd', marker='o', ax=ax3)
    ax3.set_xlabel('C1 Issue Probability')
    ax3.set_ylabel('Number of Issues Detected')
    ax3.set_title('Detection Effectiveness')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confidence Distribution
    ax4 = axes[1, 1]
    sns.boxplot(data=results_df, x='value', y='avg_confidence', ax=ax4)
    ax4.set_xlabel('C1 Issue Probability')
    ax4.set_ylabel('Average Confidence Score')
    ax4.set_title('Detection Confidence Distribution')
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('sensitivity_c1_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


def analyze_conformance_threshold_sensitivity():
    """
    Analyze sensitivity to conformance checking threshold.
    
    Research Question: How does the conformance threshold affect
    the balance between false positives and false negatives?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Conformance Threshold Sensitivity")
    print("=" * 70)
    
    analyzer = SensitivityAnalyzer()
    
    # Test range: 0.5 to 0.9
    thresholds = np.linspace(0.5, 0.9, 9)
    
    results_df = analyzer.run_single_parameter_sweep(
        parameter_name='conformance_threshold',
        values=thresholds
    )
    
    # Calculate detection rates
    grouped = results_df.groupby('value').agg({
        'conformance_triggered': ['mean', 'std'],
        'num_conformance_issues': ['mean', 'std'],
        'num_backtrack_results': ['mean', 'std']
    })
    
    print("\nDetection Rate Analysis:")
    print(grouped)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Conformance Triggering Rate
    ax1 = axes[0]
    trigger_rate = results_df.groupby('value')['conformance_triggered'].mean()
    trigger_std = results_df.groupby('value')['conformance_triggered'].std()
    
    ax1.plot(trigger_rate.index, trigger_rate.values, 
            marker='o', linewidth=2, label='Trigger Rate')
    ax1.fill_between(trigger_rate.index,
                     trigger_rate.values - trigger_std.values,
                     trigger_rate.values + trigger_std.values,
                     alpha=0.3)
    ax1.set_xlabel('Conformance Threshold')
    ax1.set_ylabel('Conformance Triggering Rate')
    ax1.set_title('Threshold Impact on Detection Sensitivity')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: ROC-like Analysis
    ax2 = axes[1]
    sns.scatterplot(data=results_df, x='conformance_triggered', 
                   y='num_backtrack_results', hue='value',
                   palette='viridis', s=100, ax=ax2)
    ax2.set_xlabel('Conformance Issues Detected')
    ax2.set_ylabel('Successful Backtracking Results')
    ax2.set_title('Detection vs. Backtracking Success')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_conformance_threshold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


# ============================================================================
# 3. RUN ANALYSES
# ============================================================================

if __name__ == "__main__":
    print("Starting Sensitivity Analysis for IoT Data Quality Pipeline")
    print("=" * 70)
    
    # Experiment 1: Quality Issue Probability
    results_c1 = analyze_quality_issue_probability_sensitivity()
    
    # Experiment 2: Conformance Threshold
    results_threshold = analyze_conformance_threshold_sensitivity()
    
    # Save results
    results_c1.to_csv('sensitivity_results_c1.csv', index=False)
    results_threshold.to_csv('sensitivity_results_threshold.csv', index=False)
    
    print("\n" + "=" * 70)
    print("Sensitivity Analysis Complete!")
    print("Results saved to CSV files")
    print("=" * 70)
