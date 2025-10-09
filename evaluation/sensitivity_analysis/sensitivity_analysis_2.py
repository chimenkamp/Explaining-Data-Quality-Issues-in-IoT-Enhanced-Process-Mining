"""
Sensitivity Analysis of IoT Data Quality Pipeline
Part 2: Multi-Parameter Interaction Analysis

This notebook analyzes interaction effects between multiple parameters
and creates sophisticated visualization matrices.
"""
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150

import sys
sys.path.append('../../../..')

from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.config.settings import QUALITY_CONFIG, QUALITY_THRESHOLDS


# ============================================================================
# 1. TWO-PARAMETER INTERACTION ANALYSIS
# ============================================================================

class InteractionAnalyzer:
    """Analyzes interactions between pairs of parameters."""
    
    def __init__(self, num_replications: int = 3):
        self.num_replications = num_replications
        self.results = []
    
    def run_two_parameter_sweep(
        self,
        param1_name: str,
        param1_values: np.ndarray,
        param2_name: str,
        param2_values: np.ndarray
    ) -> pd.DataFrame:
        """
        Run full factorial experiment for two parameters.
        
        :param param1_name: First parameter name
        :param param1_values: Values for first parameter
        :param param2_name: Second parameter name
        :param param2_values: Values for second parameter
        :return: DataFrame with results
        """
        results = []
        total = len(param1_values) * len(param2_values) * self.num_replications
        count = 0
        
        for p1_val in param1_values:
            for p2_val in param2_values:
                print(f"Progress: {count}/{total} - " +
                      f"{param1_name}={p1_val:.3f}, {param2_name}={p2_val:.3f}")
                
                for rep in range(self.num_replications):
                    metrics = self._run_two_param_experiment(
                        param1_name, p1_val,
                        param2_name, p2_val,
                        seed=count + rep
                    )
                    
                    result = {
                        'param1': param1_name,
                        'param1_value': p1_val,
                        'param2': param2_name,
                        'param2_value': p2_val,
                        'replication': rep,
                        **metrics
                    }
                    results.append(result)
                    count += 1
        
        return pd.DataFrame(results)
    
    def _run_two_param_experiment(
        self,
        param1_name: str,
        param1_value: float,
        param2_name: str,
        param2_value: float,
        seed: int
    ) -> Dict[str, float]:
        """Run experiment with two parameters varied."""
        np.random.seed(seed)
        
        # Apply both parameters
        self._apply_parameter(param1_name, param1_value)
        self._apply_parameter(param2_name, param2_value)
        
        try:
            env = IoTEnvironment(
                name="Interaction_Analysis",
                duration_hours=1,
                num_cases=12
            )
            
            data = env.generate_data()
            pipeline = PipelineManager(conformance_threshold=0.7)
            results = pipeline.run(data, env)
            
            return self._extract_metrics(results)
            
        except Exception as e:
            print(f"Error: {e}")
            return self._get_empty_metrics()
    
    def _apply_parameter(self, param_name: str, value: float):
        """Apply parameter value."""
        if param_name == 'c1_probability':
            QUALITY_CONFIG.inadequate_sampling_rate['probability'] = value
        elif param_name == 'c3_probability':
            QUALITY_CONFIG.sensor_noise['probability'] = value
        elif param_name == 'c2_probability':
            QUALITY_CONFIG.poor_sensor_placement['probability'] = value
        elif param_name == 'noise_std':
            QUALITY_CONFIG.sensor_noise['noise_std'] = value
        elif param_name == 'sampling_rate_min':
            QUALITY_CONFIG.inadequate_sampling_rate['min_sampling_rate'] = value
        elif param_name == 'fitness_threshold':
            QUALITY_THRESHOLDS.process_mining['fitness_threshold'] = value
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from pipeline results."""
        quality_issues = results.get('quality_issues', [])
        process_model = results.get('process_model', {})
        metrics_dict = process_model.get('metrics', {})
        quality_analysis = process_model.get('quality_analysis', {})
        
        return {
            'fitness': metrics_dict.get('fitness', 0.0),
            'precision': metrics_dict.get('precision', 0.0),
            'num_quality_issues': len(quality_issues),
            'avg_confidence': np.mean([qi.get('confidence', 0) 
                                      for qi in quality_issues]) if quality_issues else 0.0,
            'conformance_triggered': int(quality_analysis.get('has_conformance_issues', False)),
            'detection_accuracy': self._calculate_detection_accuracy(quality_issues)
        }
    
    def _calculate_detection_accuracy(self, quality_issues: List[Dict]) -> float:
        """Calculate detection accuracy metric."""
        if not quality_issues:
            return 0.0
        
        high_conf = len([qi for qi in quality_issues if qi.get('confidence', 0) > 0.7])
        return high_conf / len(quality_issues)
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics."""
        return {
            'fitness': 0.0,
            'precision': 0.0,
            'num_quality_issues': 0,
            'avg_confidence': 0.0,
            'conformance_triggered': 0,
            'detection_accuracy': 0.0
        }


# ============================================================================
# 2. INTERACTION MATRIX VISUALIZATION
# ============================================================================

def create_interaction_matrix(results_df: pd.DataFrame):
    """
    Create publication-quality interaction matrix showing
    parameter interactions on multiple metrics.
    
    :param results_df: Results from two-parameter sweep
    """
    # Aggregate results
    agg_df = results_df.groupby(['param1_value', 'param2_value']).agg({
        'fitness': 'mean',
        'precision': 'mean',
        'num_quality_issues': 'mean',
        'detection_accuracy': 'mean'
    }).reset_index()
    
    # Create 2x2 grid of heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = [
        ('fitness', 'Model Fitness', 'RdYlGn'),
        ('precision', 'Model Precision', 'RdYlGn'),
        ('num_quality_issues', 'Issues Detected', 'YlOrRd'),
        ('detection_accuracy', 'Detection Accuracy', 'RdYlGn')
    ]
    
    param1_name = results_df['param1'].iloc[0]
    param2_name = results_df['param2'].iloc[0]
    
    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Pivot for heatmap
        pivot_df = agg_df.pivot(
            index='param2_value',
            columns='param1_value',
            values=metric
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            center=pivot_df.mean().mean(),
            cbar_kws={'label': title},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_xlabel(f'{param1_name} Value', fontweight='bold')
        ax.set_ylabel(f'{param2_name} Value', fontweight='bold')
        ax.set_title(f'{title}\n(Interaction Effect)', fontweight='bold')
        
        # Add interaction strength annotation
        interaction_strength = _calculate_interaction_strength(pivot_df)
        ax.text(
            0.02, 0.98,
            f'Interaction: {interaction_strength:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig('interaction_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def _calculate_interaction_strength(pivot_df: pd.DataFrame) -> float:
    """
    Calculate interaction strength as variance of differences.
    
    Higher values indicate stronger interaction effects.
    """
    # Calculate row-wise and column-wise differences
    row_diffs = pivot_df.diff(axis=1).abs().mean().mean()
    col_diffs = pivot_df.diff(axis=0).abs().mean().mean()
    
    # Interaction strength is the combined variability
    return (row_diffs + col_diffs) / 2


# ============================================================================
# 3. 3D RESPONSE SURFACE VISUALIZATION
# ============================================================================

def create_response_surface(results_df: pd.DataFrame, metric: str = 'fitness'):
    """
    Create 3D response surface showing how metric varies
    with two parameters.
    
    :param results_df: Results dataframe
    :param metric: Metric to visualize
    """
    # Aggregate results
    agg_df = results_df.groupby(['param1_value', 'param2_value'])[metric].mean().reset_index()
    
    # Create meshgrid for interpolation
    x = agg_df['param1_value'].values
    y = agg_df['param2_value'].values
    z = agg_df[metric].values
    
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 6))
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(
        Xi, Yi, Zi,
        cmap='viridis',
        alpha=0.8,
        edgecolor='none'
    )
    
    ax1.scatter(x, y, z, c='red', marker='o', s=30, alpha=0.6)
    
    param1_name = results_df['param1'].iloc[0]
    param2_name = results_df['param2'].iloc[0]
    
    ax1.set_xlabel(f'\n{param1_name}', fontweight='bold')
    ax1.set_ylabel(f'\n{param2_name}', fontweight='bold')
    ax1.set_zlabel(f'\n{metric.title()}', fontweight='bold')
    ax1.set_title('Response Surface', fontweight='bold', pad=20)
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(Xi, Yi, Zi, levels=15, cmap='viridis')
    ax2.scatter(x, y, c='red', marker='o', s=30, alpha=0.6)
    
    ax2.set_xlabel(f'{param1_name}', fontweight='bold')
    ax2.set_ylabel(f'{param2_name}', fontweight='bold')
    ax2.set_title('Contour Plot', fontweight='bold')
    
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'response_surface_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


# ============================================================================
# 4. STATISTICAL SIGNIFICANCE ANALYSIS
# ============================================================================

def analyze_statistical_significance(results_df: pd.DataFrame):
    """
    Perform statistical tests to assess significance of
    parameter effects and interactions.
    
    :param results_df: Results dataframe
    """
    from scipy.stats import f_oneway, ttest_ind
    
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    
    # Test main effects
    param1_groups = [
        results_df[results_df['param1_value'] == val]['fitness'].values
        for val in results_df['param1_value'].unique()
    ]
    
    f_stat, p_value = f_oneway(*param1_groups)
    
    print(f"\nMain Effect ({results_df['param1'].iloc[0]}):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
    
    # Test interaction effect using two-way comparison
    low_low = results_df[
        (results_df['param1_value'] == results_df['param1_value'].min()) &
        (results_df['param2_value'] == results_df['param2_value'].min())
    ]['fitness'].values
    
    high_high = results_df[
        (results_df['param1_value'] == results_df['param1_value'].max()) &
        (results_df['param2_value'] == results_df['param2_value'].max())
    ]['fitness'].values
    
    t_stat, p_value_interaction = ttest_ind(low_low, high_high)
    
    print(f"\nInteraction Effect:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value_interaction:.6f}")
    print(f"  Significant: {'Yes' if p_value_interaction < 0.05 else 'No'} (α=0.05)")
    
    # Effect size (Cohen's d)
    cohens_d = (low_low.mean() - high_high.mean()) / np.sqrt(
        (low_low.std()**2 + high_high.std()**2) / 2
    )
    
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    
    effect_interpretation = (
        "Small" if abs(cohens_d) < 0.5 else
        "Medium" if abs(cohens_d) < 0.8 else
        "Large"
    )
    print(f"  Interpretation: {effect_interpretation} effect")
    
    return {
        'main_effect_p': p_value,
        'interaction_p': p_value_interaction,
        'effect_size': cohens_d
    }


# ============================================================================
# 5. RUN INTERACTION ANALYSIS
# ============================================================================

def run_full_interaction_analysis():
    """Run complete interaction analysis."""
    
    print("Starting Two-Parameter Interaction Analysis")
    print("=" * 70)
    
    analyzer = InteractionAnalyzer(num_replications=3)
    
    # Experiment: C1 Probability vs C3 Probability
    print("\nEXPERIMENT: C1 (Sampling) vs C3 (Noise) Interaction")
    
    c1_values = np.linspace(0.0, 0.6, 5)
    c3_values = np.linspace(0.0, 0.6, 5)
    
    results_df = analyzer.run_two_parameter_sweep(
        'c1_probability', c1_values,
        'c3_probability', c3_values
    )
    
    # Save results
    results_df.to_csv('interaction_c1_c3.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Interaction matrix
    create_interaction_matrix(results_df)
    
    # Response surfaces for each metric
    for metric in ['fitness', 'precision', 'detection_accuracy']:
        create_response_surface(results_df, metric)
    
    # Statistical analysis
    stats = analyze_statistical_significance(results_df)
    
    return results_df, stats


if __name__ == "__main__":
    results_df, stats = run_full_interaction_analysis()
    
    print("\n" + "=" * 70)
    print("Interaction Analysis Complete!")
    print("=" * 70)
