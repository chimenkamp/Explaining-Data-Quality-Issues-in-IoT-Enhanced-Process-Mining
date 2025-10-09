"""
Quick Start Demo: Sensitivity Analysis
Practical execution guide with minimal setup

This notebook provides a streamlined demonstration of the sensitivity
analysis with reduced computational requirements for quick validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook")
plt.rcParams['figure.dpi'] = 150

import sys
sys.path.append('../../../..')

from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.config.settings import QUALITY_CONFIG, QUALITY_THRESHOLDS


# ============================================================================
# QUICK DEMO: 5-MINUTE SENSITIVITY CHECK
# ============================================================================

def quick_sensitivity_demo():
    """
    5-minute demonstration of core sensitivity concepts.
    Uses reduced parameters for fast execution.
    """
    print("=" * 70)
    print("QUICK SENSITIVITY DEMO (5 minutes)")
    print("=" * 70)
    
    # Test 3 parameter values with 2 replications each
    c1_values = [0.0, 0.4, 0.8]  # Low, Medium, High
    results = []
    
    for c1_prob in c1_values:
        for rep in range(2):
            print(f"Testing C1={c1_prob:.1f}, Rep {rep+1}/2...")
            
            # Configure environment
            QUALITY_CONFIG.inadequate_sampling_rate['probability'] = c1_prob
            
            # Run pipeline
            np.random.seed(rep)
            env = IoTEnvironment(
                name="Quick_Demo",
                duration_hours=1,
                num_cases=10
            )
            
            data = env.generate_data()
            pipeline = PipelineManager(conformance_threshold=0.7)
            pipeline_results = pipeline.run(data, env)
            
            # Extract metrics
            metrics = pipeline_results.get('process_model', {}).get('metrics', {})
            quality_issues = pipeline_results.get('quality_issues', [])
            
            results.append({
                'c1_probability': c1_prob,
                'replication': rep,
                'fitness': metrics.get('fitness', 0),
                'precision': metrics.get('precision', 0),
                'num_issues': len(quality_issues)
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Fitness
    ax1 = axes[0]
    grouped = results_df.groupby('c1_probability')['fitness'].agg(['mean', 'std'])
    ax1.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                marker='o', linewidth=2, capsize=5, markersize=8)
    ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax1.set_xlabel('C1 Probability', fontweight='bold')
    ax1.set_ylabel('Fitness', fontweight='bold')
    ax1.set_title('Impact on Fitness', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision
    ax2 = axes[1]
    grouped = results_df.groupby('c1_probability')['precision'].agg(['mean', 'std'])
    ax2.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                marker='s', linewidth=2, capsize=5, markersize=8)
    ax2.set_xlabel('C1 Probability', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Impact on Precision', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Detection
    ax3 = axes[2]
    grouped = results_df.groupby('c1_probability')['num_issues'].agg(['mean', 'std'])
    ax3.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                marker='^', linewidth=2, capsize=5, markersize=8)
    ax3.set_xlabel('C1 Probability', fontweight='bold')
    ax3.set_ylabel('Issues Detected', fontweight='bold')
    ax3.set_title('Detection Rate', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Quick Sensitivity Demo Results', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('quick_demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print("\nFitness by C1 Probability:")
    print(results_df.groupby('c1_probability')['fitness'].describe())
    
    print("\nDetection Rate by C1 Probability:")
    print(results_df.groupby('c1_probability')['num_issues'].describe())
    
    # Sensitivity calculation
    fitness_range = results_df['fitness'].max() - results_df['fitness'].min()
    c1_range = results_df['c1_probability'].max() - results_df['c1_probability'].min()
    sensitivity = fitness_range / c1_range if c1_range > 0 else 0
    
    print(f"\nSensitivity Coefficient: {sensitivity:.3f}")
    print(f"  (Change in fitness per unit change in C1)")
    
    if sensitivity > 0.5:
        print("  Interpretation: HIGH sensitivity - C1 is critical parameter")
    elif sensitivity > 0.3:
        print("  Interpretation: MODERATE sensitivity - C1 matters")
    else:
        print("  Interpretation: LOW sensitivity - C1 has minor effect")
    
    return results_df


# ============================================================================
# PARAMETER COMPARISON DEMO
# ============================================================================

def compare_parameters_demo():
    """
    Compare sensitivity across multiple parameters.
    Shows which parameters matter most.
    """
    print("\n" + "=" * 70)
    print("MULTI-PARAMETER COMPARISON DEMO")
    print("=" * 70)
    
    # Parameters to test
    params_to_test = {
        'C1 Sampling': ('c1_probability', [0.0, 0.4, 0.8]),
        'C3 Noise': ('c3_probability', [0.0, 0.4, 0.8]),
        'Conformance Threshold': ('conformance_threshold', [0.5, 0.7, 0.9])
    }
    
    all_results = {}
    
    for param_name, (config_key, values) in params_to_test.items():
        print(f"\nTesting {param_name}...")
        results = []
        
        for value in values:
            # Reset to defaults
            QUALITY_CONFIG.inadequate_sampling_rate['probability'] = 0.2
            QUALITY_CONFIG.sensor_noise['probability'] = 0.2
            conf_threshold = 0.7
            
            # Apply parameter change
            if config_key == 'c1_probability':
                QUALITY_CONFIG.inadequate_sampling_rate['probability'] = value
            elif config_key == 'c3_probability':
                QUALITY_CONFIG.sensor_noise['probability'] = value
            elif config_key == 'conformance_threshold':
                conf_threshold = value
            
            # Run single experiment
            np.random.seed(42)
            env = IoTEnvironment(name="Compare", duration_hours=1, num_cases=10)
            data = env.generate_data()
            pipeline = PipelineManager(conformance_threshold=conf_threshold)
            pipeline_results = pipeline.run(data, env)
            
            metrics = pipeline_results.get('process_model', {}).get('metrics', {})
            
            results.append({
                'parameter': param_name,
                'value': value,
                'fitness': metrics.get('fitness', 0)
            })
        
        all_results[param_name] = pd.DataFrame(results)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for (param_name, df), color in zip(all_results.items(), colors):
        ax.plot(df['value'], df['fitness'], 
               marker='o', linewidth=2, markersize=8,
               label=param_name, color=color)
    
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, 
              linewidth=2, label='Target Threshold')
    
    ax.set_xlabel('Parameter Value (Normalized)', fontweight='bold')
    ax.set_ylabel('Model Fitness', fontweight='bold')
    ax.set_title('Parameter Sensitivity Comparison', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate sensitivities
    print("\n" + "=" * 70)
    print("SENSITIVITY RANKINGS")
    print("=" * 70)
    
    sensitivities = {}
    for param_name, df in all_results.items():
        fitness_range = df['fitness'].max() - df['fitness'].min()
        value_range = df['value'].max() - df['value'].min()
        sensitivity = fitness_range / value_range if value_range > 0 else 0
        sensitivities[param_name] = sensitivity
    
    # Sort by sensitivity
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    
    print("\nParameter Sensitivity Ranking:")
    for i, (param, sens) in enumerate(sorted_sens, 1):
        print(f"  {i}. {param:25s}: {sens:.3f}")
    
    return all_results, sensitivities


# ============================================================================
# INTERACTION EFFECT DEMO
# ============================================================================

def interaction_demo():
    """
    Demonstrate interaction between two parameters.
    Shows how combined effects differ from individual effects.
    """
    print("\n" + "=" * 70)
    print("INTERACTION EFFECT DEMO")
    print("=" * 70)
    
    # Test 3x3 grid of C1 and C3
    c1_values = [0.0, 0.4, 0.8]
    c3_values = [0.0, 0.4, 0.8]
    
    results = []
    
    for c1 in c1_values:
        for c3 in c3_values:
            print(f"Testing C1={c1:.1f}, C3={c3:.1f}...")
            
            QUALITY_CONFIG.inadequate_sampling_rate['probability'] = c1
            QUALITY_CONFIG.sensor_noise['probability'] = c3
            
            np.random.seed(42)
            env = IoTEnvironment(name="Interaction", duration_hours=1, num_cases=10)
            data = env.generate_data()
            pipeline = PipelineManager(conformance_threshold=0.7)
            pipeline_results = pipeline.run(data, env)
            
            metrics = pipeline_results.get('process_model', {}).get('metrics', {})
            
            results.append({
                'c1': c1,
                'c3': c3,
                'fitness': metrics.get('fitness', 0),
                'precision': metrics.get('precision', 0)
            })
    
    results_df = pd.DataFrame(results)
    
    # Create interaction plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Fitness heatmap
    ax1 = axes[0]
    pivot_fitness = results_df.pivot(index='c3', columns='c1', values='fitness')
    
    sns.heatmap(pivot_fitness, annot=True, fmt='.3f', cmap='RdYlGn',
               center=0.7, vmin=0.3, vmax=1.0, ax=ax1,
               cbar_kws={'label': 'Fitness'})
    ax1.set_xlabel('C1 (Sampling) Probability', fontweight='bold')
    ax1.set_ylabel('C3 (Noise) Probability', fontweight='bold')
    ax1.set_title('Fitness Interaction Effect', fontweight='bold')
    
    # Plot 2: Precision heatmap
    ax2 = axes[1]
    pivot_precision = results_df.pivot(index='c3', columns='c1', values='precision')
    
    sns.heatmap(pivot_precision, annot=True, fmt='.3f', cmap='RdYlGn',
               center=0.7, vmin=0.3, vmax=1.0, ax=ax2,
               cbar_kws={'label': 'Precision'})
    ax2.set_xlabel('C1 (Sampling) Probability', fontweight='bold')
    ax2.set_ylabel('C3 (Noise) Probability', fontweight='bold')
    ax2.set_title('Precision Interaction Effect', fontweight='bold')
    
    plt.suptitle('Parameter Interaction Analysis', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('interaction_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Interaction strength analysis
    print("\n" + "=" * 70)
    print("INTERACTION STRENGTH ANALYSIS")
    print("=" * 70)
    
    # Calculate main effects
    c1_main_effect = results_df.groupby('c1')['fitness'].mean().diff().abs().mean()
    c3_main_effect = results_df.groupby('c3')['fitness'].mean().diff().abs().mean()
    
    # Calculate interaction effect (deviation from additivity)
    interaction_matrix = np.zeros((3, 3))
    for i, c1 in enumerate(c1_values):
        for j, c3 in enumerate(c3_values):
            actual = results_df[(results_df['c1'] == c1) & 
                               (results_df['c3'] == c3)]['fitness'].values[0]
            
            # Expected if purely additive
            baseline = results_df[(results_df['c1'] == 0) & 
                                 (results_df['c3'] == 0)]['fitness'].values[0]
            c1_effect = results_df[(results_df['c1'] == c1) & 
                                  (results_df['c3'] == 0)]['fitness'].values[0] - baseline
            c3_effect = results_df[(results_df['c1'] == 0) & 
                                  (results_df['c3'] == c3)]['fitness'].values[0] - baseline
            expected = baseline + c1_effect + c3_effect
            
            interaction_matrix[i, j] = actual - expected
    
    interaction_strength = np.abs(interaction_matrix).mean()
    
    print(f"\nMain Effect (C1): {c1_main_effect:.3f}")
    print(f"Main Effect (C3): {c3_main_effect:.3f}")
    print(f"Interaction Strength: {interaction_strength:.3f}")
    
    if interaction_strength > 0.1:
        print("\n→ STRONG INTERACTION detected!")
        print("  Parameters do NOT act independently.")
        print("  Combined effects differ from sum of individual effects.")
    else:
        print("\n→ WEAK INTERACTION detected.")
        print("  Parameters act relatively independently.")
    
    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_complete_demo():
    """Run all demos in sequence."""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SENSITIVITY ANALYSIS - QUICK START DEMONSTRATION".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    start_time = datetime.now()
    
    # Demo 1: Quick sensitivity
    print("\n▶ Demo 1: Quick Sensitivity Check")
    results_1 = quick_sensitivity_demo()
    
    # Demo 2: Parameter comparison
    print("\n▶ Demo 2: Multi-Parameter Comparison")
    results_2, sensitivities = compare_parameters_demo()
    
    # Demo 3: Interaction effects
    print("\n▶ Demo 3: Interaction Effects")
    results_3 = interaction_demo()
    
    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal Runtime: {elapsed:.1f} seconds")
    print("\nGenerated Files:")
    print("  • quick_demo_results.png")
    print("  • parameter_comparison.png")
    print("  • interaction_demo.png")
    
    print("\nKey Findings:")
    print("  1. Sensitivity analysis successfully identifies critical parameters")
    print("  2. Parameter interactions can be detected and quantified")
    print("  3. Results guide optimization and robustness improvements")
    
    print("\nNext Steps:")
    print("  • Run full sensitivity analysis for publication (see other notebooks)")
    print("  • Increase replications for statistical rigor")
    print("  • Test additional parameter combinations")
    print("  • Validate with real IoT data")
    
    print("\n" + "=" * 70)
    
    return {
        'quick_demo': results_1,
        'parameter_comparison': results_2,
        'interaction_demo': results_3,
        'sensitivities': sensitivities,
        'runtime_seconds': elapsed
    }


if __name__ == "__main__":
    results = run_complete_demo()
