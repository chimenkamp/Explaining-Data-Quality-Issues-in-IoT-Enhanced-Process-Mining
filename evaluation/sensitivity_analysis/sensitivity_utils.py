"""
Sensitivity Analysis Utility Module
Batch execution, result aggregation, and statistical testing utilities

This module provides helper functions for large-scale sensitivity experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for sensitivity experiment."""
    parameter_name: str
    parameter_values: List[float]
    num_replications: int
    num_cases: int
    duration_hours: float
    conformance_threshold: float
    seed_offset: int = 0


@dataclass
class ExperimentResult:
    """Result from single experiment run."""
    parameter_name: str
    parameter_value: float
    replication: int
    fitness: float
    precision: float
    num_quality_issues: int
    avg_confidence: float
    conformance_triggered: bool
    num_backtrack_results: int
    runtime_seconds: float
    timestamp: str


# ============================================================================
# BATCH EXECUTION ENGINE
# ============================================================================

class BatchExecutor:
    """
    Execute sensitivity experiments in batch with progress tracking,
    error handling, and result persistence.
    """
    
    def __init__(self, output_dir: str = "sensitivity_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_cache = []
    
    def run_batch_experiment(
        self,
        config: ExperimentConfig,
        model_function: Callable,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Execute batch of experiments with configuration.
        
        :param config: Experiment configuration
        :param model_function: Function that runs single experiment
        :param save_intermediate: Save results after each value
        :return: DataFrame with all results
        """
        total_runs = len(config.parameter_values) * config.num_replications
        completed = 0
        
        print(f"Starting batch experiment: {config.parameter_name}")
        print(f"Total runs: {total_runs}")
        print(f"Output directory: {self.output_dir}")
        
        all_results = []
        
        for param_value in config.parameter_values:
            value_results = []
            
            for rep in range(config.num_replications):
                completed += 1
                print(f"Progress: {completed}/{total_runs} " +
                      f"({100*completed/total_runs:.1f}%) - " +
                      f"{config.parameter_name}={param_value:.3f}, Rep {rep+1}")
                
                try:
                    result = model_function(
                        param_name=config.parameter_name,
                        param_value=param_value,
                        replication=rep,
                        seed=config.seed_offset + completed
                    )
                    value_results.append(result)
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    # Continue with next run
            
            # Save intermediate results
            if save_intermediate and value_results:
                self._save_intermediate(
                    config.parameter_name,
                    param_value,
                    value_results
                )
        
        # Convert to DataFrame
        results_df = pd.DataFrame([vars(r) for r in all_results])
        
        # Save final results
        self._save_final_results(config.parameter_name, results_df)
        
        print(f"\nBatch experiment complete!")
        print(f"Total successful runs: {len(all_results)}/{total_runs}")
        
        return results_df
    
    def _save_intermediate(
        self,
        param_name: str,
        param_value: float,
        results: List[ExperimentResult]
    ):
        """Save intermediate results for recovery."""
        filename = self.output_dir / f"{param_name}_value_{param_value:.4f}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    
    def _save_final_results(self, param_name: str, results_df: pd.DataFrame):
        """Save final aggregated results."""
        # CSV for easy viewing
        csv_file = self.output_dir / f"{param_name}_results.csv"
        results_df.to_csv(csv_file, index=False)
        
        # Pickle for full data
        pkl_file = self.output_dir / f"{param_name}_results.pkl"
        results_df.to_pickle(pkl_file)
        
        # JSON for metadata
        metadata = {
            'parameter': param_name,
            'num_runs': len(results_df),
            'timestamp': datetime.now().isoformat(),
            'parameter_values': results_df['parameter_value'].unique().tolist(),
            'replications': results_df['replication'].max() + 1
        }
        
        json_file = self.output_dir / f"{param_name}_metadata.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# STATISTICAL ANALYSIS UTILITIES
# ============================================================================

class StatisticalAnalyzer:
    """Statistical testing and validation utilities."""
    
    @staticmethod
    def anova_test(
        results_df: pd.DataFrame,
        grouping_col: str,
        metric_col: str
    ) -> Dict[str, Any]:
        """
        Perform one-way ANOVA to test if parameter significantly affects metric.
        
        :param results_df: Results dataframe
        :param grouping_col: Column for grouping (parameter values)
        :param metric_col: Metric to test
        :return: Dictionary with test results
        """
        groups = []
        for value in results_df[grouping_col].unique():
            group_data = results_df[results_df[grouping_col] == value][metric_col]
            groups.append(group_data)
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        grand_mean = results_df[metric_col].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean)**2 
            for g in groups
        )
        ss_total = sum((results_df[metric_col] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'eta_squared': eta_squared,
            'effect_size': (
                'large' if eta_squared > 0.14 else
                'medium' if eta_squared > 0.06 else
                'small'
            )
        }
    
    @staticmethod
    def pairwise_comparisons(
        results_df: pd.DataFrame,
        grouping_col: str,
        metric_col: str,
        method: str = 'tukey'
    ) -> pd.DataFrame:
        """
        Perform pairwise post-hoc comparisons.
        
        :param results_df: Results dataframe
        :param grouping_col: Column for grouping
        :param metric_col: Metric to compare
        :param method: Comparison method ('tukey', 'bonferroni')
        :return: DataFrame with pairwise comparison results
        """
        from scipy.stats import ttest_ind
        
        values = sorted(results_df[grouping_col].unique())
        comparisons = []
        
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                group1 = results_df[results_df[grouping_col] == values[i]][metric_col]
                group2 = results_df[results_df[grouping_col] == values[j]][metric_col]
                
                t_stat, p_value = ttest_ind(group1, group2)
                
                # Bonferroni correction
                if method == 'bonferroni':
                    n_comparisons = len(values) * (len(values) - 1) / 2
                    p_value_adjusted = min(1.0, p_value * n_comparisons)
                else:
                    p_value_adjusted = p_value
                
                # Cohen's d
                pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
                cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                
                comparisons.append({
                    'group1': values[i],
                    'group2': values[j],
                    'mean_diff': group1.mean() - group2.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'p_value_adjusted': p_value_adjusted,
                    'significant': p_value_adjusted < 0.05,
                    'cohens_d': cohens_d
                })
        
        return pd.DataFrame(comparisons)
    
    @staticmethod
    def normality_test(data: np.ndarray) -> Dict[str, Any]:
        """Test if data follows normal distribution."""
        stat, p_value = stats.shapiro(data)
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def calculate_confidence_intervals(
        results_df: pd.DataFrame,
        grouping_col: str,
        metric_col: str,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """Calculate confidence intervals for each group."""
        groups = []
        
        for value in sorted(results_df[grouping_col].unique()):
            group_data = results_df[results_df[grouping_col] == value][metric_col]
            
            mean = group_data.mean()
            sem = stats.sem(group_data)
            ci = stats.t.interval(
                confidence,
                len(group_data) - 1,
                loc=mean,
                scale=sem
            )
            
            groups.append({
                grouping_col: value,
                'mean': mean,
                'std': group_data.std(),
                'sem': sem,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'n': len(group_data)
            })
        
        return pd.DataFrame(groups)


# ============================================================================
# RESULT AGGREGATION
# ============================================================================

class ResultAggregator:
    """Aggregate and summarize sensitivity results."""
    
    @staticmethod
    def create_summary_table(
        results_dfs: Dict[str, pd.DataFrame],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create summary table across multiple experiments.
        
        :param results_dfs: Dict of experiment_name -> results_df
        :param metrics: List of metrics to summarize
        :return: Summary dataframe
        """
        if metrics is None:
            metrics = ['fitness', 'precision', 'num_quality_issues']
        
        summary_data = []
        
        for exp_name, df in results_dfs.items():
            for metric in metrics:
                if metric in df.columns:
                    summary_data.append({
                        'experiment': exp_name,
                        'metric': metric,
                        'mean': df[metric].mean(),
                        'std': df[metric].std(),
                        'min': df[metric].min(),
                        'max': df[metric].max(),
                        'cv': df[metric].std() / df[metric].mean() if df[metric].mean() != 0 else 0,
                        'range': df[metric].max() - df[metric].min()
                    })
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def calculate_sensitivity_indices(
        results_df: pd.DataFrame,
        param_col: str,
        metric_col: str
    ) -> Dict[str, float]:
        """
        Calculate various sensitivity indices.
        
        :param results_df: Results dataframe
        :param param_col: Parameter column name
        :param metric_col: Metric column name
        :return: Dictionary of sensitivity indices
        """
        # Normalized sensitivity (change in output / change in input)
        param_range = results_df[param_col].max() - results_df[param_col].min()
        metric_range = results_df[metric_col].max() - results_df[metric_col].min()
        
        normalized_sensitivity = (
            metric_range / param_range if param_range > 0 else 0
        )
        
        # Correlation-based sensitivity
        correlation, _ = stats.pearsonr(results_df[param_col], results_df[metric_col])
        
        # Standardized regression coefficient
        scaler = StandardScaler()
        X_std = scaler.fit_transform(results_df[[param_col]])
        y_std = scaler.fit_transform(results_df[[metric_col]])
        
        beta = np.linalg.lstsq(X_std, y_std, rcond=None)[0][0][0]
        
        return {
            'normalized_sensitivity': normalized_sensitivity,
            'correlation': correlation,
            'standardized_coefficient': beta,
            'absolute_change': metric_range,
            'relative_change': metric_range / results_df[metric_col].mean() if results_df[metric_col].mean() != 0 else 0
        }
    
    @staticmethod
    def rank_parameters(
        results_dfs: Dict[str, pd.DataFrame],
        metric: str = 'fitness'
    ) -> pd.DataFrame:
        """
        Rank parameters by their sensitivity to given metric.
        
        :param results_dfs: Dict of parameter_name -> results_df
        :param metric: Metric to use for ranking
        :return: Ranked dataframe
        """
        rankings = []
        
        for param_name, df in results_dfs.items():
            if metric in df.columns and 'parameter_value' in df.columns:
                indices = ResultAggregator.calculate_sensitivity_indices(
                    df, 'parameter_value', metric
                )
                
                rankings.append({
                    'parameter': param_name,
                    'sensitivity': indices['normalized_sensitivity'],
                    'correlation': indices['correlation'],
                    'absolute_change': indices['absolute_change'],
                    'relative_change': indices['relative_change']
                })
        
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('sensitivity', ascending=False)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        return rankings_df


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

class VisualizationHelper:
    """Helper functions for creating sensitivity visualizations."""
    
    @staticmethod
    def prepare_heatmap_data(
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str
    ) -> pd.DataFrame:
        """Prepare data for interaction heatmap."""
        # Aggregate replications
        agg_df = results_df.groupby([param1, param2])[metric].mean().reset_index()
        
        # Pivot for heatmap
        pivot_df = agg_df.pivot(index=param2, columns=param1, values=metric)
        
        return pivot_df
    
    @staticmethod
    def calculate_optimal_region(
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str,
        threshold: float
    ) -> np.ndarray:
        """Calculate binary mask for optimal parameter region."""
        pivot_df = VisualizationHelper.prepare_heatmap_data(
            results_df, param1, param2, metric
        )
        
        return (pivot_df >= threshold).astype(int)


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

class ResultExporter:
    """Export sensitivity results in various formats."""
    
    @staticmethod
    def export_latex_table(
        summary_df: pd.DataFrame,
        output_file: str,
        caption: str = "Sensitivity Analysis Results"
    ):
        """Export summary table in LaTeX format."""
        latex_str = summary_df.to_latex(
            index=False,
            float_format="%.3f",
            caption=caption,
            label="tab:sensitivity"
        )
        
        with open(output_file, 'w') as f:
            f.write(latex_str)
    
    @staticmethod
    def export_publication_dataset(
        results_dfs: Dict[str, pd.DataFrame],
        output_dir: str
    ):
        """Export complete dataset for publication reproducibility."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export each experiment
        for name, df in results_dfs.items():
            # CSV
            df.to_csv(output_path / f"{name}.csv", index=False)
            
            # Statistical summary
            summary = df.describe()
            summary.to_csv(output_path / f"{name}_summary.csv")
        
        # Export metadata
        metadata = {
            'experiments': list(results_dfs.keys()),
            'total_runs': sum(len(df) for df in results_dfs.values()),
            'export_date': datetime.now().isoformat(),
            'metrics': list(results_dfs[list(results_dfs.keys())[0]].columns)
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_sensitivity_results(
    directory: str,
    parameter_name: str
) -> pd.DataFrame:
    """Load saved sensitivity results."""
    path = Path(directory)
    
    # Try pickle first (full data)
    pkl_file = path / f"{parameter_name}_results.pkl"
    if pkl_file.exists():
        return pd.read_pickle(pkl_file)
    
    # Fall back to CSV
    csv_file = path / f"{parameter_name}_results.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    
    raise FileNotFoundError(f"No results found for {parameter_name} in {directory}")


def quick_sensitivity_report(results_df: pd.DataFrame, metric: str = 'fitness'):
    """Generate quick text report of sensitivity analysis."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS QUICK REPORT")
    print("=" * 70)
    
    # Overall statistics
    print(f"\nMetric: {metric}")
    print(f"  Mean: {results_df[metric].mean():.3f}")
    print(f"  Std Dev: {results_df[metric].std():.3f}")
    print(f"  Range: [{results_df[metric].min():.3f}, {results_df[metric].max():.3f}]")
    print(f"  CV: {results_df[metric].std() / results_df[metric].mean():.3f}")
    
    # By parameter value
    if 'parameter_value' in results_df.columns:
        print("\nBy Parameter Value:")
        grouped = results_df.groupby('parameter_value')[metric].agg(['mean', 'std', 'count'])
        print(grouped)
        
        # Sensitivity
        param_range = results_df['parameter_value'].max() - results_df['parameter_value'].min()
        metric_range = grouped['mean'].max() - grouped['mean'].min()
        sensitivity = metric_range / param_range if param_range > 0 else 0
        
        print(f"\nSensitivity Coefficient: {sensitivity:.3f}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("Sensitivity Analysis Utility Module")
    print("Import this module to use batch execution and analysis tools")
