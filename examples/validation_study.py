"""
Comprehensive validation study for the IoT Data Quality Pipeline
"""
import logging
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.synthetic_environment.iot_environment import IoTEnvironment
from src.synthetic_environment.quality_injector import QualityIssueInjector
from src.pipeline.pipeline_manager import PipelineManager
from src.evaluation.metrics import PipelineEvaluator
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.export_utils import ResultsExporter


class ValidationStudy:
    """Comprehensive validation study"""

    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def run_accuracy_study(self) -> Dict[str, Any]:
        """Study quality detection accuracy with known ground truth"""

        self.logger.info("Running quality detection accuracy study...")

        results = []
        injector = QualityIssueInjector(seed=42)  # Fixed seed for reproducibility
        evaluator = PipelineEvaluator()

        # Test different scenarios
        scenarios = [
            {'name': 'clean_data', 'inject_issues': False},
            {'name': 'c1_sampling_issues', 'inject_c1': True},
            {'name': 'c3_noise_issues', 'inject_c3': True},
            {'name': 'mixed_issues', 'inject_c1': True, 'inject_c3': True}
        ]

        for scenario in scenarios:
            self.logger.info(f"Testing scenario: {scenario['name']}")

            # Create clean environment
            env = IoTEnvironment(name=f"Validation_{scenario['name']}", duration_hours=1, num_machines=3)
            env.add_welding_station()
            env.add_inspection_station()
            env.add_packaging_station()

            # Generate clean data
            data = env.generate_data()
            raw_data = data['raw_data'].copy()

            # Inject known issues
            ground_truth_issues = []

            if scenario.get('inject_c1', False):
                # Inject C1 issues by removing timestamps for each sensor
                for sensor_id in raw_data['sensor_id'].unique():
                    sensor_mask = raw_data['sensor_id'] == sensor_id
                    sensor_timestamps = raw_data[sensor_mask]['timestamp'].tolist()

                    reduced_timestamps = injector.inject_c1_inadequate_sampling(
                        sensor_timestamps, 'medium'
                    )

                    # Keep only rows with timestamps that weren't removed
                    raw_data = raw_data[
                        ~sensor_mask | raw_data['timestamp'].isin(reduced_timestamps)
                        ]

                    ground_truth_issues.append({
                        'type': 'C1_inadequate_sampling',
                        'sensor_id': sensor_id,
                        'severity': 'medium'
                    })

            if scenario.get('inject_c3', False):
                # Inject C3 noise issues
                for sensor_id in raw_data['sensor_id'].unique():
                    sensor_mask = raw_data['sensor_id'] == sensor_id
                    sensor_values = raw_data.loc[sensor_mask, 'value'].values

                    noisy_values = injector.inject_c3_sensor_noise(sensor_values, 'medium')
                    raw_data.loc[sensor_mask, 'value'] = noisy_values

                    ground_truth_issues.append({
                        'type': 'C3_sensor_noise',
                        'sensor_id': sensor_id,
                        'severity': 'medium'
                    })

            # Run pipeline
            pipeline = PipelineManager()
            pipeline_results = pipeline.run(
                {'raw_data': raw_data, 'process_instances': data['process_instances'], 'environment': env}, env)

            # Evaluate accuracy
            detected_issues = pipeline_results['quality_issues']
            accuracy_metrics = evaluator.evaluate_quality_detection_accuracy(
                detected_issues, ground_truth_issues
            )

            # Evaluate process model quality
            model_quality = evaluator.evaluate_process_model_quality(
                pipeline_results['process_model']
            )

            scenario_results = {
                'scenario': scenario['name'],
                'ground_truth_issues': len(ground_truth_issues),
                'detected_issues': len(detected_issues),
                'accuracy_metrics': accuracy_metrics,
                'model_quality': model_quality
            }

            results.append(scenario_results)

            self.logger.info(
                f"Scenario {scenario['name']}: Precision={accuracy_metrics['precision']:.3f}, Recall={accuracy_metrics['recall']:.3f}")

        # Save results
        results_file = self.output_dir / 'accuracy_study_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def run_scalability_study(self) -> Dict[str, Any]:
        """Study scalability with different data sizes"""

        self.logger.info("Running scalability study...")

        results = []

        # Test different scales
        scales = [
            {'machines': 2, 'hours': 0.5, 'name': 'small'},
            {'machines': 4, 'hours': 1, 'name': 'medium'},
            {'machines': 6, 'hours': 2, 'name': 'large'},
            {'machines': 8, 'hours': 3, 'name': 'xlarge'}
        ]

        for scale in scales:
            self.logger.info(f"Testing scale: {scale['name']}")

            # Create environment
            env = IoTEnvironment(
                name=f"Scale_{scale['name']}",
                duration_hours=scale['hours'],
                num_machines=scale['machines']
            )

            # Add machines proportionally
            num_welding = max(1, scale['machines'] // 2)
            num_inspection = max(1, scale['machines'] // 3)
            num_packaging = max(1, scale['machines'] // 4)

            for _ in range(num_welding):
                env.add_welding_station()
            for _ in range(num_inspection):
                env.add_inspection_station()
            for _ in range(num_packaging):
                env.add_packaging_station()

            # Generate data and measure performance
            perf_metrics = PerformanceMetrics()

            perf_metrics.start_stage_timing('data_generation')
            data = env.generate_data()
            perf_metrics.end_stage_timing('data_generation', len(data['raw_data']))

            perf_metrics.start_stage_timing('pipeline_execution')
            pipeline = PipelineManager()
            pipeline_results = pipeline.run(data, env)
            perf_metrics.end_stage_timing('pipeline_execution', len(data['raw_data']))

            performance_summary = perf_metrics.get_performance_summary()

            scale_results = {
                'scale': scale['name'],
                'machines': scale['machines'],
                'hours': scale['hours'],
                'raw_data_size': len(data['raw_data']),
                'process_instances': len(pipeline_results['process_instances']),
                'quality_issues': len(pipeline_results['quality_issues']),
                'total_time_seconds': performance_summary['total_pipeline_time'],
                'peak_memory_mb': performance_summary['peak_memory_usage_mb'],
                'throughput_records_per_second': len(data['raw_data']) / performance_summary['total_pipeline_time'] if
                performance_summary['total_pipeline_time'] > 0 else 0
            }

            results.append(scale_results)

            self.logger.info(
                f"Scale {scale['name']}: {scale_results['raw_data_size']} records in {scale_results['total_time_seconds']:.2f}s")

        # Save results
        results_file = self.output_dir / 'scalability_study_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    def create_validation_report(self, accuracy_results: Dict[str, Any],
                                 scalability_results: Dict[str, Any]):
        """Create comprehensive validation report"""

        report = {
            'validation_study_summary': {
                'study_date': pd.Timestamp.now().isoformat(),
                'total_scenarios_tested': len(accuracy_results),
                'scalability_configs_tested': len(scalability_results)
            },
            'accuracy_study': {
                'average_precision': np.mean([r['accuracy_metrics']['precision'] for r in accuracy_results]),
                'average_recall': np.mean([r['accuracy_metrics']['recall'] for r in accuracy_results]),
                'average_f1': np.mean([r['accuracy_metrics']['f1_score'] for r in accuracy_results])
            },
            'scalability_study': {
                'max_throughput_records_per_second': max(
                    [r['throughput_records_per_second'] for r in scalability_results]),
                'linear_scalability_coefficient': self._calculate_scalability_coefficient(scalability_results)
            },
            'detailed_results': {
                'accuracy_results': accuracy_results,
                'scalability_results': scalability_results
            }
        }

        # Save report
        report_file = self.output_dir / 'validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create visualizations
        self._create_validation_plots(accuracy_results, scalability_results)

        self.logger.info(f"Validation report saved to {report_file}")

        return report

    def _calculate_scalability_coefficient(self, scalability_results: List[Dict[str, Any]]) -> float:
        """Calculate linear scalability coefficient"""

        data_sizes = [r['raw_data_size'] for r in scalability_results]
        throughputs = [r['throughput_records_per_second'] for r in scalability_results]

        # Linear regression coefficient (should be close to 1 for perfect linear scalability)
        correlation = np.corrcoef(data_sizes, throughputs)[0, 1]

        return correlation

    def _create_validation_plots(self, accuracy_results: List[Dict[str, Any]],
                                 scalability_results: List[Dict[str, Any]]):
        """Create validation study plots"""

        # Accuracy plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Precision/Recall by scenario
        scenarios = [r['scenario'] for r in accuracy_results]
        precisions = [r['accuracy_metrics']['precision'] for r in accuracy_results]
        recalls = [r['accuracy_metrics']['recall'] for r in accuracy_results]

        ax1.bar(scenarios, precisions, alpha=0.7, label='Precision')
        ax1.bar(scenarios, recalls, alpha=0.7, label='Recall')
        ax1.set_title('Quality Detection Accuracy by Scenario')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        # Model quality by scenario
        model_fitness = [r['model_quality']['fitness'] for r in accuracy_results]
        model_complexity = [r['model_quality']['complexity'] for r in accuracy_results]

        ax2.bar(scenarios, model_fitness, alpha=0.7, label='Fitness')
        ax2.bar(scenarios, [1 - c for c in model_complexity], alpha=0.7, label='1-Complexity')
        ax2.set_title('Process Model Quality by Scenario')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)

        # Scalability - throughput vs data size
        data_sizes = [r['raw_data_size'] for r in scalability_results]
        throughputs = [r['throughput_records_per_second'] for r in scalability_results]

        ax3.scatter(data_sizes, throughputs)
        ax3.plot(data_sizes, throughputs, 'r--', alpha=0.5)
        ax3.set_title('Throughput vs Data Size')
        ax3.set_xlabel('Raw Data Size (records)')
        ax3.set_ylabel('Throughput (records/sec)')

        # Memory usage vs data size
        memory_usage = [r['peak_memory_mb'] for r in scalability_results]

        ax4.scatter(data_sizes, memory_usage)
        ax4.plot(data_sizes, memory_usage, 'g--', alpha=0.5)
        ax4.set_title('Memory Usage vs Data Size')
        ax4.set_xlabel('Raw Data Size (records)')
        ax4.set_ylabel('Peak Memory (MB)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'validation_study_plots.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Run comprehensive validation study"""

    logging.basicConfig(level=logging.INFO)

    print("🔬 Starting Comprehensive Validation Study")
    print("=" * 80)

    study = ValidationStudy()

    try:
        # Run accuracy study
        print("\n📊 Running Quality Detection Accuracy Study...")
        accuracy_results = study.run_accuracy_study()

        # Run scalability study
        print("\n📈 Running Scalability Study...")
        scalability_results = study.run_scalability_study()

        # Create comprehensive report
        print("\n📝 Creating Validation Report...")
        report = study.create_validation_report(accuracy_results, scalability_results)

        print("\n" + "=" * 80)
        print("✅ VALIDATION STUDY COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: {study.output_dir}")
        print("=" * 80)

        # Print summary
        print(f"\nACCURACY SUMMARY:")
        print(f"Average Precision: {report['accuracy_study']['average_precision']:.3f}")
        print(f"Average Recall: {report['accuracy_study']['average_recall']:.3f}")
        print(f"Average F1-Score: {report['accuracy_study']['average_f1']:.3f}")

        print(f"\nSCALABILITY SUMMARY:")
        print(f"Max Throughput: {report['scalability_study']['max_throughput_records_per_second']:.0f} records/sec")
        print(f"Scalability Coefficient: {report['scalability_study']['linear_scalability_coefficient']:.3f}")

    except Exception as e:
        print(f"\n❌ Validation study failed: {e}")
        logging.error(f"Validation study failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()