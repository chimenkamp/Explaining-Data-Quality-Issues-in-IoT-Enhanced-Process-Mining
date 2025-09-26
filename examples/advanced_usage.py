"""Advanced usage examples for the IoT Data Quality Pipeline"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import pipeline components
from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.explainability.insights import InsightGenerator
from src.explainability.explanations import ExplanationGenerator
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.export_utils import ResultsExporter
from src.utils.visualization_utils import VisualizationHelper


def example_1_custom_environment():
    """Example 1: Creating a custom IoT environment with specific quality issues"""

    print("=== Example 1: Custom Environment with Targeted Quality Issues ===")

    # Create environment with specific configuration
    env = IoTEnvironment(
        name="Precision Manufacturing Line",
        duration_hours=4,
        num_machines=6
    )

    # Add multiple welding stations with known issues
    for i in range(3):
        env.add_welding_station()

    # Add inspection stations
    for i in range(2):
        env.add_inspection_station()

    # Add packaging station
    env.add_packaging_station()

    print(f"Created environment with {len(env.machines)} machines")

    # Generate data
    data = env.generate_data()
    print(f"Generated {len(data['raw_data'])} sensor readings")
    print(f"Time span: {data['raw_data']['timestamp'].min()} to {data['raw_data']['timestamp'].max()}")

    # Analyze sensor coverage
    sensor_summary = data['raw_data'].groupby('sensor_id').agg({
        'value': ['count', 'mean', 'std'],
        'timestamp': ['min', 'max']
    }).round(2)

    print("\nSensor Summary:")
    print(sensor_summary.head())

    return env, data


def example_2_performance_monitoring():
    """Example 2: Pipeline execution with performance monitoring"""

    print("\n=== Example 2: Performance Monitoring ===")

    # Create environment and data
    env = IoTEnvironment(name="Performance Test", duration_hours=1, num_machines=4)
    env.add_welding_station()
    env.add_welding_station()
    env.add_inspection_station()
    env.add_packaging_station()

    data = env.generate_data()

    # Initialize performance metrics
    perf_metrics = PerformanceMetrics()

    # Create enhanced pipeline manager with performance tracking
    pipeline = PipelineManager()

    # Override pipeline run method to include performance tracking
    perf_metrics.start_stage_timing('total_pipeline')

    # Run individual stages with timing
    perf_metrics.start_stage_timing('quality_detection')
    detected_issues = pipeline.quality_detector.detect_all_issues(data['raw_data'], env)
    perf_metrics.end_stage_timing('quality_detection', len(data['raw_data']))

    perf_metrics.start_stage_timing('preprocessing')
    preprocessed = pipeline.preprocessor.preprocess(data['raw_data'])
    perf_metrics.end_stage_timing('preprocessing', len(preprocessed))

    perf_metrics.start_stage_timing('event_abstraction')
    events = pipeline.event_abstractor.abstract_events(preprocessed)
    perf_metrics.end_stage_timing('event_abstraction', len(events))

    perf_metrics.start_stage_timing('case_correlation')
    cases = pipeline.case_correlator.correlate_cases(events)
    perf_metrics.end_stage_timing('case_correlation', len(cases))

    perf_metrics.start_stage_timing('process_mining')
    model = pipeline.process_miner.discover_process(cases)
    perf_metrics.end_stage_timing('process_mining', len(cases))

    perf_metrics.end_stage_timing('total_pipeline', len(data['raw_data']))

    # Record quality detection metrics
    perf_metrics.record_quality_detection_metrics(
        'quality_detection', detected_issues, len(data['raw_data'])
    )

    # Get performance summary
    summary = perf_metrics.get_performance_summary()

    print(f"\nPerformance Summary:")
    print(f"Total pipeline time: {summary['total_pipeline_time']:.2f} seconds")
    print(f"Peak memory usage: {summary['peak_memory_usage_mb']:.1f} MB")

    print("\nStage Performance:")
    for stage, perf in summary['stage_performance'].items():
        print(f"  {stage}: {perf['time_seconds']:.2f}s ({perf['time_percentage']:.1f}%)")
        if 'throughput_records_per_sec' in perf:
            print(f"    Throughput: {perf['throughput_records_per_sec']:.0f} records/sec")

    print(f"\nQuality Detection: {summary['quality_detection_summary']['total_issues_detected']} issues")
    print(f"High confidence ratio: {summary['quality_detection_summary']['high_confidence_ratio']:.1%}")

    return summary


def example_3_advanced_insights():
    """Example 3: Advanced insight generation and explanation"""

    print("\n=== Example 3: Advanced Insights and Explanations ===")

    # Create environment and run pipeline
    env = IoTEnvironment(name="Insight Analysis", duration_hours=2, num_machines=3)
    env.add_welding_station()
    env.add_inspection_station()
    env.add_packaging_station()

    data = env.generate_data()
    pipeline = PipelineManager()
    results = pipeline.run(data, env)

    # Generate advanced insights
    insight_generator = InsightGenerator()
    insights = insight_generator.generate_insights(results)

    print(f"\nGenerated {len(insights)} insights:")

    # Display top insights with detailed analysis
    for i, insight in enumerate(insights[:3], 1):
        print(f"\n--- Insight {i} ---")
        print(f"Type: {insight['type']}")
        print(f"Category: {insight['category']}")
        print(f"Message: {insight['message']}")
        print(f"Confidence: {insight['confidence']:.2f}")
        print(f"Actionable: {insight['actionable']}")
        print(f"Importance Score: {insight.get('importance_score', 0):.2f}")

        if 'recommendations' in insight:
            print(f"Recommendations:")
            for rec in insight['recommendations'][:2]:
                print(f"  • {rec}")

    # Generate detailed explanations for high-priority issues
    explanation_generator = ExplanationGenerator()

    high_priority_issues = [
        issue for issue in results['quality_issues']
        if issue.get('confidence', 0) > 0.6
    ]

    print(f"\n=== Detailed Explanations for {len(high_priority_issues)} High-Priority Issues ===")

    for i, issue in enumerate(high_priority_issues[:2], 1):
        print(f"\n--- Explanation {i} ---")

        explanation = explanation_generator.generate_explanation(issue, results)

        print(f"Issue: {explanation['issue_summary']}")
        print(f"Root Cause: {explanation['root_cause_analysis']['primary_cause']}")

        impact = explanation['impact_analysis']
        print(f"Business Impact: {impact['business_impact']}")
        print(f"Technical Impact: {impact['technical_impact']}")

        if impact['immediate_impacts']:
            print("Immediate Impacts:")
            for impact_item in impact['immediate_impacts'][:2]:
                print(f"  • {impact_item}")

        strategy = explanation['remediation_strategy']
        print(f"Implementation Priority: {strategy['implementation_priority']}")

        if strategy['immediate_actions']:
            print("Immediate Actions:")
            for action in strategy['immediate_actions'][:2]:
                print(f"  • {action}")

    return insights, high_priority_issues


def example_4_custom_visualization():
    """Example 4: Advanced visualization and reporting"""

    print("\n=== Example 4: Advanced Visualization and Reporting ===")

    # Create environment and run pipeline
    env = IoTEnvironment(name="Visualization Demo", duration_hours=1, num_machines=4)
    env.add_welding_station()
    env.add_welding_station()
    env.add_inspection_station()
    env.add_packaging_station()

    data = env.generate_data()
    pipeline = PipelineManager()
    results = pipeline.run(data, env)

    # Create advanced visualizations
    viz_helper = VisualizationHelper()

    # Create sensor heatmap
    print("Creating sensor activity heatmap...")
    heatmap_fig = viz_helper.create_sensor_heatmap(
        results['raw_data'],
        results['quality_issues']
    )

    # Create issue correlation network
    print("Creating quality issue correlation network...")
    network_fig = viz_helper.create_issue_correlation_network(
        results['quality_issues']
    )

    # Create quality evolution chart
    print("Creating quality evolution chart...")
    evolution_fig = viz_helper.create_quality_evolution_chart(results)

    # Create confidence distribution
    print("Creating confidence distribution...")
    confidence_fig = viz_helper.create_confidence_distribution(
        results['quality_issues']
    )

    # Export results in multiple formats
    exporter = ResultsExporter()

    print("\nExporting results...")

    # Export to JSON
    exporter.export_to_json(results, 'advanced_example_results.json')

    # Export to Excel
    exporter.export_to_excel(results, 'advanced_example_results.xlsx')

    # Export quality report
    exporter.export_quality_report(results, 'advanced_quality_report.json')

    print("Results exported successfully!")

    # Display summary statistics
    print(f"\n=== Visualization Summary ===")
    print(f"Sensors analyzed: {results['raw_data']['sensor_id'].nunique()}")
    print(f"Quality issues found: {len(results['quality_issues'])}")
    print(f"Process instances: {len(results['process_instances'])}")

    if results['process_model'] and 'metrics' in results['process_model']:
        metrics = results['process_model']['metrics']
        print(f"Process model fitness: {metrics.get('fitness', 0):.3f}")
        print(f"Process model complexity: {metrics.get('complexity', 0):.3f}")

    return results


def example_5_batch_analysis():
    """Example 5: Batch analysis of multiple environments"""

    print("\n=== Example 5: Batch Analysis ===")

    batch_results = []

    # Analyze different environment configurations
    configs = [
        {"name": "Small_Line", "machines": 2, "hours": 1},
        {"name": "Medium_Line", "machines": 4, "hours": 2},
        {"name": "Large_Line", "machines": 6, "hours": 3}
    ]

    for config in configs:
        print(f"\nAnalyzing {config['name']}...")

        # Create environment
        env = IoTEnvironment(
            name=config['name'],
            duration_hours=config['hours'],
            num_machines=config['machines']
        )

        # Add machines based on size
        num_machines = config['machines']
        for i in range(num_machines // 2):
            env.add_welding_station()
        for i in range(max(1, num_machines // 3)):
            env.add_inspection_station()
        for i in range(max(1, num_machines // 4)):
            env.add_packaging_station()

        # Generate data and run pipeline
        data = env.generate_data()
        pipeline = PipelineManager()
        results = pipeline.run(data, env)

        # Collect metrics
        config_results = {
            'config': config,
            'raw_data_size': len(results['raw_data']),
            'quality_issues_count': len(results['quality_issues']),
            'process_instances_count': len(results['process_instances']),
            'unique_sensors': results['raw_data']['sensor_id'].nunique(),
            'time_span_hours': (
                                       results['raw_data']['timestamp'].max() -
                                       results['raw_data']['timestamp'].min()
                               ).total_seconds() / 3600
        }

        if results['process_model'] and 'metrics' in results['process_model']:
            metrics = results['process_model']['metrics']
            config_results.update({
                'model_fitness': metrics.get('fitness', 0),
                'model_precision': metrics.get('precision', 0),
                'model_complexity': metrics.get('complexity', 0)
            })

        batch_results.append(config_results)

        print(f"  Data size: {config_results['raw_data_size']} readings")
        print(f"  Quality issues: {config_results['quality_issues_count']}")
        print(f"  Fitness: {config_results.get('model_fitness', 0):.3f}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(batch_results)

    print(f"\n=== Batch Analysis Results ===")
    print(comparison_df[['config', 'raw_data_size', 'quality_issues_count', 'model_fitness']].to_string(index=False))

    # Analyze trends
    print(f"\n=== Trends ===")
    print(
        f"Quality issues vs data size correlation: {comparison_df['quality_issues_count'].corr(comparison_df['raw_data_size']):.3f}")

    if 'model_fitness' in comparison_df.columns:
        print(
            f"Model fitness vs complexity correlation: {comparison_df['model_fitness'].corr(comparison_df['model_complexity']):.3f}")

    return comparison_df


def main():
    """Run all advanced examples"""

    logging.basicConfig(level=logging.INFO)

    print("🚀 Running Advanced IoT Data Quality Pipeline Examples")
    print("=" * 80)

    try:
        # Run examples
        env, data = example_1_custom_environment()
        perf_summary = example_2_performance_monitoring()
        insights, issues = example_3_advanced_insights()
        viz_results = example_4_custom_visualization()
        batch_df = example_5_batch_analysis()

        print("\n" + "=" * 80)
        print("✅ All advanced examples completed successfully!")
        print("📁 Check output files for detailed results")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        logging.error(f"Advanced examples failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()