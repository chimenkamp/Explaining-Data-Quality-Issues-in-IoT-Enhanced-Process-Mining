import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.synthetic_environment.iot_environment import IoTEnvironment
from src.pipeline.pipeline_manager import PipelineManager
from src.explainability.insights import InsightGenerator
from src.explainability.explanations import ExplanationGenerator


def setup_demo_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)


def create_demo_environment():
    """Create a comprehensive demo IoT environment"""
    logger = logging.getLogger(__name__)

    logger.info("Creating synthetic IoT manufacturing environment with ground truth...")
    env = IoTEnvironment(
        name="Smart Manufacturing Line Demo",
        duration_hours=1,
        num_cases=20
    )

    return env


def run_complete_pipeline_demo(env):
    """Run the complete pipeline demonstration"""
    logger = logging.getLogger(__name__)

    logger.info("Generating synthetic data with quality issues...")
    data = env.generate_data()

    logger.info(f"Generated {len(data['raw_data'])} sensor readings")
    logger.info(f"Created {len(data['process_instances'])} process instances")

    print("\n" + "=" * 80)
    print("SYNTHETIC DATA OVERVIEW")
    print("=" * 80)

    raw_data = data['raw_data']
    print(f"Total sensor readings: {len(raw_data)}")
    print(f"Sensors involved: {raw_data['sensor_id'].nunique()}")
    print(f"Time span: {raw_data['timestamp'].min()} to {raw_data['timestamp'].max()}")

    if 'activity' in raw_data.columns:
        print(f"Activities detected: {raw_data['activity'].nunique()}")

    quality_flags = [qf for qf in raw_data['quality_flags'] if qf]
    print(f"Readings with quality flags: {len(quality_flags)}")

    logger.info("Initializing pipeline manager...")
    pipeline = PipelineManager()

    logger.info("Running complete IoT data quality pipeline...")
    results = pipeline.run(data, env)

    return results


def analyze_results(results):
    """Analyze and display pipeline results"""
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 80)
    print("PIPELINE RESULTS ANALYSIS")
    print("=" * 80)

    quality_issues = results['quality_issues']
    print(f"\n📊 QUALITY ISSUES DETECTED: {len(quality_issues)}")

    if quality_issues:
        issue_types = {}
        for issue in quality_issues:
            issue_type = issue['type']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)

        for issue_type, issues in issue_types.items():
            print(f"\n  {issue_type}: {len(issues)} instances")
            for issue in issues[:2]:
                print(f"    • {issue['description']} (Confidence: {issue.get('confidence', 0):.2f})")
            if len(issues) > 2:
                print(f"    • ... and {len(issues) - 2} more")

    process_model = results['process_model']
    if process_model and 'metrics' in process_model:
        metrics = process_model['metrics']
        print(f"\n🔄 PROCESS MODEL METRICS:")
        print(f"  Fitness Score: {metrics.get('fitness', 0):.3f}")
        print(f"  Precision Score: {metrics.get('precision', 0):.3f}")
        print(f"  Quality-Weighted Fitness: {metrics.get('quality_weighted_fitness', 0):.3f}")

        model_data = process_model.get('model', {})
        print(f"  Activities Discovered: {len(model_data.get('activities', []))}")

    case_instances = results['process_instances']
    if not case_instances.empty:
        print(f"\n📋 PROCESS INSTANCES: {len(case_instances)} cases")
        print(f"  Average case duration: {case_instances['duration'].mean():.1f} seconds")
        print(f"  Average events per case: {case_instances['num_events'].mean():.1f}")

        quality_scores = case_instances['case_quality_score']
        print(f"  Case quality scores: {quality_scores.mean():.2f} ± {quality_scores.std():.2f}")


def generate_insights_and_explanations(results):
    """Generate and display insights and explanations"""
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 80)
    print("INSIGHTS AND EXPLANATIONS")
    print("=" * 80)

    logger.info("Generating explainable insights...")
    insight_generator = InsightGenerator()
    insights = insight_generator.generate_insights(results)

    print(f"\n💡 GENERATED INSIGHTS: {len(insights)}")

    for i, insight in enumerate(insights[:5], 1):
        print(f"\n  {i}. {insight['message']}")
        print(f"     Confidence: {insight['confidence']:.2f} | Actionable: {insight['actionable']}")
        if insight.get('recommendations'):
            print(f"     Recommendations: {', '.join(insight['recommendations'][:2])}")

    explanation_generator = ExplanationGenerator()

    quality_issues = results['quality_issues']
    high_priority_issues = [
        issue for issue in quality_issues
        if issue.get('confidence', 0) > 0.7 and issue.get('severity') in ['high', 'medium']
    ]

    print(f"\n📝 DETAILED EXPLANATIONS: {len(high_priority_issues)} high-priority issues")

    for i, issue in enumerate(high_priority_issues[:2], 1):
        print(f"\n  --- EXPLANATION {i} ---")
        explanation = explanation_generator.generate_explanation(issue, results)

        print(f"  Issue: {explanation['issue_summary']}")
        print(f"  Technical: {explanation['technical_explanation'][:200]}...")

        root_cause = explanation['root_cause_analysis']
        print(f"  Root Cause: {root_cause['primary_cause']}")

        impact = explanation['impact_analysis']
        print(f"  Business Impact: {impact['business_impact']}")
        print(f"  Immediate Impacts: {', '.join(impact['immediate_impacts'][:2])}")

        strategy = explanation['remediation_strategy']
        print(f"  Priority: {strategy['implementation_priority']}")
        print(f"  Immediate Actions: {', '.join(strategy['immediate_actions'][:2])}")


def create_visualizations(results):
    """Create and save visualizations"""
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Get enhanced visualizations
    visualization = results.get('visualization', {})

    # Save Petri net with quality overlay
    if 'petri_net_quality_overlay' in visualization:
        petri_net_fig = visualization['petri_net_quality_overlay']
        petri_net_path = output_dir / 'petri_net_quality_overlay.html'
        petri_net_fig.write_html(str(petri_net_path))
        logger.info(f"Saved Petri net quality overlay to {petri_net_path}")
        print(f"✓ Petri net visualization saved: {petri_net_path}")

        # Also save as static image if possible
        try:
            petri_net_img_path = output_dir / 'petri_net_quality_overlay.png'
            petri_net_fig.write_image(str(petri_net_img_path), width=1200, height=800)
            logger.info(f"Saved Petri net as image to {petri_net_img_path}")
            print(f"✓ Petri net image saved: {petri_net_img_path}")
        except Exception as e:
            logger.warning(f"Could not save Petri net as image (kaleido may not be installed): {e}")

    # Save process model visualization
    if 'process_model' in visualization:
        process_model_fig = visualization['process_model']
        process_model_path = output_dir / 'process_model_quality.html'
        process_model_fig.write_html(str(process_model_path))
        logger.info(f"Saved process model visualization to {process_model_path}")
        print(f"✓ Process model saved: {process_model_path}")

    # Save quality dashboard
    if 'quality_dashboard' in visualization:
        dashboard_fig = visualization['quality_dashboard']
        dashboard_path = output_dir / 'quality_dashboard.html'
        dashboard_fig.write_html(str(dashboard_path))
        logger.info(f"Saved quality dashboard to {dashboard_path}")
        print(f"✓ Quality dashboard saved: {dashboard_path}")

    # Quality Issues Distribution (matplotlib)
    quality_issues = results['quality_issues']
    if quality_issues:
        issue_types = [issue['type'] for issue in quality_issues]
        issue_counts = pd.Series(issue_types).value_counts()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        issue_counts.plot(kind='bar', color='skyblue')
        plt.title('Quality Issues Distribution')
        plt.xlabel('Issue Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

        confidences = [issue.get('confidence', 0) for issue in quality_issues]
        severities = [issue.get('severity', 'medium') for issue in quality_issues]

        plt.subplot(1, 2, 2)
        severity_colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        colors = [severity_colors.get(sev, 'gray') for sev in severities]
        plt.scatter(confidences, range(len(confidences)), c=colors, alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Issue Index')
        plt.title('Issue Confidence by Severity')

        plt.tight_layout()
        plt.savefig(output_dir / 'quality_issues_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved quality issues visualization to {output_dir / 'quality_issues_analysis.png'}")

    # Process Model Metrics (matplotlib)
    process_model = results.get('process_model', {})
    if process_model and 'metrics' in process_model:
        metrics = process_model['metrics']

        plt.figure(figsize=(10, 6))

        metric_names = ['Fitness', 'Precision', 'Quality-Weighted\nFitness']
        metric_values = [
            metrics.get('fitness', 0),
            metrics.get('precision', 0),
            metrics.get('quality_weighted_fitness', 0)
        ]

        bars = plt.bar(metric_names, metric_values,
                       color=['#2E86AB', '#A23B72', '#C73E1D'])
        plt.title('Process Model Quality Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / 'process_model_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved process model metrics to {output_dir / 'process_model_metrics.png'}")


def save_results_summary(results):
    """Save a comprehensive results summary"""
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    summary_file = output_dir / 'pipeline_results_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("IoT DATA QUALITY PIPELINE - RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        quality_issues = results['quality_issues']
        f.write(f"QUALITY ISSUES DETECTED: {len(quality_issues)}\n")
        f.write("-" * 40 + "\n")

        issue_summary = {}
        for issue in quality_issues:
            issue_type = issue['type']
            if issue_type not in issue_summary:
                issue_summary[issue_type] = {'count': 0, 'avg_confidence': 0}
            issue_summary[issue_type]['count'] += 1
            issue_summary[issue_type]['avg_confidence'] += issue.get('confidence', 0)

        for issue_type, data in issue_summary.items():
            avg_conf = data['avg_confidence'] / data['count']
            f.write(f"{issue_type}: {data['count']} issues (avg confidence: {avg_conf:.2f})\n")

        f.write("\n")

        process_model = results.get('process_model', {})
        if process_model and 'metrics' in process_model:
            metrics = process_model['metrics']
            f.write("PROCESS MODEL METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Fitness: {metrics.get('fitness', 0):.3f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.3f}\n")
            f.write(f"Quality-Weighted Fitness: {metrics.get('quality_weighted_fitness', 0):.3f}\n")

        case_instances = results['process_instances']
        if not case_instances.empty:
            f.write(f"\nPROCESS INSTANCES: {len(case_instances)} cases\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average duration: {case_instances['duration'].mean():.1f} seconds\n")
            f.write(f"Average events per case: {case_instances['num_events'].mean():.1f}\n")
            f.write(f"Average quality score: {case_instances['case_quality_score'].mean():.2f}\n")

    print(f"\n📄 Results summary saved to: {summary_file}")


def main():
    """Main demo function"""
    logger = setup_demo_logging()

    print("🚀 Starting IoT Data Quality Pipeline Demo")
    print("=" * 80)

    try:
        env = create_demo_environment()
        results = run_complete_pipeline_demo(env)
        analyze_results(results)
        generate_insights_and_explanations(results)
        create_visualizations(results)
        save_results_summary(results)

        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("📁 Check the 'demo_output' folder for:")
        print("   • petri_net_quality_overlay.html - Interactive Petri net with quality insights")
        print("   • process_model_quality.html - Process model visualization")
        print("   • quality_dashboard.html - Quality metrics dashboard")
        print("   • quality_issues_analysis.png - Issue distribution charts")
        print("   • process_model_metrics.png - Model quality metrics")
        print("   • pipeline_results_summary.txt - Text summary")
        print("📋 Check 'demo.log' for detailed execution logs")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Check demo.log for detailed error information")


if __name__ == "__main__":
    main()