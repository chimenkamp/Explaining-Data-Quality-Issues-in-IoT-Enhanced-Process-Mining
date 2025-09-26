import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime


class ResultsExporter:
    """Exports pipeline results in various formats"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def export_to_json(self, results: Dict[str, Any], filename: str):
        """Export results to JSON format"""

        # Create a serializable version of results
        serializable_results = self._make_json_serializable(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        self.logger.info(f"Results exported to JSON: {filename}")

    def export_to_excel(self, results: Dict[str, Any], filename: str):
        """Export results to Excel format with multiple sheets"""

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            # Quality Issues sheet
            quality_issues = results.get('quality_issues', [])
            if quality_issues:
                quality_df = pd.json_normalize(quality_issues)
                quality_df.to_excel(writer, sheet_name='Quality_Issues', index=False)

            # Process Instances sheet
            process_instances = results.get('process_instances', pd.DataFrame())
            if not process_instances.empty:
                process_instances.to_excel(writer, sheet_name='Process_Instances', index=False)

            # Structured Events sheet
            structured_events = results.get('structured_events', pd.DataFrame())
            if not structured_events.empty:
                # Limit to first 10000 rows to avoid Excel limitations
                events_to_export = structured_events.head(10000)
                events_to_export.to_excel(writer, sheet_name='Structured_Events', index=False)

            # Process Model Metrics sheet
            process_model = results.get('process_model', {})
            if process_model and 'metrics' in process_model:
                metrics_df = pd.DataFrame([process_model['metrics']])
                metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)

        self.logger.info(f"Results exported to Excel: {filename}")

    def export_quality_report(self, results: Dict[str, Any], filename: str):
        """Export a comprehensive quality report"""

        quality_issues = results.get('quality_issues', [])
        process_model = results.get('process_model', {})

        report = {
            'report_generated_at': datetime.now().isoformat(),
            'summary': {
                'total_quality_issues': len(quality_issues),
                'issue_types': list(set([issue['type'] for issue in quality_issues])),
                'high_severity_issues': len([issue for issue in quality_issues if issue.get('severity') == 'high']),
                'high_confidence_issues': len([issue for issue in quality_issues if issue.get('confidence', 0) > 0.8])
            },
            'detailed_issues': quality_issues,
            'process_model_analysis': process_model.get('quality_analysis', {}),
            'recommendations': self._generate_consolidated_recommendations(quality_issues)
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Quality report exported: {filename}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""

        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def _generate_consolidated_recommendations(self, quality_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate consolidated recommendations from all quality issues"""

        all_recommendations = []

        for issue in quality_issues:
            if 'recommendations' in issue:
                all_recommendations.extend(issue['recommendations'])

        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()

        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)

        return unique_recommendations