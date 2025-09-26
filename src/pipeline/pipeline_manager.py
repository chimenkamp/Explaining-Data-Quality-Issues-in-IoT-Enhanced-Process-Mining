import logging
from typing import Dict, Any, List
import pandas as pd

from .preprocessing import Preprocessor
from .event_abstraction import EventAbstractor
from .case_correlation import CaseCorrelator
from .process_mining import ProcessMiner
from .visualization import EnhancedVisualizer

from ..data_quality.detectors import QualityIssueDetector
from ..data_quality.classifiers import QualityClassifier
from ..data_quality.propagation import QualityPropagator


class PipelineManager:
    """Manages the entire IoT data quality pipeline with conformance-based detection"""

    def __init__(self, conformance_threshold: float = 0.7):
        self.logger = logging.getLogger(__name__)
        self.conformance_threshold = conformance_threshold

        # Initialize pipeline components
        self.preprocessor = Preprocessor()
        self.event_abstractor = EventAbstractor()
        self.case_correlator = CaseCorrelator()
        self.process_miner = ProcessMiner(conformance_threshold=conformance_threshold)
        self.visualizer = EnhancedVisualizer()

        # Initialize quality components
        self.quality_detector = QualityIssueDetector()
        self.quality_classifier = QualityClassifier()
        self.quality_propagator = QualityPropagator()

        self.pipeline_results = {}

    def run(self, data: Dict[str, Any], environment) -> Dict[str, Any]:
        """Run the complete pipeline with conformance-based quality detection"""

        raw_data = data['raw_data']
        self.logger.info(f"Starting pipeline with {len(raw_data)} raw readings")

        # ========== STAGE 1: Initial Quality Detection ==========
        self.logger.info("Stage 1: Detecting quality issues in raw data")
        detected_issues = self.quality_detector.detect_all_issues(raw_data, environment)
        classified_issues = self.quality_classifier.classify_issues(detected_issues, raw_data)

        self.logger.info(f"Detected {len(classified_issues)} initial quality issues")

        # ========== STAGE 2: Preprocessing with Quality Propagation ==========
        self.logger.info("Stage 2: Preprocessing")
        preprocessed_data = self.preprocessor.preprocess(raw_data)

        preprocessing_propagation = self.quality_propagator.propagate_issues(
            classified_issues, 'preprocessing', preprocessed_data
        )

        # ========== STAGE 3: Event Abstraction with Quality Propagation ==========
        self.logger.info("Stage 3: Event abstraction")
        structured_events = self.event_abstractor.abstract_events(preprocessed_data)

        event_propagation = self.quality_propagator.propagate_issues(
            preprocessing_propagation['propagated_issues'],
            'event_abstraction', structured_events
        )

        # ========== STAGE 4: Case Correlation with Quality Propagation ==========
        self.logger.info("Stage 4: Case correlation")
        process_instances = self.case_correlator.correlate_cases(structured_events)

        case_propagation = self.quality_propagator.propagate_issues(
            event_propagation['propagated_issues'],
            'case_correlation', process_instances
        )

        # ========== STAGE 5: Process Mining with Conformance Checking ==========
        self.logger.info("Stage 5: Process mining with conformance checking")
        process_model = self.process_miner.discover_process(process_instances)

        mining_propagation = self.quality_propagator.propagate_issues(
            case_propagation['propagated_issues'],
            'process_mining', process_model
        )

        # ========== STAGE 6: Conformance-Based Quality Detection ==========
        self.logger.info("Stage 6: Conformance-based quality analysis")

        conformance_issues = process_model.get('conformance_issues', [])
        quality_analysis = process_model.get('quality_analysis', {})

        if quality_analysis.get('has_conformance_issues', False):
            self.logger.warning(f"Conformance issues detected: {len(conformance_issues)}")

            # Trigger backtracking for conformance issues
            backtracking_results = self._perform_conformance_backtracking(
                conformance_issues,
                quality_analysis,
                {
                    'raw_data': raw_data,
                    'preprocessed_data': preprocessed_data,
                    'structured_events': structured_events,
                    'process_instances': process_instances,
                    'initial_quality_issues': classified_issues,
                    'propagated_issues': mining_propagation['propagated_issues']
                }
            )

            # Integrate backtracking results into quality issues
            integrated_issues = self._integrate_conformance_findings(
                classified_issues,
                conformance_issues,
                backtracking_results
            )

            self.logger.info(f"Backtracking identified {len(backtracking_results)} root causes")
        else:
            self.logger.info("No conformance issues detected - model quality is acceptable")
            integrated_issues = classified_issues
            backtracking_results = []

        # ========== STAGE 7: Enhanced Visualization with Quality Information ==========
        self.logger.info("Stage 7: Enhanced visualization")
        visualization = self.visualizer.create_enhanced_visualization(
            process_model, integrated_issues
        )

        # ========== Compile Final Results ==========
        results = {
            'raw_data': raw_data,
            'preprocessed_data': preprocessed_data,
            'structured_events': structured_events,
            'process_instances': process_instances,
            'process_model': process_model,
            'visualization': visualization,
            'quality_issues': integrated_issues,
            'conformance_issues': conformance_issues,
            'backtracking_results': backtracking_results,
            'quality_analysis': quality_analysis,
            'propagation_results': {
                'preprocessing': preprocessing_propagation,
                'event_abstraction': event_propagation,
                'case_correlation': case_propagation,
                'process_mining': mining_propagation
            },
            'conformance_triggered': quality_analysis.get('has_conformance_issues', False)
        }

        self.pipeline_results = results
        self.logger.info("Pipeline execution completed")

        return results

    def _perform_conformance_backtracking(self, conformance_issues: List[Dict[str, Any]],
                                          quality_analysis: Dict[str, Any],
                                          pipeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform backtracking from conformance issues to root causes"""

        backtracking_results = []

        # Get backtracking results from quality analysis
        analysis_backtracking = quality_analysis.get('backtracking_results', [])

        for backtrack in analysis_backtracking:
            # Enhance with detailed pipeline context
            enhanced_backtrack = {
                'conformance_issue': backtrack['conformance_issue'],
                'conformance_value': backtrack['conformance_value'],
                'root_causes': backtrack.get('potential_root_causes', []),
                'affected_cases': backtrack.get('affected_cases', []),
                'confidence': backtrack.get('confidence', 0.0),
                'backtrack_path': self._construct_backtrack_path(
                    backtrack, pipeline_data
                ),
                'evidence_chain': self._build_evidence_chain(
                    backtrack, pipeline_data
                ),
                'actionable_insights': self._generate_actionable_insights(
                    backtrack, pipeline_data
                )
            }

            backtracking_results.append(enhanced_backtrack)

        return backtracking_results

    def _construct_backtrack_path(self, backtrack: Dict[str, Any],
                                  pipeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construct the backtracking path from model to root cause"""

        path = []

        # Stage 5: Process Model Level
        path.append({
            'stage': 'Process Model',
            'stage_number': 5,
            'observation': f"Conformance issue: {backtrack['conformance_issue']}",
            'metric_value': backtrack['conformance_value'],
            'evidence': f"Model quality metric below threshold"
        })

        # Stage 4: Case Level
        affected_cases = backtrack.get('affected_cases', [])
        if affected_cases:
            path.append({
                'stage': 'Case Correlation',
                'stage_number': 4,
                'observation': f"{len(affected_cases)} cases with quality issues",
                'affected_items': affected_cases[:5],  # First 5 cases
                'evidence': "Cases contain quality flags from earlier stages"
            })

        # Stage 3: Event Level
        root_causes = backtrack.get('potential_root_causes', [])
        if root_causes:
            for root_cause in root_causes[:3]:  # Top 3 root causes
                issue_type = root_cause.get('issue_type', 'unknown')

                # Determine which stage this originated from
                if issue_type in ['C1_inadequate_sampling', 'C3_sensor_noise', 'C4_range_too_small']:
                    path.append({
                        'stage': 'Raw Data',
                        'stage_number': 1,
                        'observation': f"Quality issue detected: {issue_type}",
                        'probability': root_cause.get('probability', 0),
                        'evidence': f"Sensor-level issue affecting {root_cause.get('occurrence_count', 0)} cases",
                        'explanation': root_cause.get('explanation', '')
                    })

        return path

    def _build_evidence_chain(self, backtrack: Dict[str, Any],
                              pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a chain of evidence linking conformance issue to root cause"""

        evidence_chain = {
            'model_level': {},
            'case_level': {},
            'event_level': {},
            'raw_data_level': {},
            'chain_strength': 0.0
        }

        # Model level evidence
        evidence_chain['model_level'] = {
            'conformance_issue': backtrack['conformance_issue'],
            'metric_value': backtrack['conformance_value'],
            'severity': 'high' if backtrack['conformance_value'] < 0.5 else 'medium'
        }

        # Case level evidence
        affected_cases = backtrack.get('affected_cases', [])
        if affected_cases:
            process_instances = pipeline_data.get('process_instances', pd.DataFrame())
            if not process_instances.empty:
                affected_case_data = process_instances[
                    process_instances['case_id'].isin(affected_cases)
                ]

                evidence_chain['case_level'] = {
                    'affected_case_count': len(affected_cases),
                    'avg_case_quality': affected_case_data[
                        'case_quality_score'].mean() if 'case_quality_score' in affected_case_data.columns else 0,
                    'total_quality_issues': affected_case_data[
                        'num_quality_issues'].sum() if 'num_quality_issues' in affected_case_data.columns else 0
                }

        # Root cause level evidence
        root_causes = backtrack.get('potential_root_causes', [])
        if root_causes:
            primary_root = root_causes[0]  # Highest probability root cause

            evidence_chain['raw_data_level'] = {
                'primary_root_cause': primary_root.get('issue_type', 'unknown'),
                'occurrence_count': primary_root.get('occurrence_count', 0),
                'probability': primary_root.get('probability', 0),
                'relevance': primary_root.get('relevance', 'unknown'),
                'explanation': primary_root.get('explanation', '')
            }

            # Calculate chain strength
            evidence_chain['chain_strength'] = (
                    backtrack.get('confidence', 0) *
                    primary_root.get('probability', 0)
            )

        return evidence_chain

    def _generate_actionable_insights(self, backtrack: Dict[str, Any],
                                      pipeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from backtracking results"""

        insights = []

        root_causes = backtrack.get('potential_root_causes', [])

        for root_cause in root_causes[:3]:  # Top 3 root causes
            issue_type = root_cause.get('issue_type', 'unknown')
            probability = root_cause.get('probability', 0)

            insight = {
                'issue': issue_type,
                'confidence': probability,
                'impact': f"Causing {backtrack['conformance_issue']} (value: {backtrack['conformance_value']:.3f})",
                'recommendation': self._get_recommendation_for_issue(issue_type),
                'priority': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.4 else 'LOW',
                'affected_sensors': self._identify_affected_sensors(root_cause, pipeline_data)
            }

            insights.append(insight)

        return insights

    def _get_recommendation_for_issue(self, issue_type: str) -> str:
        """Get recommendation for a specific issue type"""

        recommendations = {
            'C1_inadequate_sampling': 'Increase sensor sampling rate to at least 2 Hz',
            'C2_poor_placement': 'Review and adjust sensor placement to eliminate overlaps',
            'C3_sensor_noise': 'Calibrate sensors and check for electrical interference',
            'C4_range_too_small': 'Upgrade sensors to models with larger measurement ranges',
            'C5_high_volume': 'Scale data processing infrastructure or implement edge computing'
        }

        return recommendations.get(issue_type, 'Review sensor configuration and maintenance')

    def _identify_affected_sensors(self, root_cause: Dict[str, Any],
                                   pipeline_data: Dict[str, Any]) -> List[str]:
        """Identify which specific sensors are affected"""

        affected_sensors = []

        # Get affected cases
        affected_case_ids = root_cause.get('affected_cases', [])

        if affected_case_ids:
            # Find sensors involved in affected cases
            raw_data = pipeline_data.get('raw_data', pd.DataFrame())
            if not raw_data.empty and 'case_id' in raw_data.columns:
                affected_data = raw_data[raw_data['case_id'].isin(affected_case_ids)]
                affected_sensors = affected_data['sensor_id'].unique().tolist()

        return affected_sensors

    def _integrate_conformance_findings(self, initial_issues: List[Dict[str, Any]],
                                        conformance_issues: List[Dict[str, Any]],
                                        backtracking_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Integrate conformance findings into the quality issues list"""

        integrated_issues = initial_issues.copy()

        # Add conformance-detected issues
        for conf_issue in conformance_issues:
            # Check if this conformance issue has backtracking results
            related_backtrack = None
            for bt in backtracking_results:
                if bt['conformance_issue'] == conf_issue['type']:
                    related_backtrack = bt
                    break

            # Create enhanced conformance issue
            enhanced_issue = {
                'type': f"conformance_{conf_issue['type']}",
                'description': conf_issue['description'],
                'severity': conf_issue['severity'],
                'confidence': related_backtrack['confidence'] if related_backtrack else 0.5,
                'evidence': {
                    'conformance_metric': conf_issue.get('conformance_metric', 'unknown'),
                    'metric_value': conf_issue.get('value', 0),
                    'threshold': conf_issue.get('threshold', 0)
                },
                'backtracking': related_backtrack,
                'source': 'conformance_checking',
                'stage_detected': 'process_mining'
            }

            integrated_issues.append(enhanced_issue)

        return integrated_issues