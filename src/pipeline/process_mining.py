# src/pipeline/process_mining.py - REFACTORED with pm4py
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.conversion.log import converter as log_converter


class ProcessMiner:
    """Discovers process models using pm4py with quality-aware conformance checking"""

    def __init__(self, noise_threshold: float = 0.2, conformance_threshold: float = 0.7):
        self.noise_threshold = noise_threshold
        self.conformance_threshold = conformance_threshold
        self.logger = logging.getLogger(__name__)

    def discover_process(self, case_instances: pd.DataFrame) -> Dict[str, Any]:
        """Discover process model from case instances using pm4py Inductive Miner"""

        if case_instances.empty:
            return self._create_empty_model()


        # Convert case instances to pm4py event log format
        event_log = self._convert_to_event_log(case_instances)

        if event_log is None or len(event_log) == 0:
            self.logger.warning("Event log conversion failed or empty")
            return self._create_empty_model()

        # Apply Inductive Miner to discover process model
        self.logger.info(f"Discovering process model with Inductive Miner ({len(event_log)} traces)")
        net, initial_marking, final_marking = inductive_miner.apply(
            event_log,
            variant=inductive_miner.Variants.IMf,  # Inductive Miner - infrequent
            parameters={'noise_threshold': self.noise_threshold}
        )

        # Calculate conformance metrics
        conformance_results = self._calculate_conformance(
            event_log, net, initial_marking, final_marking
        )

        # Detect quality issues based on conformance
        conformance_issues = self._detect_conformance_issues(
            conformance_results, case_instances
        )

        # Extract model information
        model_info = self._extract_model_info(net, initial_marking, final_marking)

        # Calculate quality-weighted metrics
        quality_metrics = self._calculate_quality_metrics(
            conformance_results, case_instances
        )

        # Analyze quality issue impact on model
        quality_analysis = self._analyze_quality_impact_on_model(
            conformance_issues, conformance_results, case_instances
        )

        return {
            'model': {
                'petri_net': net,
                'initial_marking': initial_marking,
                'final_marking': final_marking,
                'places': list(net.places),
                'transitions': list(net.transitions),
                'arcs': list(net.arcs),
                **model_info
            },
            'event_log': event_log,
            'conformance': conformance_results,
            'metrics': quality_metrics,
            'conformance_issues': conformance_issues,
            'quality_analysis': quality_analysis,
            'discovery_algorithm': 'Inductive Miner (IMf)'
        }



    def _convert_to_event_log(self, case_instances: pd.DataFrame) -> Any:
        """Convert case instances to pm4py event log format"""

        try:
            # Create event log DataFrame
            events = []

            for _, case in case_instances.iterrows():
                case_id = case['case_id']
                activity_sequence = case.get('activity_sequence', [])

                if not activity_sequence:
                    continue

                # Get timing information if available
                start_time = case.get('start_time', datetime.now())
                duration = case.get('duration', 0)

                # Calculate event timestamps
                num_events = len(activity_sequence)
                if num_events > 0:
                    time_per_event = duration / num_events if duration > 0 else 1

                    for i, activity in enumerate(activity_sequence):
                        event_time = start_time + timedelta(seconds=i * time_per_event)

                        event = {
                            'case:concept:name': str(case_id),
                            'concept:name': str(activity),
                            'time:timestamp': event_time,
                            'case_quality_score': case.get('case_quality_score', 1.0),
                            'num_quality_issues': case.get('num_quality_issues', 0)
                        }
                        events.append(event)

            if not events:
                self.logger.warning("No events to convert to log")
                return None

            # Convert to DataFrame
            event_df = pd.DataFrame(events)

            # Sort by case and time
            event_df = event_df.sort_values(['case:concept:name', 'time:timestamp'])

            # Convert to pm4py event log
            event_log = log_converter.apply(
                event_df,
                variant=log_converter.Variants.TO_EVENT_LOG,
                parameters={
                    log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'
                }
            )

            self.logger.info(f"Converted {len(events)} events into {len(event_log)} traces")
            return event_log

        except Exception as e:
            self.logger.error(f"Error converting to event log: {e}", exc_info=True)
            return None

    def _calculate_conformance(self, event_log, net, initial_marking, final_marking) -> Dict[str, Any]:
        """Calculate conformance metrics using pm4py"""

        conformance_results = {
            'fitness': 0.0,
            'precision': 0.0,
            'generalization': 0.0,
            'simplicity': 0.0,
            'trace_fitness': [],
            'alignments': None
        }

        try:
            # Calculate fitness (replay fitness)
            fitness_result = replay_fitness.apply(
                event_log, net, initial_marking, final_marking,
                variant=replay_fitness.Variants.TOKEN_BASED
            )
            conformance_results['fitness'] = fitness_result['average_trace_fitness']
            conformance_results['trace_fitness'] = fitness_result.get('trace_fitness', [])

            self.logger.info(f"Fitness calculated: {conformance_results['fitness']:.3f}")

        except Exception as e:
            self.logger.warning(f"Error calculating fitness: {e}")

        try:
            # Calculate precision
            precision_result = precision_evaluator.apply(
                event_log, net, initial_marking, final_marking,
                variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
            )
            conformance_results['precision'] = precision_result

            self.logger.info(f"Precision calculated: {conformance_results['precision']:.3f}")

        except Exception as e:
            self.logger.warning(f"Error calculating precision: {e}")

        try:
            # Calculate generalization
            generalization_result = generalization_evaluator.apply(
                event_log, net, initial_marking, final_marking
            )
            conformance_results['generalization'] = generalization_result

        except Exception as e:
            self.logger.warning(f"Error calculating generalization: {e}")

        try:
            # Calculate simplicity
            simplicity_result = simplicity_evaluator.apply(net)
            conformance_results['simplicity'] = simplicity_result

        except Exception as e:
            self.logger.warning(f"Error calculating simplicity: {e}")

        # Calculate alignments for detailed conformance analysis
        try:
            alignments_result = alignments.apply(
                event_log, net, initial_marking, final_marking,
                variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR
            )
            conformance_results['alignments'] = alignments_result

        except Exception as e:
            self.logger.warning(f"Error calculating alignments: {e}")

        return conformance_results

    def _detect_conformance_issues(self, conformance_results: Dict[str, Any],
                                   case_instances: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect quality issues based on conformance checking"""

        conformance_issues = []

        # Issue 1: Low Fitness
        fitness = conformance_results.get('fitness', 1.0)
        if fitness < self.conformance_threshold:
            issue = {
                'type': 'low_fitness',
                'severity': 'high' if fitness < 0.5 else 'medium',
                'description': f'Model fitness is low ({fitness:.3f})',
                'conformance_metric': 'fitness',
                'value': fitness,
                'threshold': self.conformance_threshold,
                'affected_traces': self._identify_low_fitness_traces(conformance_results),
                'potential_causes': [
                    'Missing events in traces',
                    'Incorrect event ordering',
                    'Data quality issues affecting event detection',
                    'Process variants not captured by model'
                ]
            }
            conformance_issues.append(issue)
            self.logger.warning(f"Detected low fitness issue: {fitness:.3f}")

        # Issue 2: Low Precision
        precision = conformance_results.get('precision', 1.0)
        if precision < self.conformance_threshold:
            issue = {
                'type': 'low_precision',
                'severity': 'medium',
                'description': f'Model precision is low ({precision:.3f})',
                'conformance_metric': 'precision',
                'value': precision,
                'threshold': self.conformance_threshold,
                'potential_causes': [
                    'Model allows too many behaviors',
                    'Noisy data creating spurious paths',
                    'Overgeneralized model structure',
                    'Sensor noise creating false events'
                ]
            }
            conformance_issues.append(issue)
            self.logger.warning(f"Detected low precision issue: {precision:.3f}")

        # Issue 3: Low Simplicity (Complex Model)
        simplicity = conformance_results.get('simplicity', 1.0)
        if simplicity < 0.5:  # Complexity threshold
            issue = {
                'type': 'high_complexity',
                'severity': 'medium',
                'description': f'Model is overly complex (simplicity: {simplicity:.3f})',
                'conformance_metric': 'simplicity',
                'value': simplicity,
                'threshold': 0.5,
                'potential_causes': [
                    'Excessive noise in sensor data',
                    'Many process variants due to quality issues',
                    'Poor event abstraction',
                    'Inconsistent sensor readings'
                ]
            }
            conformance_issues.append(issue)
            self.logger.warning(f"Detected high complexity issue: {simplicity:.3f}")

        # Issue 4: Trace-Level Conformance Issues
        trace_fitness = conformance_results.get('trace_fitness', [])
        if trace_fitness:
            low_fitness_traces = [tf for tf in trace_fitness if tf.get('trace_fitness', 1.0) < 0.7]

            if len(low_fitness_traces) > len(trace_fitness) * 0.2:  # More than 20% of traces
                issue = {
                    'type': 'widespread_nonconformance',
                    'severity': 'high',
                    'description': f'{len(low_fitness_traces)} traces ({len(low_fitness_traces) / len(trace_fitness):.1%}) have low fitness',
                    'conformance_metric': 'trace_fitness',
                    'affected_trace_count': len(low_fitness_traces),
                    'total_traces': len(trace_fitness),
                    'potential_causes': [
                        'Systematic data quality issues',
                        'Missing sensor coverage',
                        'Event detection failures',
                        'Case correlation errors'
                    ]
                }
                conformance_issues.append(issue)
                self.logger.warning(f"Detected widespread nonconformance: {len(low_fitness_traces)} traces")

        return conformance_issues

    def _identify_low_fitness_traces(self, conformance_results: Dict[str, Any]) -> List[str]:
        """Identify traces with low fitness scores"""

        trace_fitness = conformance_results.get('trace_fitness', [])
        low_fitness_traces = []

        for tf in trace_fitness:
            if tf.get('trace_fitness', 1.0) < 0.7:
                trace_id = tf.get('case_id', 'unknown')
                low_fitness_traces.append(trace_id)

        return low_fitness_traces

    def _analyze_quality_impact_on_model(self, conformance_issues: List[Dict[str, Any]],
                                         conformance_results: Dict[str, Any],
                                         case_instances: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how quality issues from earlier stages impact the process model"""

        analysis = {
            'has_conformance_issues': len(conformance_issues) > 0,
            'issue_count': len(conformance_issues),
            'primary_conformance_issue': None,
            'backtracking_results': [],
            'quality_issue_correlation': {}
        }

        if not conformance_issues:
            return analysis

        # Identify primary conformance issue (highest severity)
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        primary_issue = max(
            conformance_issues,
            key=lambda x: severity_order.get(x.get('severity', 'low'), 0)
        )
        analysis['primary_conformance_issue'] = primary_issue

        # Backtrack conformance issues to earlier quality issues
        for conf_issue in conformance_issues:
            backtrack_result = self._backtrack_conformance_issue(
                conf_issue, case_instances
            )
            analysis['backtracking_results'].append(backtrack_result)

        # Correlate with earlier quality issues
        analysis['quality_issue_correlation'] = self._correlate_with_quality_issues(
            conformance_issues, case_instances
        )

        return analysis

    def _backtrack_conformance_issue(self, conformance_issue: Dict[str, Any],
                                     case_instances: pd.DataFrame) -> Dict[str, Any]:
        """Backtrack a conformance issue to find root causes in earlier pipeline stages"""

        backtrack_result = {
            'conformance_issue': conformance_issue['type'],
            'conformance_value': conformance_issue.get('value', 0),
            'potential_root_causes': [],
            'affected_cases': [],
            'confidence': 0.0
        }

        # Analyze case quality scores
        if not case_instances.empty and 'case_quality_score' in case_instances.columns:
            low_quality_cases = case_instances[case_instances['case_quality_score'] < 0.7]

            if len(low_quality_cases) > 0:
                backtrack_result['affected_cases'] = low_quality_cases['case_id'].tolist()

                # Analyze quality issues in affected cases
                quality_issue_counts = defaultdict(int)

                for _, case in low_quality_cases.iterrows():
                    case_quality_issues = case.get('quality_issues', [])
                    for issue in case_quality_issues:
                        # Extract issue type from issue string
                        for issue_type in ['C1_inadequate_sampling', 'C2_poor_placement',
                                           'C3_sensor_noise', 'C4_range_too_small', 'C5_high_volume']:
                            if issue_type in issue:
                                quality_issue_counts[issue_type] += 1

                # Determine most likely root causes
                if quality_issue_counts:
                    total_issues = sum(quality_issue_counts.values())

                    for issue_type, count in sorted(quality_issue_counts.items(),
                                                    key=lambda x: x[1], reverse=True):
                        probability = count / total_issues

                        root_cause = {
                            'issue_type': issue_type,
                            'occurrence_count': count,
                            'probability': probability,
                            'affected_cases': [
                                case['case_id'] for _, case in low_quality_cases.iterrows()
                                if any(issue_type in qi for qi in case.get('quality_issues', []))
                            ]
                        }

                        # Map to conformance issue
                        if conformance_issue['type'] == 'low_fitness':
                            if issue_type in ['C1_inadequate_sampling', 'C4_range_too_small', 'C5_high_volume']:
                                root_cause['relevance'] = 'high'
                                root_cause['explanation'] = self._explain_fitness_connection(issue_type)
                            else:
                                root_cause['relevance'] = 'medium'

                        elif conformance_issue['type'] == 'low_precision':
                            if issue_type in ['C3_sensor_noise', 'C2_poor_placement']:
                                root_cause['relevance'] = 'high'
                                root_cause['explanation'] = self._explain_precision_connection(issue_type)
                            else:
                                root_cause['relevance'] = 'medium'

                        elif conformance_issue['type'] == 'high_complexity':
                            if issue_type == 'C3_sensor_noise':
                                root_cause['relevance'] = 'high'
                                root_cause['explanation'] = self._explain_complexity_connection(issue_type)
                            else:
                                root_cause['relevance'] = 'medium'

                        backtrack_result['potential_root_causes'].append(root_cause)

                    # Calculate overall confidence based on correlation strength
                    if backtrack_result['potential_root_causes']:
                        max_prob = max(rc['probability'] for rc in backtrack_result['potential_root_causes'])
                        backtrack_result['confidence'] = max_prob

        return backtrack_result

    def _explain_fitness_connection(self, issue_type: str) -> str:
        """Explain how a quality issue connects to low fitness"""

        explanations = {
            'C1_inadequate_sampling': 'Inadequate sampling causes missing events, which directly reduces model fitness as traces cannot be properly replayed',
            'C4_range_too_small': 'Limited sensor range creates blind spots, missing critical events and reducing trace fitness',
            'C5_high_volume': 'High data volume leads to dropped events and incomplete traces, lowering model fitness'
        }
        return explanations.get(issue_type, 'This quality issue affects event detection, impacting model fitness')

    def _explain_precision_connection(self, issue_type: str) -> str:
        """Explain how a quality issue connects to low precision"""

        explanations = {
            'C3_sensor_noise': 'Sensor noise creates false events, adding spurious paths to the model and reducing precision',
            'C2_poor_placement': 'Poor sensor placement causes overlapping readings, leading to incorrect event detection and imprecise models'
        }
        return explanations.get(issue_type, 'This quality issue affects model precision')

    def _explain_complexity_connection(self, issue_type: str) -> str:
        """Explain how a quality issue connects to high complexity"""

        explanations = {
            'C3_sensor_noise': 'Sensor noise generates many false event variants, creating a complex spaghetti model with excessive paths'
        }
        return explanations.get(issue_type, 'This quality issue increases model complexity')

    def _correlate_with_quality_issues(self, conformance_issues: List[Dict[str, Any]],
                                       case_instances: pd.DataFrame) -> Dict[str, Any]:
        """Correlate conformance issues with earlier quality issues"""

        correlation = {
            'total_quality_issues': 0,
            'issue_breakdown': {},
            'correlation_strength': 0.0,
            'primary_culprits': []
        }

        if case_instances.empty:
            return correlation

        # Count quality issues across all cases
        all_quality_issues = []
        for _, case in case_instances.iterrows():
            case_issues = case.get('quality_issues', [])
            all_quality_issues.extend(case_issues)

        correlation['total_quality_issues'] = len(all_quality_issues)

        # Break down by issue type
        issue_counts = Counter()
        for issue_str in all_quality_issues:
            for issue_type in ['C1_inadequate_sampling', 'C2_poor_placement',
                               'C3_sensor_noise', 'C4_range_too_small', 'C5_high_volume']:
                if issue_type in issue_str:
                    issue_counts[issue_type] += 1

        correlation['issue_breakdown'] = dict(issue_counts)

        # Calculate correlation strength
        if conformance_issues and all_quality_issues:
            # Simple correlation: ratio of cases with quality issues that also have conformance issues
            cases_with_quality_issues = case_instances[
                case_instances['num_quality_issues'] > 0
                ]

            if len(cases_with_quality_issues) > 0:
                correlation['correlation_strength'] = len(cases_with_quality_issues) / len(case_instances)

        # Identify primary culprits
        if issue_counts:
            top_3 = issue_counts.most_common(3)
            correlation['primary_culprits'] = [
                {'issue_type': issue_type, 'count': count, 'percentage': count / len(all_quality_issues)}
                for issue_type, count in top_3
            ]

        return correlation

    def _extract_model_info(self, net, initial_marking, final_marking) -> Dict[str, Any]:
        """Extract information from Petri net model"""

        return {
            'num_places': len(net.places),
            'num_transitions': len(net.transitions),
            'num_arcs': len(net.arcs),
            'activities': [t.label for t in net.transitions if t.label is not None],
            'start_activities': self._get_start_activities(net, initial_marking),
            'end_activities': self._get_end_activities(net, final_marking)
        }

    def _get_start_activities(self, net, initial_marking) -> List[str]:
        """Get start activities from Petri net"""
        start_activities = []
        for place in initial_marking:
            for arc in place.out_arcs:
                if arc.target.label:
                    start_activities.append(arc.target.label)
        return list(set(start_activities))

    def _get_end_activities(self, net, final_marking) -> List[str]:
        """Get end activities from Petri net"""
        end_activities = []
        for place in final_marking:
            for arc in place.in_arcs:
                if arc.source.label:
                    end_activities.append(arc.source.label)
        return list(set(end_activities))

    def _calculate_quality_metrics(self, conformance_results: Dict[str, Any],
                                   case_instances: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality-aware metrics"""

        metrics = {
            'fitness': conformance_results.get('fitness', 0.0),
            'precision': conformance_results.get('precision', 0.0),
            'generalization': conformance_results.get('generalization', 0.0),
            'simplicity': conformance_results.get('simplicity', 0.0),
            'complexity': 1.0 - conformance_results.get('simplicity', 0.0),
            'quality_weighted_fitness': 0.0
        }

        # Calculate quality-weighted fitness
        if not case_instances.empty and 'case_quality_score' in case_instances.columns:
            quality_scores = case_instances['case_quality_score'].values
            avg_quality = np.mean(quality_scores)

            metrics['quality_weighted_fitness'] = metrics['fitness'] * avg_quality
            metrics['average_case_quality'] = avg_quality

        return metrics


    def _create_empty_model(self) -> Dict[str, Any]:
        """Create empty process model structure"""
        return {
            'model': {
                'activities': [],
                'places': [],
                'transitions': [],
                'arcs': []
            },
            'metrics': {
                'fitness': 0.0,
                'precision': 0.0,
                'complexity': 0.0
            },
            'conformance_issues': [
                {'type': 'no_data', 'severity': 'high', 'description': 'No data available for process discovery'}],
            'quality_analysis': {'has_conformance_issues': True, 'issue_count': 1}
        }