import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class ExplanationGenerator:
    """Generates detailed explanations for quality issues and their impacts"""

    def __init__(self):
        self.explanation_templates = self._initialize_explanation_templates()

    def generate_explanation(self, issue: Dict[str, Any],
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate detailed explanation for a specific quality issue"""

        issue_type = issue.get('type', 'unknown')

        if issue_type in self.explanation_templates:
            template = self.explanation_templates[issue_type]

            explanation = {
                'issue_summary': self._generate_issue_summary(issue),
                'technical_explanation': template['technical_explanation'],
                'root_cause_analysis': self._analyze_root_cause(issue, template),
                'impact_analysis': self._analyze_impact(issue, context),
                'evidence_presentation': self._present_evidence(issue),
                'remediation_strategy': self._generate_remediation_strategy(issue, template),
                'prevention_measures': template['prevention_measures']
            }

            return explanation

        return self._generate_generic_explanation(issue)

    def _generate_issue_summary(self, issue: Dict[str, Any]) -> str:
        """Generate a concise summary of the issue"""

        issue_type = issue.get('type', 'unknown')
        confidence = issue.get('confidence', 0.5)
        severity = issue.get('severity', 'medium')

        type_descriptions = {
            'C1_inadequate_sampling': 'Sensor sampling rate is insufficient to capture all process events',
            'C2_poor_placement': 'Sensor placement causes overlapping or inconsistent readings',
            'C3_sensor_noise': 'High levels of noise and outliers in sensor data',
            'C4_range_too_small': 'Sensor measurement range is insufficient for process requirements',
            'C5_high_volume': 'High data volume causes processing delays and data loss'
        }

        base_description = type_descriptions.get(issue_type, f'Quality issue of type {issue_type}')

        return f"{base_description} (Confidence: {confidence:.1%}, Severity: {severity})"

    def _analyze_root_cause(self, issue: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the root cause of the issue"""

        evidence = issue.get('evidence', {})

        root_cause = {
            'primary_cause': template['primary_cause'],
            'contributing_factors': template.get('contributing_factors', []),
            'evidence_supporting_cause': self._extract_supporting_evidence(evidence, template),
            'likelihood_assessment': issue.get('confidence', 0.5)
        }

        return root_cause

    def _extract_supporting_evidence(self, evidence: Dict[str, Any],
                                     template: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize evidence that supports the identified root cause"""

        supporting_evidence = {
            'quantitative_indicators': {},
            'qualitative_indicators': [],
            'correlation_strength': 0.0
        }

        # Extract quantitative evidence based on issue type
        issue_type = template.get('primary_cause', '')

        # Map evidence keys to human-readable indicators
        evidence_mapping = {
            'sampling_rate': 'Measured sampling rate',
            'gap_count': 'Number of data gaps detected',
            'outlier_ratio': 'Proportion of outlier readings',
            'noise_level': 'Signal noise level',
            'clipped_ratio': 'Proportion of clipped values',
            'inconsistency_score': 'Sensor inconsistency score',
            'drop_ratio': 'Data drop rate',
            'timing_cv': 'Timing variability coefficient'
        }

        for key, value in evidence.items():
            if isinstance(value, (int, float)):
                human_key = evidence_mapping.get(key, key)
                supporting_evidence['quantitative_indicators'][human_key] = value

        # Determine correlation strength based on evidence quality
        if len(supporting_evidence['quantitative_indicators']) > 0:
            # Calculate average deviation from expected values
            deviations = []

            if 'sampling_rate' in evidence:
                expected_rate = 2.0  # Expected 2 Hz
                actual_rate = evidence['sampling_rate']
                if actual_rate < expected_rate:
                    deviations.append(1.0 - (actual_rate / expected_rate))

            if 'outlier_ratio' in evidence:
                expected_ratio = 0.05  # Expected 5% outliers
                actual_ratio = evidence['outlier_ratio']
                if actual_ratio > expected_ratio:
                    deviations.append(min(1.0, actual_ratio / expected_ratio))

            if 'drop_ratio' in evidence:
                actual_drop = evidence['drop_ratio']
                deviations.append(min(1.0, actual_drop))

            if deviations:
                supporting_evidence['correlation_strength'] = min(1.0, sum(deviations) / len(deviations))
            else:
                supporting_evidence['correlation_strength'] = 0.5

        # Add qualitative indicators
        if 'sampling_rate' in evidence and evidence['sampling_rate'] < 1.0:
            supporting_evidence['qualitative_indicators'].append(
                'Sampling rate is below recommended minimum of 1 Hz'
            )

        if 'outlier_ratio' in evidence and evidence['outlier_ratio'] > 0.1:
            supporting_evidence['qualitative_indicators'].append(
                'High proportion of outlier readings indicates sensor issues'
            )

        if 'gap_count' in evidence and evidence['gap_count'] > 10:
            supporting_evidence['qualitative_indicators'].append(
                'Multiple data gaps suggest intermittent sensor failures'
            )

        if 'inconsistency_score' in evidence and evidence['inconsistency_score'] > 1.0:
            supporting_evidence['qualitative_indicators'].append(
                'High inconsistency in readings suggests placement or interference issues'
            )

        # If no qualitative indicators found, add generic one
        if not supporting_evidence['qualitative_indicators']:
            supporting_evidence['qualitative_indicators'].append(
                'Evidence patterns consistent with identified root cause'
            )

        return supporting_evidence

    def _analyze_impact(self, issue: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the impact of the issue across pipeline stages"""

        impact_analysis = {
            'immediate_impacts': [],
            'cascading_effects': [],
            'affected_pipeline_stages': [],
            'business_impact': 'Medium',  # Default
            'technical_impact': 'Medium'
        }

        # Extract stage effects if available
        stage_effects = issue.get('stage_effects', {})

        # Immediate impacts from stage effects
        if stage_effects.get('missing_samples'):
            impact_analysis['immediate_impacts'].append('Missing sensor readings in time series')
        if stage_effects.get('noise_amplification'):
            impact_analysis['immediate_impacts'].append('Amplified noise in processed data')
        if stage_effects.get('missing_events'):
            impact_analysis['immediate_impacts'].append('Undetected process events')

        # Cascading effects
        if stage_effects.get('spaghetti_model'):
            impact_analysis['cascading_effects'].append('Overly complex process model with many spurious paths')
        if stage_effects.get('low_fitness'):
            impact_analysis['cascading_effects'].append('Poor model fitness reducing analytical value')
        if stage_effects.get('case_fragmentation'):
            impact_analysis['cascading_effects'].append('Fragmented process instances affecting analysis')

        # Determine business impact based on severity and effects
        severity = issue.get('severity', 'medium')
        if severity == 'high' or len(impact_analysis['cascading_effects']) > 2:
            impact_analysis['business_impact'] = 'High'
        elif severity == 'low' and len(impact_analysis['immediate_impacts']) <= 1:
            impact_analysis['business_impact'] = 'Low'

        return impact_analysis

    def _present_evidence(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Present evidence in an explainable format"""

        evidence = issue.get('evidence', {})

        presented_evidence = {
            'quantitative_evidence': {},
            'qualitative_evidence': {},
            'visual_indicators': [],
            'statistical_measures': {}
        }

        # Categorize evidence
        for key, value in evidence.items():
            if isinstance(value, (int, float)):
                presented_evidence['quantitative_evidence'][key] = {
                    'value': value,
                    'interpretation': self._interpret_quantitative_evidence(key, value)
                }
            elif isinstance(value, str):
                presented_evidence['qualitative_evidence'][key] = value
            elif isinstance(value, list):
                if all(isinstance(item, (int, float)) for item in value):
                    presented_evidence['statistical_measures'][key] = {
                        'values': value,
                        'summary': self._summarize_numeric_list(value)
                    }

        # Add visualization recommendations
        issue_type = issue.get('type')
        if issue_type == 'C1_inadequate_sampling':
            presented_evidence['visual_indicators'].append('Time series plot showing gaps in data')
        elif issue_type == 'C3_sensor_noise':
            presented_evidence['visual_indicators'].append('Histogram showing outlier distribution')
            presented_evidence['visual_indicators'].append('Time series with noise levels highlighted')

        return presented_evidence

    def _generate_remediation_strategy(self, issue: Dict[str, Any],
                                       template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed remediation strategy"""

        strategy = {
            'immediate_actions': template.get('immediate_actions', []),
            'short_term_solutions': template.get('short_term_solutions', []),
            'long_term_improvements': template.get('long_term_improvements', []),
            'implementation_priority': self._assess_implementation_priority(issue),
            'resource_requirements': template.get('resource_requirements', {}),
            'success_metrics': template.get('success_metrics', [])
        }

        # Customize based on specific evidence
        evidence = issue.get('evidence', {})
        strategy = self._customize_strategy_for_evidence(strategy, evidence, issue)

        return strategy

    def _assess_implementation_priority(self, issue: Dict[str, Any]) -> str:
        """Assess implementation priority based on issue characteristics"""

        confidence = issue.get('confidence', 0.5)
        severity = issue.get('severity', 'medium')

        # High confidence + high severity = critical priority
        if confidence > 0.8 and severity == 'high':
            return 'Critical'
        elif confidence > 0.7 and severity in ['high', 'medium']:
            return 'High'
        elif confidence > 0.5:
            return 'Medium'
        else:
            return 'Low'

    def _customize_strategy_for_evidence(self, strategy: Dict[str, Any],
                                         evidence: Dict[str, Any],
                                         issue: Dict[str, Any]) -> Dict[str, Any]:
        """Customize remediation strategy based on specific evidence"""

        issue_type = issue.get('type')

        if issue_type == 'C1_inadequate_sampling':
            sampling_rate = evidence.get('sampling_rate_estimate', 0)
            if sampling_rate < 0.5:
                strategy['immediate_actions'].insert(0,
                                                     f'Increase sampling rate from {sampling_rate:.2f} Hz to at least 2 Hz')

        elif issue_type == 'C3_sensor_noise':
            outlier_ratio = evidence.get('outlier_ratio', 0)
            if outlier_ratio > 0.2:
                strategy['immediate_actions'].insert(0,
                                                     f'Investigate cause of {outlier_ratio:.1%} outlier rate')

        return strategy

    def _interpret_quantitative_evidence(self, key: str, value: float) -> str:
        """Interpret quantitative evidence values"""

        interpretations = {
            'sampling_rate': f'Current rate of {value:.2f} Hz may miss fast events',
            'outlier_ratio': f'{value:.1%} of readings are outliers (threshold: 5%)',
            'confidence': f'Detection confidence is {value:.1%}',
            'gap_count': f'{int(value)} gaps detected in sensor data',
            'noise_level': f'Noise level is {value:.3f} (normalized)',
            'clipped_ratio': f'{value:.1%} of readings appear clipped'
        }

        return interpretations.get(key, f'Value: {value}')

    def _summarize_numeric_list(self, values: List[float]) -> Dict[str, float]:
        """Summarize a list of numeric values"""

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }

    def _generate_generic_explanation(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic explanation for unknown issue types"""

        return {
            'issue_summary': f"Quality issue detected: {issue.get('type', 'unknown')}",
            'technical_explanation': 'Specific technical details not available for this issue type',
            'root_cause_analysis': {
                'primary_cause': 'Unknown - requires further investigation',
                'contributing_factors': [],
                'evidence_supporting_cause': issue.get('evidence', {}),
                'likelihood_assessment': issue.get('confidence', 0.5)
            },
            'impact_analysis': {
                'immediate_impacts': ['Potential data quality degradation'],
                'cascading_effects': ['May affect downstream analysis'],
                'business_impact': 'Unknown'
            },
            'evidence_presentation': self._present_evidence(issue),
            'remediation_strategy': {
                'immediate_actions': ['Investigate issue characteristics'],
                'short_term_solutions': ['Implement monitoring'],
                'long_term_improvements': ['Develop specific remediation plan']
            }
        }

    def _initialize_explanation_templates(self) -> Dict[str, Any]:
        """Initialize explanation templates for different issue types"""

        return {
            'C1_inadequate_sampling': {
                'technical_explanation': (
                    'Inadequate sampling rate occurs when sensors sample too slowly to capture '
                    'all relevant process events. This typically happens when fast-changing '
                    'processes are monitored with low-frequency sensors, causing short-duration '
                    'events to be missed entirely.'
                ),
                'primary_cause': 'Sensor sampling frequency is insufficient for process dynamics',
                'contributing_factors': [
                    'Outdated sensor configuration',
                    'Hardware limitations of legacy sensors',
                    'Process speed increased without sensor updates',
                    'Power management reducing sampling rates'
                ],
                'immediate_actions': [
                    'Review and increase sensor sampling rates',
                    'Identify critical sensors with timing issues',
                    'Implement temporary high-frequency monitoring'
                ],
                'short_term_solutions': [
                    'Update sensor configuration files',
                    'Implement adaptive sampling strategies',
                    'Add interpolation for missing data points'
                ],
                'long_term_improvements': [
                    'Upgrade to higher-frequency sensors',
                    'Implement edge computing for real-time processing',
                    'Deploy intelligent sampling algorithms'
                ],
                'resource_requirements': {
                    'technical_effort': 'Medium',
                    'cost_estimate': 'Low to Medium',
                    'timeline': '1-4 weeks'
                },
                'success_metrics': [
                    'Reduction in data gaps',
                    'Improved event detection accuracy',
                    'Increased model fitness scores'
                ],
                'prevention_measures': [
                    'Regular sensor performance monitoring',
                    'Automated sampling rate optimization',
                    'Process-sensor alignment reviews'
                ]
            },

            'C2_poor_placement': {
                'technical_explanation': (
                    'Poor sensor placement results in sensors providing overlapping, inconsistent, '
                    'or incomplete coverage of the process. This can lead to ambiguous readings, '
                    'missed events, or false event detection due to sensors being positioned '
                    'suboptimally relative to the process being monitored.'
                ),
                'primary_cause': 'Sensors are not optimally positioned for process monitoring',
                'contributing_factors': [
                    'Lack of process understanding during sensor installation',
                    'Physical constraints limiting optimal placement',
                    'Process layout changes after sensor installation',
                    'Insufficient sensor coverage planning'
                ],
                'immediate_actions': [
                    'Audit current sensor positions',
                    'Map sensor coverage against process areas',
                    'Identify overlapping or conflicting readings'
                ],
                'short_term_solutions': [
                    'Relocate problematic sensors',
                    'Add sensors for missing coverage areas',
                    'Implement sensor fusion algorithms'
                ],
                'long_term_improvements': [
                    'Develop comprehensive sensor placement strategy',
                    'Use simulation tools for optimal placement',
                    'Implement dynamic sensor networks'
                ],
                'resource_requirements': {
                    'technical_effort': 'High',
                    'cost_estimate': 'Medium to High',
                    'timeline': '2-8 weeks'
                },
                'success_metrics': [
                    'Reduced reading inconsistencies',
                    'Improved process coverage',
                    'Better event correlation accuracy'
                ],
                'prevention_measures': [
                    'Sensor placement modeling and simulation',
                    'Regular coverage assessment',
                    'Process change impact analysis'
                ]
            },

            'C3_sensor_noise': {
                'technical_explanation': (
                    'Sensor noise and outliers occur when sensors produce erroneous readings '
                    'due to electrical interference, calibration drift, environmental factors, '
                    'or sensor degradation. This manifests as random fluctuations, sudden '
                    'spikes, or systematic bias in sensor measurements.'
                ),
                'primary_cause': 'Sensor hardware issues or environmental interference',
                'contributing_factors': [
                    'Electrical interference from nearby equipment',
                    'Sensor calibration drift over time',
                    'Environmental factors (temperature, humidity, vibration)',
                    'Aging sensor components',
                    'Power supply instabilities'
                ],
                'immediate_actions': [
                    'Check sensor calibration status',
                    'Inspect for electrical interference sources',
                    'Review sensor maintenance logs'
                ],
                'short_term_solutions': [
                    'Recalibrate affected sensors',
                    'Implement noise filtering algorithms',
                    'Shield sensors from interference sources'
                ],
                'long_term_improvements': [
                    'Upgrade to higher-quality sensors',
                    'Implement predictive maintenance',
                    'Deploy environmental monitoring for sensor locations'
                ],
                'resource_requirements': {
                    'technical_effort': 'Medium',
                    'cost_estimate': 'Low to Medium',
                    'timeline': '1-6 weeks'
                },
                'success_metrics': [
                    'Reduced outlier frequency',
                    'Improved signal-to-noise ratio',
                    'More stable sensor readings'
                ],
                'prevention_measures': [
                    'Regular sensor calibration schedule',
                    'Environmental monitoring',
                    'Interference source identification and mitigation'
                ]
            },

            'C4_range_too_small': {
                'technical_explanation': (
                    'Sensor range limitations occur when the measurement range of sensors '
                    'is insufficient to capture the full range of process values. This results '
                    'in clipped readings, blind spots during certain process phases, and '
                    'loss of information about process extremes.'
                ),
                'primary_cause': 'Sensor measurement range is inadequate for process requirements',
                'contributing_factors': [
                    'Process specifications changed after sensor installation',
                    'Original sensor specification was insufficient',
                    'Multiple operating modes with different ranges',
                    'Extreme conditions not considered in original design'
                ],
                'immediate_actions': [
                    'Identify value clipping incidents',
                    'Map process value ranges vs sensor capabilities',
                    'Document missed process phases'
                ],
                'short_term_solutions': [
                    'Adjust sensor configuration for extended range',
                    'Implement range switching algorithms',
                    'Add complementary sensors for extreme values'
                ],
                'long_term_improvements': [
                    'Upgrade to sensors with larger measurement ranges',
                    'Implement multi-range sensor arrays',
                    'Deploy adaptive range adjustment systems'
                ],
                'resource_requirements': {
                    'technical_effort': 'Medium to High',
                    'cost_estimate': 'Medium',
                    'timeline': '2-6 weeks'
                },
                'success_metrics': [
                    'Elimination of value clipping',
                    'Complete process phase coverage',
                    'Improved measurement accuracy at extremes'
                ],
                'prevention_measures': [
                    'Comprehensive process range analysis',
                    'Safety margin in sensor specifications',
                    'Regular range adequacy reviews'
                ]
            },

            'C5_high_volume': {
                'technical_explanation': (
                    'High data volume/velocity issues occur when the rate of data generation '
                    'exceeds the processing or storage capabilities of the system. This leads '
                    'to buffer overflows, dropped samples, timestamp drift, and processing delays '
                    'that can compromise data quality and real-time analysis capabilities.'
                ),
                'primary_cause': 'Data processing infrastructure cannot handle current data rates',
                'contributing_factors': [
                    'Increased number of sensors without infrastructure scaling',
                    'Higher sampling frequencies exceeding processing capacity',
                    'Network bandwidth limitations',
                    'Insufficient processing power or memory',
                    'Inefficient data processing algorithms'
                ],
                'immediate_actions': [
                    'Monitor system resource utilization',
                    'Identify data processing bottlenecks',
                    'Implement temporary data throttling'
                ],
                'short_term_solutions': [
                    'Optimize data processing algorithms',
                    'Implement data compression and buffering',
                    'Upgrade network infrastructure'
                ],
                'long_term_improvements': [
                    'Scale processing infrastructure',
                    'Implement edge computing solutions',
                    'Deploy distributed processing architecture'
                ],
                'resource_requirements': {
                    'technical_effort': 'High',
                    'cost_estimate': 'Medium to High',
                    'timeline': '4-12 weeks'
                },
                'success_metrics': [
                    'Reduced data loss rates',
                    'Improved timestamp accuracy',
                    'Consistent processing latency'
                ],
                'prevention_measures': [
                    'Capacity planning and monitoring',
                    'Scalable architecture design',
                    'Performance testing and optimization'
                ]
            }
        }