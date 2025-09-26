import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta


class InsightGenerator:
    """Generates explainable insights from quality issues and pipeline results"""

    def __init__(self):
        self.insight_templates = self._initialize_insight_templates()

    def generate_insights(self, pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights from pipeline results"""

        insights = []

        quality_issues = pipeline_results.get('quality_issues', [])
        process_model = pipeline_results.get('process_model', {})
        case_instances = pipeline_results.get('process_instances', pd.DataFrame())

        # Generate insights for each category
        insights.extend(self._generate_quality_insights(quality_issues))
        insights.extend(self._generate_model_insights(process_model))
        insights.extend(self._generate_actionable_insights(quality_issues, process_model))
        insights.extend(self._generate_causal_insights(quality_issues, pipeline_results))
        insights.extend(self._generate_information_gain_insights(quality_issues))

        # Rank insights by importance
        ranked_insights = self._rank_insights(insights)

        return ranked_insights

    def _generate_quality_insights(self, quality_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights about data quality issues"""

        insights = []

        if not quality_issues:
            return insights

        # Analyze issue patterns
        issue_types = [issue['type'] for issue in quality_issues]
        issue_counts = Counter(issue_types)

        # Most common issue type
        most_common_issue = issue_counts.most_common(1)[0]
        insights.append({
            'type': 'quality_pattern',
            'category': 'data_quality',
            'message': f"Most prevalent quality issue: {most_common_issue[0]} ({most_common_issue[1]} occurrences)",
            'confidence': 0.9,
            'actionable': True,
            'evidence': {
                'issue_distribution': dict(issue_counts),
                'total_issues': len(quality_issues)
            },
            'recommendations': self._get_recommendations_for_issue(most_common_issue[0])
        })

        # High confidence issues
        high_confidence_issues = [issue for issue in quality_issues
                                  if issue.get('confidence', 0) > 0.8]

        if high_confidence_issues:
            insights.append({
                'type': 'high_confidence_issues',
                'category': 'data_quality',
                'message': f"Found {len(high_confidence_issues)} high-confidence quality issues requiring immediate attention",
                'confidence': 0.95,
                'actionable': True,
                'evidence': {
                    'high_confidence_count': len(high_confidence_issues),
                    'issue_types': [issue['type'] for issue in high_confidence_issues]
                },
                'recommendations': ['Prioritize investigation of high-confidence issues',
                                    'Implement monitoring for detected patterns']
            })

        # Severity analysis
        severity_counts = Counter(issue.get('severity', 'medium') for issue in quality_issues)
        if severity_counts.get('high', 0) > len(quality_issues) * 0.3:
            insights.append({
                'type': 'severity_concern',
                'category': 'data_quality',
                'message': f"High proportion of severe quality issues ({severity_counts['high']}/{len(quality_issues)})",
                'confidence': 0.85,
                'actionable': True,
                'evidence': {
                    'severity_distribution': dict(severity_counts)
                },
                'recommendations': ['Urgent system review required', 'Check sensor maintenance schedules']
            })

        return insights

    def _generate_model_insights(self, process_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about the process model"""

        insights = []

        if not process_model or 'metrics' not in process_model:
            return insights

        metrics = process_model['metrics']
        model_data = process_model.get('model', {})

        # Fitness insights
        fitness = metrics.get('fitness', 0)
        if fitness < 0.6:
            insights.append({
                'type': 'low_fitness',
                'category': 'process_model',
                'message': f"Low model fitness ({fitness:.2f}) indicates poor alignment between observed behavior and discovered model",
                'confidence': 0.8,
                'actionable': True,
                'evidence': {
                    'fitness_score': fitness,
                    'threshold': 0.6
                },
                'recommendations': [
                    'Review event abstraction parameters',
                    'Check for missing or incorrect activity classifications',
                    'Validate case correlation logic'
                ]
            })

        # Complexity insights
        complexity = metrics.get('complexity', 0)
        if complexity > 0.8:
            insights.append({
                'type': 'high_complexity',
                'category': 'process_model',
                'message': f"Process model is overly complex ({complexity:.2f}), indicating potential data quality issues",
                'confidence': 0.75,
                'actionable': True,
                'evidence': {
                    'complexity_score': complexity,
                    'num_activities': len(model_data.get('activities', [])),
                    'num_relations': len(model_data.get('causality_relations', []))
                },
                'recommendations': [
                    'Investigate noise in sensor data',
                    'Review event detection thresholds',
                    'Consider data filtering approaches'
                ]
            })

        # Activity insights
        activities = model_data.get('activities', [])
        if len(activities) < 3:
            insights.append({
                'type': 'insufficient_activities',
                'category': 'process_model',
                'message': f"Only {len(activities)} distinct activities detected, suggesting incomplete process capture",
                'confidence': 0.7,
                'actionable': True,
                'evidence': {
                    'activity_count': len(activities),
                    'activities': activities
                },
                'recommendations': [
                    'Review sensor coverage and placement',
                    'Lower event detection thresholds',
                    'Check for sensor range limitations'
                ]
            })

        return insights

    def _generate_actionable_insights(self, quality_issues: List[Dict[str, Any]],
                                      process_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights linking quality issues to specific actions"""

        insights = []

        # Group issues by sensor
        sensor_issues = defaultdict(list)
        for issue in quality_issues:
            sensor_id = issue.get('sensor_id', 'unknown')
            sensor_issues[sensor_id].append(issue)

        # Generate sensor-specific recommendations
        for sensor_id, issues in sensor_issues.items():
            if len(issues) > 1:
                issue_types = [issue['type'] for issue in issues]

                # Multiple issues on same sensor
                insights.append({
                    'type': 'sensor_multiple_issues',
                    'category': 'actionable',
                    'message': f"Sensor {sensor_id} has multiple quality issues: {', '.join(set(issue_types))}",
                    'confidence': 0.85,
                    'actionable': True,
                    'evidence': {
                        'sensor_id': sensor_id,
                        'issue_count': len(issues),
                        'issue_types': issue_types
                    },
                    'recommendations': self._get_sensor_specific_recommendations(sensor_id, issues)
                })

        # Configuration-specific insights
        config_insights = self._generate_configuration_insights(quality_issues)
        insights.extend(config_insights)

        return insights

    def _generate_causal_insights(self, quality_issues: List[Dict[str, Any]],
                                  pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate causal insights linking root causes to observed effects"""

        insights = []

        # Analyze causal chains
        causal_chains = self._identify_causal_chains(quality_issues, pipeline_results)

        for chain in causal_chains:
            insights.append({
                'type': 'causal_chain',
                'category': 'causal_analysis',
                'message': f"Causal chain detected: {chain['root_cause']} → {' → '.join(chain['effects'])}",
                'confidence': chain['confidence'],
                'actionable': True,
                'evidence': chain['evidence'],
                'recommendations': [
                    f"Address root cause: {chain['root_cause']}",
                    "Monitor downstream effects during remediation"
                ]
            })

        # System-level insights
        system_insights = self._generate_system_level_insights(quality_issues, pipeline_results)
        insights.extend(system_insights)

        return insights

    def _generate_information_gain_insights(self, quality_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights about information gain from quality issues"""

        insights = []

        # Calculate total information gain
        total_interpretability = sum(issue.get('information_gain', {}).get('interpretability_gain', 0)
                                     for issue in quality_issues)
        total_actionability = sum(issue.get('information_gain', {}).get('actionability_gain', 0)
                                  for issue in quality_issues)

        if total_interpretability > 2.0:  # Threshold for significant insight
            insights.append({
                'type': 'high_interpretability_gain',
                'category': 'information_gain',
                'message': f"Quality issues provide significant interpretability gains (score: {total_interpretability:.1f})",
                'confidence': 0.8,
                'actionable': False,
                'evidence': {
                    'interpretability_score': total_interpretability,
                    'contributing_issues': len([issue for issue in quality_issues
                                                if issue.get('information_gain', {}).get('interpretability_gain',
                                                                                         0) > 0.5])
                },
                'recommendations': [
                    'Leverage quality issues for process understanding',
                    'Document quality patterns for future reference'
                ]
            })

        if total_actionability > 2.0:
            insights.append({
                'type': 'high_actionability_gain',
                'category': 'information_gain',
                'message': f"Quality issues provide clear actionable insights (score: {total_actionability:.1f})",
                'confidence': 0.8,
                'actionable': True,
                'evidence': {
                    'actionability_score': total_actionability
                },
                'recommendations': [
                    'Prioritize issues with high actionability scores',
                    'Implement systematic remediation plan'
                ]
            })

        return insights

    def _identify_causal_chains(self, quality_issues: List[Dict[str, Any]],
                                pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify causal chains from root causes to effects"""

        chains = []

        # Group issues by type to identify patterns
        c1_issues = [issue for issue in quality_issues if issue['type'] == 'C1_inadequate_sampling']
        c3_issues = [issue for issue in quality_issues if issue['type'] == 'C3_sensor_noise']

        # Example causal chain: C1 → missing events → incomplete model
        if c1_issues:
            model_metrics = pipeline_results.get('process_model', {}).get('metrics', {})
            if model_metrics.get('fitness', 1.0) < 0.6:
                chains.append({
                    'root_cause': 'Inadequate sampling rate',
                    'effects': ['Missing short-duration events', 'Incomplete process model', 'Low model fitness'],
                    'confidence': 0.8,
                    'evidence': {
                        'c1_issue_count': len(c1_issues),
                        'model_fitness': model_metrics.get('fitness', 0)
                    }
                })

        # Example causal chain: C3 → noisy events → complex model
        if c3_issues:
            model_metrics = pipeline_results.get('process_model', {}).get('metrics', {})
            if model_metrics.get('complexity', 0) > 0.7:
                chains.append({
                    'root_cause': 'Sensor noise and outliers',
                    'effects': ['False event detection', 'Spaghetti process model', 'High model complexity'],
                    'confidence': 0.75,
                    'evidence': {
                        'c3_issue_count': len(c3_issues),
                        'model_complexity': model_metrics.get('complexity', 0)
                    }
                })

        return chains

    def _generate_system_level_insights(self, quality_issues: List[Dict[str, Any]],
                                        pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system-level insights"""

        insights = []

        # Overall system health
        total_issues = len(quality_issues)
        high_severity_count = len([issue for issue in quality_issues
                                   if issue.get('severity') == 'high'])

        if high_severity_count > total_issues * 0.5:
            insights.append({
                'type': 'system_health_critical',
                'category': 'system_analysis',
                'message': f"System health is critical: {high_severity_count}/{total_issues} issues are high severity",
                'confidence': 0.9,
                'actionable': True,
                'evidence': {
                    'high_severity_ratio': high_severity_count / total_issues,
                    'total_issues': total_issues
                },
                'recommendations': [
                    'Immediate system maintenance required',
                    'Review sensor calibration and placement',
                    'Implement enhanced monitoring'
                ]
            })

        # Data pipeline effectiveness
        model_metrics = pipeline_results.get('process_model', {}).get('metrics', {})
        overall_quality = (model_metrics.get('fitness', 0) +
                           (1 - model_metrics.get('complexity', 1)) +
                           model_metrics.get('precision', 0)) / 3

        if overall_quality < 0.5:
            insights.append({
                'type': 'pipeline_effectiveness_low',
                'category': 'system_analysis',
                'message': f"Data pipeline effectiveness is low (score: {overall_quality:.2f})",
                'confidence': 0.8,
                'actionable': True,
                'evidence': {
                    'effectiveness_score': overall_quality,
                    'component_scores': model_metrics
                },
                'recommendations': [
                    'Review and tune pipeline parameters',
                    'Implement quality-aware processing',
                    'Consider alternative algorithms for noisy environments'
                ]
            })

        return insights

    def _get_recommendations_for_issue(self, issue_type: str) -> List[str]:
        """Get specific recommendations for an issue type"""

        recommendations = {
            'C1_inadequate_sampling': [
                'Increase sensor sampling rates',
                'Review sensor configuration files',
                'Consider upgrading to faster sensors',
                'Implement adaptive sampling strategies'
            ],
            'C2_poor_placement': [
                'Review sensor placement and coverage',
                'Check for sensor interference',
                'Validate sensor mounting and positioning',
                'Consider additional sensors for better coverage'
            ],
            'C3_sensor_noise': [
                'Check sensor calibration and maintenance',
                'Implement noise filtering algorithms',
                'Review electrical interference sources',
                'Consider sensor replacement if degraded'
            ],
            'C4_range_too_small': [
                'Upgrade sensors with larger measurement ranges',
                'Review process requirements vs sensor specifications',
                'Implement range extension techniques',
                'Consider multiple sensors for full range coverage'
            ],
            'C5_high_volume': [
                'Optimize data processing infrastructure',
                'Implement data compression and buffering',
                'Review network bandwidth and latency',
                'Consider edge computing solutions'
            ]
        }

        return recommendations.get(issue_type, ['Review sensor configuration', 'Consult technical documentation'])

    def _get_sensor_specific_recommendations(self, sensor_id: str, issues: List[Dict[str, Any]]) -> List[str]:
        """Get sensor-specific recommendations"""

        recommendations = [f'Prioritize maintenance for sensor {sensor_id}']

        # Add issue-specific recommendations
        issue_types = set(issue['type'] for issue in issues)

        if 'C1_inadequate_sampling' in issue_types:
            recommendations.append('Update sensor sampling configuration')
        if 'C3_sensor_noise' in issue_types:
            recommendations.append('Calibrate sensor and check for interference')
        if 'C4_range_too_small' in issue_types:
            recommendations.append('Verify sensor range matches process requirements')

        return recommendations

    def _generate_configuration_insights(self, quality_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate configuration-related insights"""

        insights = []

        # Check for sampling rate issues across multiple sensors
        sampling_issues = [issue for issue in quality_issues
                           if issue['type'] == 'C1_inadequate_sampling']

        if len(sampling_issues) > 1:
            sensors_affected = set(issue.get('sensor_id', '') for issue in sampling_issues)
            insights.append({
                'type': 'widespread_sampling_issues',
                'category': 'configuration',
                'message': f"Sampling rate issues detected across {len(sensors_affected)} sensors",
                'confidence': 0.85,
                'actionable': True,
                'evidence': {
                    'affected_sensors': list(sensors_affected),
                    'issue_count': len(sampling_issues)
                },
                'recommendations': [
                    'Review global sampling configuration',
                    'Implement centralized sensor management',
                    'Check system clock synchronization'
                ]
            })

        return insights

    def _rank_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank insights by importance and actionability"""

        def calculate_importance_score(insight):
            score = insight.get('confidence', 0.5)

            # Boost actionable insights
            if insight.get('actionable', False):
                score += 0.2

            # Boost high-severity insights
            if 'critical' in insight.get('message', '').lower():
                score += 0.3
            elif 'high' in insight.get('message', '').lower():
                score += 0.1

            # Boost causal insights
            if insight.get('category') == 'causal_analysis':
                score += 0.15

            return score

        # Sort by importance score (descending)
        ranked = sorted(insights, key=calculate_importance_score, reverse=True)

        # Add ranking information
        for i, insight in enumerate(ranked):
            insight['rank'] = i + 1
            insight['importance_score'] = calculate_importance_score(insight)

        return ranked

    def _initialize_insight_templates(self) -> Dict[str, Any]:
        """Initialize templates for different types of insights"""

        return {
            'quality_issue_template': {
                'type': '',
                'category': 'data_quality',
                'message': '',
                'confidence': 0.5,
                'actionable': False,
                'evidence': {},
                'recommendations': []
            },
            'model_insight_template': {
                'type': '',
                'category': 'process_model',
                'message': '',
                'confidence': 0.5,
                'actionable': False,
                'evidence': {},
                'recommendations': []
            },
            'system_insight_template': {
                'type': '',
                'category': 'system_analysis',
                'message': '',
                'confidence': 0.5,
                'actionable': True,
                'evidence': {},
                'recommendations': []
            }
        }