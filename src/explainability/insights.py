import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta


class InsightGenerator:
    """Generates explainable insights from quality issues and pipeline results
    
    Follows the BNF template structure with:
    - Header (ID, Process Area, Business Goal, Analysis Type, Relevance)
    - Contextual Explanation (Observed Behavior, Data Scope, Pipeline Notes, Context)
    - Narrative Layer (Executive Summary, Analyst Detail)
    - Action Mapping (Recommended Actions with responsible actors)
    - Confidence & Affect (Confidence Level, User Guidance, Interpretation Note)
    - References (Source Log, Version, Last Update, Contact)
    """

    def __init__(self):
        self.insight_templates = self._initialize_insight_templates()
        self.insight_counter = 0

    def generate_insights(self, pipeline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights from pipeline results following BNF template"""

        insights = []

        quality_issues = pipeline_results.get('quality_issues', [])
        process_model = pipeline_results.get('process_model', {})
        case_instances = pipeline_results.get('process_instances', pd.DataFrame())
        
        # Extract metadata for references
        metadata = pipeline_results.get('metadata', {})
        log_id = metadata.get('log_id', 'UNKNOWN')
        timeframe = metadata.get('timeframe', 'Not specified')
        analyst_contact = metadata.get('analyst_contact', 'system-generated')

        # Generate insights for each category
        insights.extend(self._generate_quality_insights(quality_issues, log_id, timeframe, analyst_contact))
        insights.extend(self._generate_model_insights(process_model, log_id, timeframe, analyst_contact))
        insights.extend(self._generate_actionable_insights(quality_issues, process_model, log_id, timeframe, analyst_contact))
        insights.extend(self._generate_causal_insights(quality_issues, pipeline_results, log_id, timeframe, analyst_contact))
        insights.extend(self._generate_conformance_insights(pipeline_results, log_id, timeframe, analyst_contact))

        # Rank insights by importance
        ranked_insights = self._rank_insights(insights)
        
        # Format insights according to BNF template
        formatted_insights = [self._format_insight_to_bnf(insight) for insight in ranked_insights]

        return formatted_insights

    def _generate_quality_insights(self, quality_issues: List[Dict[str, Any]], 
                                   log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
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
            'recommendations': self._get_recommendations_for_issue(most_common_issue[0]),
            'log_id': log_id,
            'timeframe': timeframe,
            'analyst_contact': analyst_contact
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
                                    'Implement monitoring for detected patterns'],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
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
                'recommendations': ['Urgent system review required', 'Check sensor maintenance schedules'],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
            })

        return insights

    def _generate_model_insights(self, process_model: Dict[str, Any],
                                 log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
            })

        return insights

    def _generate_actionable_insights(self, quality_issues: List[Dict[str, Any]],
                                      process_model: Dict[str, Any],
                                      log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
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
                    'recommendations': self._get_sensor_specific_recommendations(sensor_id, issues),
                    'log_id': log_id,
                    'timeframe': timeframe,
                    'analyst_contact': analyst_contact
                })

        # Configuration-specific insights
        config_insights = self._generate_configuration_insights(quality_issues, log_id, timeframe, analyst_contact)
        insights.extend(config_insights)

        return insights

    def _generate_causal_insights(self, quality_issues: List[Dict[str, Any]],
                                  pipeline_results: Dict[str, Any],
                                  log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
            })

        # System-level insights
        system_insights = self._generate_system_level_insights(quality_issues, pipeline_results, log_id, timeframe, analyst_contact)
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
                                        pipeline_results: Dict[str, Any],
                                        log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
            })

        return insights
    
    def _generate_conformance_insights(self, pipeline_results: Dict[str, Any],
                                       log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
        """Generate insights from conformance checking results"""
        insights = []
        
        conformance_data = pipeline_results.get('conformance_results', {})
        
        # Check if conformance_data is empty (handle both dict and DataFrame)
        if isinstance(conformance_data, dict) and len(conformance_data) == 0:
            return insights
        if isinstance(conformance_data, pd.DataFrame) and len(conformance_data) == 0:
            return insights
        if conformance_data is None:
            return insights
        
        # Check if we have DataFrame or list
        if isinstance(conformance_data, pd.DataFrame):
            conformance_df = conformance_data
        elif isinstance(conformance_data, list) and len(conformance_data) > 0:
            conformance_df = pd.DataFrame(conformance_data)
        else:
            return insights
        
        if len(conformance_df) == 0:
            return insights
        
        # Analyze conformance patterns
        if 'fitness' in conformance_df.columns:
            avg_fitness = conformance_df['fitness'].mean()
            low_fitness_cases = conformance_df[conformance_df['fitness'] < 0.5]
            
            if len(low_fitness_cases) > 0:
                insights.append({
                    'type': 'conformance_deviation',
                    'category': 'conformance_analysis',
                    'message': f"{len(low_fitness_cases)} cases show low conformance (fitness < 0.5)",
                    'confidence': 0.85,
                    'actionable': True,
                    'evidence': {
                        'low_fitness_count': len(low_fitness_cases),
                        'average_fitness': avg_fitness,
                        'affected_cases': low_fitness_cases['case_id'].tolist() if 'case_id' in low_fitness_cases.columns else []
                    },
                    'recommendations': [
                        'Investigate deviating cases for quality issues',
                        'Review process model accuracy',
                        'Check for process variants not captured in model'
                    ],
                    'log_id': log_id,
                    'timeframe': timeframe,
                    'analyst_contact': analyst_contact
                })
        
        # Correlation between quality issues and conformance
        if 'quality_issues' in conformance_df.columns and 'fitness' in conformance_df.columns:
            correlation = conformance_df[['quality_issues', 'fitness']].corr().iloc[0, 1]
            
            if abs(correlation) > 0.3:
                insights.append({
                    'type': 'quality_conformance_correlation',
                    'category': 'conformance_analysis',
                    'message': f"{'Strong negative' if correlation < 0 else 'Positive'} correlation ({correlation:.2f}) between quality issues and conformance",
                    'confidence': 0.8,
                    'actionable': True,
                    'evidence': {
                        'correlation': correlation,
                        'interpretation': 'More quality issues lead to lower conformance' if correlation < 0 else 'Quality issues do not directly impact conformance'
                    },
                    'recommendations': [
                        'Focus on quality improvement to enhance conformance' if correlation < 0 else 'Investigate other factors affecting conformance',
                        'Implement quality-aware conformance checking'
                    ],
                    'log_id': log_id,
                    'timeframe': timeframe,
                    'analyst_contact': analyst_contact
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

    def _generate_configuration_insights(self, quality_issues: List[Dict[str, Any]],
                                         log_id: str, timeframe: str, analyst_contact: str) -> List[Dict[str, Any]]:
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
                ],
                'log_id': log_id,
                'timeframe': timeframe,
                'analyst_contact': analyst_contact
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
    
    def _format_insight_to_bnf(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Format insight according to BNF template structure"""
        
        self.insight_counter += 1
        insight_id = f"IOTPM-{self.insight_counter:03d}"
        
        # Determine process area based on category
        process_area_mapping = {
            'data_quality': 'IoT Data Quality & Event Abstraction',
            'process_model': 'Process Discovery & Model Quality',
            'causal_analysis': 'Root Cause Analysis & Traceability',
            'conformance_analysis': 'Conformance Checking & Deviation Analysis',
            'actionable': 'System Configuration & Optimization',
            'system_analysis': 'System Health & Performance'
        }
        
        process_area = process_area_mapping.get(insight.get('category', 'system_analysis'), 
                                                 'IoT-Enhanced Process Mining')
        
        # Determine analysis type
        analysis_type = "Directed" if insight.get('actionable', False) else "Exploratory"
        
        # Format according to BNF structure
        formatted = {
            # --- HEADER ---
            'header': {
                'insight_id': insight_id,
                'process_area': process_area,
                'business_goal': insight.get('business_goal', self._infer_business_goal(insight)),
                'analysis_type': analysis_type,
                'relevance': insight.get('message', '')
            },
            
            # --- CONTEXTUAL EXPLANATION ---
            'contextual_explanation': {
                'observed_behavior': insight.get('message', ''),
                'data_scope': f"Event log: {insight.get('log_id', 'UNKNOWN')}, timeframe: {insight.get('timeframe', 'Not specified')}",
                'pipeline_notes': insight.get('pipeline_notes', self._generate_pipeline_notes(insight)),
                'business_context': insight.get('business_context', self._generate_business_context(insight))
            },
            
            # --- NARRATIVE LAYER ---
            'narrative': {
                'executive_summary': {
                    'summary': self._generate_executive_summary(insight),
                    'impact': self._generate_impact_metric(insight),
                    'key_takeaway': self._generate_key_takeaway(insight)
                },
                'analyst_detail': {
                    'deep_dive': insight.get('evidence', {}),
                    'linked_kpis': self._extract_kpis(insight),
                    'supporting_examples': insight.get('supporting_examples', 'See evidence data')
                }
            },
            
            # --- ACTION MAPPING ---
            'action_mapping': {
                'recommended_actions': self._format_recommendations(insight.get('recommendations', []))
            },
            
            # --- CONFIDENCE & AFFECT ---
            'confidence_and_affect': {
                'confidence_level': insight.get('confidence', 0.5),
                'confidence_type': self._determine_confidence_type(insight),
                'user_guidance': self._determine_user_guidance(insight),
                'interpretation_note': self._generate_interpretation_note(insight)
            },
            
            # --- REFERENCES ---
            'references': {
                'source_log': insight.get('log_id', 'UNKNOWN'),
                'system_origin': insight.get('system_origin', 'IoT Sensor Framework'),
                'version': insight.get('version', 'v1.0'),
                'last_update': datetime.now().strftime('%Y-%m-%d'),
                'analyst_contact': insight.get('analyst_contact', 'system-generated')
            },
            
            # Keep original data for backward compatibility
            'original_insight': insight
        }
        
        return formatted
    
    def _infer_business_goal(self, insight: Dict[str, Any]) -> str:
        """Infer business goal from insight type and category"""
        category = insight.get('category', '')
        insight_type = insight.get('type', '')
        
        goal_mapping = {
            'data_quality': 'Improve data quality and reduce sensor-related errors in process event logs',
            'process_model': 'Enhance process model accuracy and discover true process behavior',
            'causal_analysis': 'Identify root causes of process deviations and quality issues',
            'conformance_analysis': 'Detect and explain conformance violations in IoT-enhanced processes',
            'actionable': 'Optimize sensor configuration and system performance',
            'system_analysis': 'Maintain system health and ensure reliable process monitoring'
        }
        
        return goal_mapping.get(category, 'Improve IoT-enhanced process mining outcomes')
    
    def _generate_pipeline_notes(self, insight: Dict[str, Any]) -> str:
        """Generate pipeline processing notes"""
        evidence = insight.get('evidence', {})
        
        notes = []
        if 'total_issues' in evidence:
            notes.append(f"Analyzed {evidence['total_issues']} quality issues")
        if 'affected_sensors' in evidence:
            notes.append(f"Affected sensors: {', '.join(evidence['affected_sensors'])}")
        
        if notes:
            return '; '.join(notes)
        
        return "Standard pipeline processing with quality propagation and conformance checking"
    
    def _generate_business_context(self, insight: Dict[str, Any]) -> str:
        """Generate business context explanation"""
        category = insight.get('category', '')
        
        context_mapping = {
            'data_quality': 'Quality issues at sensor level propagate through event abstraction to affect process discovery',
            'process_model': 'Process model quality directly impacts the reliability of process analysis and optimization',
            'causal_analysis': 'Understanding causal chains enables targeted remediation at the root cause level',
            'conformance_analysis': 'Conformance deviations indicate process behavior that deviates from expected patterns',
            'actionable': 'Configuration changes at infrastructure level can prevent future quality issues',
            'system_analysis': 'Overall system health affects the trustworthiness of all process mining results'
        }
        
        return context_mapping.get(category, 'This insight relates to IoT-enhanced process mining quality')
    
    def _generate_executive_summary(self, insight: Dict[str, Any]) -> str:
        """Generate concise executive summary"""
        message = insight.get('message', '')
        # Take first sentence or up to 150 characters
        if '. ' in message:
            return message.split('. ')[0] + '.'
        return message[:150] + '...' if len(message) > 150 else message
    
    def _generate_impact_metric(self, insight: Dict[str, Any]) -> str:
        """Generate impact metric description"""
        evidence = insight.get('evidence', {})
        
        # Try to extract quantitative impact
        if 'total_issues' in evidence:
            return f"{evidence['total_issues']} quality issues detected"
        elif 'fitness_score' in evidence:
            return f"Fitness score: {evidence['fitness_score']:.2f}"
        elif 'model_complexity' in evidence:
            return f"Model complexity: {evidence['model_complexity']:.2f}"
        elif 'affected_cases' in evidence:
            return f"{evidence['affected_cases']} cases affected"
        
        confidence = insight.get('confidence', 0.5)
        return f"Confidence level: {confidence:.1%}"
    
    def _generate_key_takeaway(self, insight: Dict[str, Any]) -> str:
        """Generate key takeaway message"""
        insight_type = insight.get('type', '')
        
        takeaway_mapping = {
            'quality_pattern': 'Focus remediation efforts on most prevalent quality issue type',
            'high_confidence_issues': 'Immediate action required on high-confidence detections',
            'low_fitness': 'Model does not accurately represent observed process behavior',
            'high_complexity': 'Excessive model complexity indicates noise or quality problems',
            'causal_chain': 'Root cause identified - address at source for maximum impact',
            'conformance_deviation': 'Process deviations linked to specific quality issues',
            'sensor_multiple_issues': 'Single sensor requires comprehensive maintenance',
            'system_health_critical': 'System-wide intervention needed to restore quality'
        }
        
        return takeaway_mapping.get(insight_type, 'Action needed to improve process mining quality')
    
    def _extract_kpis(self, insight: Dict[str, Any]) -> List[str]:
        """Extract relevant KPIs from insight"""
        kpis = []
        evidence = insight.get('evidence', {})
        category = insight.get('category', '')
        
        if category == 'data_quality':
            kpis.extend(['Data Quality Score', 'Event Completeness', 'Sensor Reliability'])
        elif category == 'process_model':
            kpis.extend(['Model Fitness', 'Model Precision', 'Model Complexity'])
        elif category == 'conformance_analysis':
            kpis.extend(['Conformance Rate', 'Deviation Count', 'Token-Based Fitness'])
        
        return kpis
    
    def _format_recommendations(self, recommendations: List[str]) -> List[Dict[str, Any]]:
        """Format recommendations with responsible actors and priorities"""
        formatted_recs = []
        
        # Define responsibility mapping
        responsibility_mapping = {
            'sensor': 'IoT Infrastructure Team',
            'calibrat': 'Maintenance Team',
            'configuration': 'System Administrator',
            'algorithm': 'Data Science Team',
            'process': 'Process Owner',
            'model': 'Process Mining Analyst',
            'review': 'Process Owner',
            'monitor': 'Operations Team',
            'implement': 'IT Development Team'
        }
        
        for i, rec in enumerate(recommendations):
            # Determine responsible actor
            rec_lower = rec.lower()
            responsible = 'Process Owner'  # Default
            
            for keyword, actor in responsibility_mapping.items():
                if keyword in rec_lower:
                    responsible = actor
                    break
            
            # Determine priority
            priority = 'High' if i == 0 else ('Medium' if i < 3 else 'Low')
            if 'urgent' in rec_lower or 'immediate' in rec_lower or 'critical' in rec_lower:
                priority = 'High'
            
            formatted_recs.append({
                'action': rec,
                'responsible': responsible,
                'priority': priority,
                'expected_outcome': self._infer_expected_outcome(rec)
            })
        
        return formatted_recs
    
    def _infer_expected_outcome(self, recommendation: str) -> str:
        """Infer expected outcome from recommendation"""
        rec_lower = recommendation.lower()
        
        if 'increase' in rec_lower or 'improve' in rec_lower:
            return 'Improved data quality and reduced issues'
        elif 'calibrat' in rec_lower:
            return 'More accurate sensor readings'
        elif 'review' in rec_lower or 'check' in rec_lower:
            return 'Identification of root cause'
        elif 'implement' in rec_lower:
            return 'Systematic prevention of future issues'
        elif 'upgrade' in rec_lower:
            return 'Enhanced system capabilities'
        
        return 'Reduced quality issues and improved process visibility'
    
    def _determine_confidence_type(self, insight: Dict[str, Any]) -> str:
        """Determine type of confidence score"""
        evidence = insight.get('evidence', {})
        
        if 'fitness_score' in evidence or 'model_complexity' in evidence:
            return 'Model-based'
        elif len(evidence) > 3:
            return 'Derived from data variance'
        else:
            return 'Heuristic'
    
    def _determine_user_guidance(self, insight: Dict[str, Any]) -> str:
        """Determine user guidance based on confidence and severity"""
        confidence = insight.get('confidence', 0.5)
        severity = insight.get('severity', 'medium')
        
        if confidence > 0.8:
            return 'High trust'
        elif confidence > 0.6 and severity != 'high':
            return 'Moderate confidence'
        else:
            return 'Caution advised'
    
    def _generate_interpretation_note(self, insight: Dict[str, Any]) -> str:
        """Generate interpretation guidance note"""
        confidence = insight.get('confidence', 0.5)
        evidence = insight.get('evidence', {})
        
        notes = []
        
        if confidence > 0.8:
            notes.append('High confidence detection based on robust evidence')
        elif confidence < 0.6:
            notes.append('Exploratory finding requiring validation')
        
        if 'total_issues' in evidence and evidence['total_issues'] > 100:
            notes.append('based on large sample size')
        
        if len(evidence) < 2:
            notes.append('limited evidence available')
        
        if notes:
            return '; '.join(notes).capitalize()
        
        return 'Standard confidence level with typical evidence quality'
    
    @staticmethod
    def format_insight_for_display(insight: Dict[str, Any], extended: bool = False) -> str:
        """Format a BNF-structured insight for human-readable display
        
        Args:
            insight: BNF-formatted insight dictionary
            extended: If True, include analyst details (extended format)
        
        Returns:
            YAML-like formatted string
        """
        
        lines = []
        
        # Header section
        header = insight.get('header', {})
        lines.append(f"Insight ID: {header.get('insight_id', 'UNKNOWN')}")
        lines.append(f"Process Area: {header.get('process_area', 'Unknown')}")
        lines.append(f"Business Goal: {header.get('business_goal', 'Not specified')}")
        lines.append(f"Analysis Type: {header.get('analysis_type', 'Exploratory')}")
        lines.append(f"Relevance: {header.get('relevance', 'Not specified')}")
        lines.append("")
        
        # Contextual Explanation
        context = insight.get('contextual_explanation', {})
        lines.append(f"Observed Behavior: {context.get('observed_behavior', 'Not specified')}")
        lines.append(f"Underlying Data Scope: {context.get('data_scope', 'Not specified')}")
        lines.append(f"Applied Filters / Pipeline Notes: {context.get('pipeline_notes', 'Standard processing')}")
        lines.append(f"Context: {context.get('business_context', 'Not specified')}")
        lines.append("")
        
        # Narrative Layer - Executive Summary
        narrative = insight.get('narrative', {})
        exec_summary = narrative.get('executive_summary', {})
        lines.append(f"Summary: {exec_summary.get('summary', 'Not specified')}")
        lines.append(f"Impact: {exec_summary.get('impact', 'Not quantified')}")
        lines.append(f"Key Takeaway: {exec_summary.get('key_takeaway', 'Not specified')}")
        lines.append("")
        
        # Analyst Details (if extended)
        if extended:
            analyst_detail = narrative.get('analyst_detail', {})
            lines.append("Analyst Notes:")
            deep_dive = analyst_detail.get('deep_dive', {})
            if isinstance(deep_dive, dict):
                for key, value in deep_dive.items():
                    lines.append(f"   {key}: {value}")
            else:
                lines.append(f"   {deep_dive}")
            
            kpis = analyst_detail.get('linked_kpis', [])
            if kpis:
                lines.append(f"Linked KPIs: {', '.join(kpis)}")
            
            examples = analyst_detail.get('supporting_examples', '')
            if examples:
                lines.append(f"Supporting Examples: {examples}")
            lines.append("")
        
        # Action Mapping
        lines.append("Recommended Actions:")
        actions = insight.get('action_mapping', {}).get('recommended_actions', [])
        for action in actions:
            action_text = action.get('action', 'Not specified')
            responsible = action.get('responsible', 'Not assigned')
            priority = action.get('priority', 'Medium')
            outcome = action.get('expected_outcome', '')
            
            lines.append(f"- {action_text}")
            lines.append(f"  (Responsible: {responsible}, Priority: {priority})")
            if outcome:
                lines.append(f"  Expected Outcome: {outcome}")
        lines.append("")
        
        # Confidence & Affect
        confidence = insight.get('confidence_and_affect', {})
        conf_level = confidence.get('confidence_level', 0.5)
        conf_type = confidence.get('confidence_type', 'Heuristic')
        lines.append(f"Confidence Level: {conf_level:.0%} ({conf_type})")
        lines.append(f"User Guidance: {confidence.get('user_guidance', 'Review recommended')}")
        lines.append(f"Interpretation Note: {confidence.get('interpretation_note', 'Standard analysis')}")
        lines.append("")
        
        # References
        refs = insight.get('references', {})
        lines.append(f"Source Log: {refs.get('source_log', 'UNKNOWN')} ({refs.get('system_origin', 'System')})")
        lines.append(f"Version: {refs.get('version', 'v1.0')}")
        lines.append(f"Last Update: {refs.get('last_update', 'Unknown')}")
        lines.append(f"Analyst/Producer Contact: {refs.get('analyst_contact', 'Not specified')}")
        
        return '\n'.join(lines)
