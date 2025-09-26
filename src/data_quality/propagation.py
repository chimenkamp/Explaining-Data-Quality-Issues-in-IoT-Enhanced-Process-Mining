import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta


class QualityPropagator:
    """Propagates quality issues through the data pipeline"""

    def __init__(self):
        self.propagation_rules = self._initialize_propagation_rules()

    def propagate_issues(self, classified_issues: List[Dict[str, Any]],
                         pipeline_stage: str, stage_data: Any) -> Dict[str, Any]:
        """Propagate quality issues to the next pipeline stage"""

        propagated_issues = []
        stage_signatures = {}

        for issue in classified_issues:
            propagated_issue = self._propagate_single_issue(
                issue, pipeline_stage, stage_data
            )
            propagated_issues.append(propagated_issue)

            # Extract signatures for this stage
            if 'signatures' in issue:
                stage_key = self._get_stage_key(pipeline_stage)
                if stage_key in issue['signatures']:
                    signature = issue['signatures'][stage_key]
                    if signature not in stage_signatures:
                        stage_signatures[signature] = []
                    stage_signatures[signature].append(issue)

        return {
            'propagated_issues': propagated_issues,
            'stage_signatures': stage_signatures,
            'stage': pipeline_stage
        }

    def _propagate_single_issue(self, issue: Dict[str, Any],
                                stage: str, stage_data: Any) -> Dict[str, Any]:
        """Propagate a single quality issue through pipeline stage"""

        propagated_issue = issue.copy()

        # Add stage-specific effects
        stage_effects = self._calculate_stage_effects(issue, stage, stage_data)
        propagated_issue['stage_effects'] = stage_effects

        # Update probability based on propagation
        original_prob = propagated_issue.get('posterior_probability', 0.5)
        propagation_factor = stage_effects.get('propagation_factor', 1.0)

        propagated_issue['propagated_probability'] = min(1.0, original_prob * propagation_factor)

        # Add information gain metrics
        info_gain = self._calculate_information_gain(issue, stage, stage_data)
        propagated_issue['information_gain'] = info_gain

        return propagated_issue

    def _calculate_stage_effects(self, issue: Dict[str, Any],
                                 stage: str, stage_data: Any) -> Dict[str, Any]:
        """Calculate how the issue affects this pipeline stage"""

        effects = {}
        issue_type = issue['type']

        if stage == 'preprocessing':
            effects.update(self._preprocessing_effects(issue_type, stage_data))
        elif stage == 'event_abstraction':
            effects.update(self._event_abstraction_effects(issue_type, stage_data))
        elif stage == 'case_correlation':
            effects.update(self._case_correlation_effects(issue_type, stage_data))
        elif stage == 'process_mining':
            effects.update(self._process_mining_effects(issue_type, stage_data))
        elif stage == 'visualization':
            effects.update(self._visualization_effects(issue_type, stage_data))

        return effects

    def _preprocessing_effects(self, issue_type: str, stage_data: Any) -> Dict[str, Any]:
        """Effects on preprocessing stage"""
        effects = {}

        if issue_type == 'C1_inadequate_sampling':
            effects = {
                'missing_samples': True,
                'interpolation_artifacts': True,
                'propagation_factor': 1.1
            }
        elif issue_type == 'C2_poor_placement':
            effects = {
                'data_inconsistency': True,
                'filtering_challenges': True,
                'propagation_factor': 1.2
            }
        elif issue_type == 'C3_sensor_noise':
            effects = {
                'noise_amplification': True,
                'outlier_propagation': True,
                'propagation_factor': 1.3
            }
        elif issue_type == 'C4_range_too_small':
            effects = {
                'information_loss': True,
                'clipping_artifacts': True,
                'propagation_factor': 1.2
            }
        elif issue_type == 'C5_high_volume':
            effects = {
                'processing_delays': True,
                'buffer_overflows': True,
                'propagation_factor': 1.1
            }

        return effects

    def _event_abstraction_effects(self, issue_type: str, stage_data: Any) -> Dict[str, Any]:
        """Effects on event abstraction stage"""
        effects = {}

        if issue_type == 'C1_inadequate_sampling':
            effects = {
                'missing_events': True,
                'event_merging': True,
                'temporal_distortion': True,
                'propagation_factor': 1.4
            }
        elif issue_type == 'C2_poor_placement':
            effects = {
                'incorrect_event_boundaries': True,
                'overlapping_events': True,
                'wrong_activity_labels': True,
                'propagation_factor': 1.5
            }
        elif issue_type == 'C3_sensor_noise':
            effects = {
                'false_events': True,
                'event_fragmentation': True,
                'incorrect_classifications': True,
                'propagation_factor': 1.6
            }
        elif issue_type == 'C4_range_too_small':
            effects = {
                'activity_masking': True,
                'missing_event_types': True,
                'oversimplified_events': True,
                'propagation_factor': 1.3
            }
        elif issue_type == 'C5_high_volume':
            effects = {
                'event_ordering_issues': True,
                'timestamp_drift': True,
                'incomplete_events': True,
                'propagation_factor': 1.2
            }

        return effects

    def _case_correlation_effects(self, issue_type: str, stage_data: Any) -> Dict[str, Any]:
        """Effects on case correlation stage"""
        effects = {}

        if issue_type == 'C1_inadequate_sampling':
            effects = {
                'case_fragmentation': True,
                'missing_case_relationships': True,
                'propagation_factor': 1.3
            }
        elif issue_type == 'C2_poor_placement':
            effects = {
                'wrong_case_assignments': True,
                'case_merging_errors': True,
                'propagation_factor': 1.4
            }
        elif issue_type == 'C3_sensor_noise':
            effects = {
                'spurious_cases': True,
                'case_splitting_errors': True,
                'propagation_factor': 1.5
            }
        elif issue_type == 'C4_range_too_small':
            effects = {
                'incomplete_cases': True,
                'case_boundary_errors': True,
                'propagation_factor': 1.2
            }
        elif issue_type == 'C5_high_volume':
            effects = {
                'case_ordering_problems': True,
                'concurrent_case_confusion': True,
                'propagation_factor': 1.3
            }

        return effects

    def _process_mining_effects(self, issue_type: str, stage_data: Any) -> Dict[str, Any]:
        """Effects on process mining stage"""
        effects = {}

        if issue_type == 'C1_inadequate_sampling':
            effects = {
                'oversimplified_model': True,
                'missing_process_paths': True,
                'incorrect_dependencies': True,
                'propagation_factor': 1.5
            }
        elif issue_type == 'C2_poor_placement':
            effects = {
                'wrong_process_structure': True,
                'parallel_artifacts': True,
                'propagation_factor': 1.6
            }
        elif issue_type == 'C3_sensor_noise':
            effects = {
                'spaghetti_model': True,
                'low_fitness': True,
                'excessive_complexity': True,
                'propagation_factor': 1.8
            }
        elif issue_type == 'C4_range_too_small':
            effects = {
                'incomplete_model': True,
                'missing_activities': True,
                'propagation_factor': 1.4
            }
        elif issue_type == 'C5_high_volume':
            effects = {
                'model_instability': True,
                'convergence_issues': True,
                'propagation_factor': 1.3
            }

        return effects

    def _visualization_effects(self, issue_type: str, stage_data: Any) -> Dict[str, Any]:
        """Effects on visualization stage"""
        effects = {}

        # All issues can affect visualization clarity
        effects = {
            'visualization_complexity': True,
            'interpretation_difficulty': True,
            'misleading_representations': True,
            'propagation_factor': 1.2
        }

        if issue_type == 'C3_sensor_noise':
            effects['visual_clutter'] = True
            effects['propagation_factor'] = 1.4

        return effects

    def _calculate_information_gain(self, issue: Dict[str, Any],
                                    stage: str, stage_data: Any) -> Dict[str, Any]:
        """Calculate information gain from the quality issue"""

        info_gain = {
            'interpretability_gain': 0.0,
            'actionability_gain': 0.0,
            'explainability_gain': 0.0
        }

        issue_type = issue['type']
        confidence = issue.get('confidence', 0.5)

        # Higher confidence issues provide more information gain
        base_gain = confidence * 0.8

        # Different issue types provide different types of insights
        if issue_type == 'C1_inadequate_sampling':
            info_gain['interpretability_gain'] = base_gain * 0.9  # High interpretability
            info_gain['actionability_gain'] = base_gain * 0.8  # High actionability
            info_gain['explainability_gain'] = base_gain * 0.7

        elif issue_type == 'C2_poor_placement':
            info_gain['interpretability_gain'] = base_gain * 0.7
            info_gain['actionability_gain'] = base_gain * 0.9  # Very actionable
            info_gain['explainability_gain'] = base_gain * 0.8

        elif issue_type == 'C3_sensor_noise':
            info_gain['interpretability_gain'] = base_gain * 0.6
            info_gain['actionability_gain'] = base_gain * 0.6
            info_gain['explainability_gain'] = base_gain * 0.9  # High explainability

        elif issue_type == 'C4_range_too_small':
            info_gain['interpretability_gain'] = base_gain * 0.8
            info_gain['actionability_gain'] = base_gain * 0.7
            info_gain['explainability_gain'] = base_gain * 0.8

        elif issue_type == 'C5_high_volume':
            info_gain['interpretability_gain'] = base_gain * 0.5
            info_gain['actionability_gain'] = base_gain * 0.8
            info_gain['explainability_gain'] = base_gain * 0.6

        return info_gain

    def _get_stage_key(self, pipeline_stage: str) -> str:
        """Map pipeline stage to signature key"""
        stage_mapping = {
            'preprocessing': 'raw_data',
            'event_abstraction': 'events',
            'case_correlation': 'process_instances',
            'process_mining': 'process_model',
            'visualization': 'visualization'
        }
        return stage_mapping.get(pipeline_stage, 'unknown')

    def _initialize_propagation_rules(self) -> Dict[str, Any]:
        """Initialize rules for how issues propagate through stages"""
        return {
            'C1_inadequate_sampling': {
                'amplification_stages': ['event_abstraction', 'process_mining'],
                'damping_stages': ['preprocessing'],
                'critical_stages': ['event_abstraction']
            },
            'C2_poor_placement': {
                'amplification_stages': ['case_correlation', 'process_mining'],
                'damping_stages': [],
                'critical_stages': ['case_correlation']
            },
            'C3_sensor_noise': {
                'amplification_stages': ['event_abstraction', 'process_mining'],
                'damping_stages': ['preprocessing'],
                'critical_stages': ['process_mining']
            },
            'C4_range_too_small': {
                'amplification_stages': ['event_abstraction'],
                'damping_stages': [],
                'critical_stages': ['event_abstraction']
            },
            'C5_high_volume': {
                'amplification_stages': ['case_correlation'],
                'damping_stages': ['preprocessing'],
                'critical_stages': ['case_correlation']
            }
        }