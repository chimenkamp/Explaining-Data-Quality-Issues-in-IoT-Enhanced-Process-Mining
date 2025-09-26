import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class QualityClassifier:
    """Classifies and assigns probabilities to quality issues"""

    def __init__(self):
        self.root_cause_priors = {
            'C1_inadequate_sampling': 0.2,
            'C2_poor_placement': 0.15,
            'C3_sensor_noise': 0.3,
            'C4_range_too_small': 0.2,
            'C5_high_volume': 0.15
        }

    def classify_issues(self, detected_issues: List[Dict[str, Any]],
                        raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Classify detected issues and assign confidence scores"""
        classified_issues = []

        for issue in detected_issues:
            # Calculate confidence based on evidence strength
            confidence = self._calculate_confidence(issue)

            # Calculate likelihood ratios
            likelihood = self._calculate_likelihood(issue, raw_data)

            # Apply Bayesian reasoning
            posterior_prob = self._calculate_posterior(issue['type'], likelihood)

            classified_issue = {
                **issue,
                'confidence': confidence,
                'likelihood': likelihood,
                'posterior_probability': posterior_prob,
                'classification_method': 'bayesian_inference'
            }

            classified_issues.append(classified_issue)

        return classified_issues

    def _calculate_confidence(self, issue: Dict[str, Any]) -> float:
        """Calculate confidence score based on evidence"""
        evidence = issue.get('evidence', {})

        if issue['type'] == 'C1_inadequate_sampling':
            # Higher confidence for more extreme sampling issues
            sampling_rate = evidence.get('sampling_rate', 1.0)
            gap_count = evidence.get('gap_count', 0)
            return min(1.0, (2.0 - sampling_rate) / 2.0 + gap_count / 100.0)

        elif issue['type'] == 'C2_poor_placement':
            # Based on inconsistency score
            inconsistency = evidence.get('inconsistency_score', 0)
            return min(1.0, inconsistency / 3.0)

        elif issue['type'] == 'C3_sensor_noise':
            # Based on outlier ratio and noise level
            outlier_ratio = evidence.get('outlier_ratio', 0)
            noise_level = evidence.get('noise_level', 0)
            return min(1.0, outlier_ratio * 2 + noise_level)

        elif issue['type'] == 'C4_range_too_small':
            # Based on clipping ratio
            clipped_ratio = evidence.get('clipped_ratio', 0)
            return min(1.0, clipped_ratio * 2)

        elif issue['type'] == 'C5_high_volume':
            # Based on timing irregularity and drop ratio
            timing_cv = evidence.get('timing_cv', 0)
            drop_ratio = evidence.get('drop_ratio', 0)
            return min(1.0, timing_cv / 2.0 + drop_ratio)

        return 0.5  # Default moderate confidence

    def _calculate_likelihood(self, issue: Dict[str, Any], raw_data: pd.DataFrame) -> float:
        """Calculate likelihood of issue given the data"""
        # This would be more sophisticated in practice
        # For now, use heuristics based on issue type and evidence

        base_likelihood = 0.5
        evidence = issue.get('evidence', {})

        # Adjust likelihood based on severity and evidence strength
        severity_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2
        }.get(issue.get('severity', 'medium'), 1.0)

        return min(1.0, base_likelihood * severity_multiplier * issue.get('confidence', 0.5))

    def _calculate_posterior(self, issue_type: str, likelihood: float) -> float:
        """Calculate posterior probability using Bayesian inference"""
        prior = self.root_cause_priors.get(issue_type, 0.1)

        # Simple Bayesian update (in practice would be more sophisticated)
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))

        return posterior
