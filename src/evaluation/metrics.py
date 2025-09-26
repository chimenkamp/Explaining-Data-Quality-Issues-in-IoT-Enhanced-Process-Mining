"""
Evaluation metrics for assessing pipeline performance
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import logging


class PipelineEvaluator:
    """Evaluates overall pipeline performance and quality"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_quality_detection_accuracy(self,
                                            detected_issues: List[Dict[str, Any]],
                                            ground_truth_issues: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate quality detection accuracy against ground truth"""

        # Convert to sets of issue types for comparison
        detected_types = set(issue['type'] for issue in detected_issues)
        ground_truth_types = set(issue['type'] for issue in ground_truth_issues)

        # Calculate basic metrics
        true_positives = len(detected_types & ground_truth_types)
        false_positives = len(detected_types - ground_truth_types)
        false_negatives = len(ground_truth_types - detected_types)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def evaluate_process_model_quality(self, process_model: Dict[str, Any],
                                       ground_truth_model: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate process model quality"""

        metrics = process_model.get('metrics', {})

        quality_score = {
            'fitness': metrics.get('fitness', 0.0),
            'precision': metrics.get('precision', 0.0),
            'complexity': metrics.get('complexity', 0.0),
            'overall_quality': 0.0
        }

        # Calculate overall quality (higher fitness and precision, lower complexity is better)
        quality_score['overall_quality'] = (
                0.4 * quality_score['fitness'] +
                0.4 * quality_score['precision'] +
                0.2 * (1.0 - quality_score['complexity'])
        )

        return quality_score

    def evaluate_information_gain(self, insights: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate information gain from generated insights"""

        if not insights:
            return {'total_gain': 0.0, 'actionable_ratio': 0.0, 'avg_confidence': 0.0}

        total_gain = sum(insight.get('importance_score', 0) for insight in insights)
        actionable_count = sum(1 for insight in insights if insight.get('actionable', False))
        avg_confidence = np.mean([insight.get('confidence', 0) for insight in insights])

        return {
            'total_gain': total_gain,
            'actionable_ratio': actionable_count / len(insights),
            'avg_confidence': avg_confidence,
            'insight_count': len(insights)
        }
