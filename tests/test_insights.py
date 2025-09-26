import unittest
from src.explainability.insights import InsightGenerator
from src.explainability.explanations import ExplanationGenerator
import pandas as pd

class TestExplainability(unittest.TestCase):
    """Test cases for explainability components"""

    def setUp(self):
        self.insight_generator = InsightGenerator()
        self.explanation_generator = ExplanationGenerator()

        # Sample pipeline results
        self.sample_results = {
            'quality_issues': [
                {
                    'type': 'C1_inadequate_sampling',
                    'description': 'Low sampling rate detected',
                    'confidence': 0.8,
                    'severity': 'high',
                    'evidence': {'sampling_rate': 0.5}
                }
            ],
            'process_model': {
                'metrics': {
                    'fitness': 0.7,
                    'precision': 0.8,
                    'complexity': 0.6
                }
            },
            'process_instances': pd.DataFrame({
                'case_id': ['case_001'],
                'case_quality_score': [0.8]
            })
        }

    def test_insight_generation(self):
        """Test insight generation"""
        insights = self.insight_generator.generate_insights(self.sample_results)

        self.assertIsInstance(insights, list)

        if len(insights) > 0:
            insight = insights[0]
            required_keys = ['message', 'confidence', 'actionable']
            for key in required_keys:
                self.assertIn(key, insight)

    def test_explanation_generation(self):
        """Test detailed explanation generation"""
        issue = self.sample_results['quality_issues'][0]
        explanation = self.explanation_generator.generate_explanation(issue)

        self.assertIsInstance(explanation, dict)

        required_keys = [
            'issue_summary', 'technical_explanation',
            'root_cause_analysis', 'impact_analysis',
            'remediation_strategy'
        ]

        for key in required_keys:
            self.assertIn(key, explanation)

