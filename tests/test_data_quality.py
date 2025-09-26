import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_quality.detectors import QualityIssueDetector
from src.data_quality.classifiers import QualityClassifier
from src.data_quality.propagation import QualityPropagator


class TestQualityDetection(unittest.TestCase):
    """Test cases for quality issue detection"""

    def setUp(self):
        """Set up test data"""
        self.detector = QualityIssueDetector()

        # Create sample data with known quality issues
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1S')

        # Normal data
        self.normal_data = pd.DataFrame({
            'sensor_id': ['SENSOR_01'] * 100,
            'timestamp': timestamps,
            'value': np.random.normal(100, 5, 100),
            'unit': ['W'] * 100,
            'case_id': [1] * 100,
            'activity': ['Normal_Activity'] * 100,
            'quality_flags': [{}] * 100
        })

        # Data with gaps (C1 issue)
        gap_timestamps = timestamps.delete(range(20, 40))  # Remove 20 timestamps
        self.gap_data = pd.DataFrame({
            'sensor_id': ['SENSOR_02'] * len(gap_timestamps),
            'timestamp': gap_timestamps,
            'value': np.random.normal(100, 5, len(gap_timestamps)),
            'unit': ['W'] * len(gap_timestamps),
            'case_id': [1] * len(gap_timestamps),
            'activity': ['Gap_Activity'] * len(gap_timestamps),
            'quality_flags': [{}] * len(gap_timestamps)
        })

        # Noisy data (C3 issue)
        noisy_values = np.random.normal(100, 5, 90)
        # Add outliers
        noisy_values = np.concatenate([noisy_values, [200, 250, -50, 300, 350, 400, -100, 500, 600, 700]])

        self.noisy_data = pd.DataFrame({
            'sensor_id': ['SENSOR_03'] * 100,
            'timestamp': timestamps,
            'value': noisy_values,
            'unit': ['W'] * 100,
            'case_id': [1] * 100,
            'activity': ['Noisy_Activity'] * 100,
            'quality_flags': [{}] * 100
        })

    def test_inadequate_sampling_detection(self):
        """Test C1 inadequate sampling detection"""
        issues = self.detector._detect_inadequate_sampling(self.gap_data, 'SENSOR_02')

        self.assertGreater(len(issues), 0)
        issue = issues[0]
        self.assertEqual(issue['type'], 'C1_inadequate_sampling')
        self.assertIn('large sampling gaps', issue['description'])

    def test_noise_outlier_detection(self):
        """Test C3 noise and outlier detection"""
        issues = self.detector._detect_noise_outliers(self.noisy_data, 'SENSOR_03')

        self.assertGreater(len(issues), 0)
        issue = issues[0]
        self.assertEqual(issue['type'], 'C3_sensor_noise')
        self.assertIn('outliers', issue['description'])

    def test_detect_all_issues(self):
        """Test detecting all issues in combined dataset"""
        combined_data = pd.concat([self.normal_data, self.gap_data, self.noisy_data], ignore_index=True)

        # Mock environment object
        class MockEnvironment:
            pass

        issues = self.detector.detect_all_issues(combined_data, MockEnvironment())

        self.assertIsInstance(issues, list)
        # Should detect at least the gap and noise issues
        issue_types = [issue['type'] for issue in issues]
        self.assertIn('C1_inadequate_sampling', issue_types)
        self.assertIn('C3_sensor_noise', issue_types)


class TestQualityClassification(unittest.TestCase):
    """Test cases for quality issue classification"""

    def setUp(self):
        self.classifier = QualityClassifier()

        # Sample detected issues
        self.sample_issues = [
            {
                'type': 'C1_inadequate_sampling',
                'sensor_id': 'SENSOR_01',
                'description': 'Low sampling rate: 0.5 Hz',
                'severity': 'high',
                'evidence': {
                    'sampling_rate': 0.5,
                    'gap_count': 15
                }
            },
            {
                'type': 'C3_sensor_noise',
                'sensor_id': 'SENSOR_02',
                'description': '25 outliers (15% of readings)',
                'severity': 'medium',
                'evidence': {
                    'outlier_ratio': 0.15,
                    'noise_level': 0.12
                }
            }
        ]

    def test_classify_issues(self):
        """Test issue classification"""
        classified = self.classifier.classify_issues(
            self.sample_issues,
            pd.DataFrame()  # Empty raw data for this test
        )

        self.assertEqual(len(classified), 2)

        for issue in classified:
            self.assertIn('confidence', issue)
            self.assertIn('likelihood', issue)
            self.assertIn('posterior_probability', issue)
            self.assertGreaterEqual(issue['confidence'], 0.0)
            self.assertLessEqual(issue['confidence'], 1.0)

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        c1_issue = self.sample_issues[0]
        confidence = self.classifier._calculate_confidence(c1_issue)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestQualityPropagation(unittest.TestCase):
    """Test cases for quality issue propagation"""

    def setUp(self):
        self.propagator = QualityPropagator()

        self.sample_classified_issues = [
            {
                'type': 'C1_inadequate_sampling',
                'sensor_id': 'SENSOR_01',
                'confidence': 0.8,
                'posterior_probability': 0.7,
                'signatures': {
                    'raw_data': 'irregular_sampling',
                    'events': 'missing_events',
                    'process_model': 'incomplete_paths'
                }
            }
        ]

    def test_propagate_issues(self):
        """Test issue propagation through pipeline stages"""
        result = self.propagator.propagate_issues(
            self.sample_classified_issues,
            'event_abstraction',
            pd.DataFrame()  # Mock stage data
        )

        self.assertIn('propagated_issues', result)
        self.assertIn('stage_signatures', result)
        self.assertIn('stage', result)
        self.assertEqual(result['stage'], 'event_abstraction')

        propagated = result['propagated_issues'][0]
        self.assertIn('stage_effects', propagated)
        self.assertIn('propagated_probability', propagated)
        self.assertIn('information_gain', propagated)

