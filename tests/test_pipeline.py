import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pipeline.pipeline_manager import PipelineManager
from src.pipeline.preprocessing import Preprocessor
from src.pipeline.event_abstraction import EventAbstractor


class TestPipelineComponents(unittest.TestCase):
    """Test cases for pipeline components"""

    def setUp(self):
        """Set up test data"""
        # Create sample raw data
        timestamps = pd.date_range(start='2024-01-01', periods=50, freq='2S')

        self.sample_raw_data = pd.DataFrame({
            'sensor_id': ['PWR_01'] * 25 + ['TEMP_01'] * 25,
            'timestamp': list(timestamps[:25]) + list(timestamps[25:]),
            'value': list(np.random.normal(150, 20, 25)) + list(np.random.normal(25, 5, 25)),
            'unit': ['W'] * 25 + ['°C'] * 25,
            'case_id': [1] * 50,
            'activity': ['Welding_Station_Weld'] * 50,
            'quality_flags': [{}] * 50
        })

    def test_preprocessor(self):
        """Test data preprocessing"""
        preprocessor = Preprocessor()
        processed = preprocessor.preprocess(self.sample_raw_data)

        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(len(processed), len(self.sample_raw_data))

        # Should add new columns
        expected_new_columns = ['smoothed_value', 'normalized_value', 'time_diff', 'value_diff']
        for col in expected_new_columns:
            self.assertIn(col, processed.columns)

    def test_event_abstractor(self):
        """Test event abstraction"""
        # First preprocess the data
        preprocessor = Preprocessor()
        processed_data = preprocessor.preprocess(self.sample_raw_data)

        abstractor = EventAbstractor()
        events = abstractor.abstract_events(processed_data)

        self.assertIsInstance(events, pd.DataFrame)

        if len(events) > 0:  # Events might not be detected with small test data
            required_columns = ['activity', 'start_time', 'end_time', 'duration']
            for col in required_columns:
                self.assertIn(col, events.columns)


class TestPipelineManager(unittest.TestCase):
    """Test cases for pipeline manager"""

    def setUp(self):
        self.pipeline = PipelineManager()

    def test_pipeline_initialization(self):
        """Test pipeline component initialization"""
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertIsNotNone(self.pipeline.event_abstractor)
        self.assertIsNotNone(self.pipeline.case_correlator)
        self.assertIsNotNone(self.pipeline.process_miner)
        self.assertIsNotNone(self.pipeline.quality_detector)
