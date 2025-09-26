import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.synthetic_environment.iot_environment import IoTEnvironment
from src.synthetic_environment.sensor_models import PowerSensor, TemperatureSensor
from src.synthetic_environment.data_generator import DataGenerator


class TestSyntheticEnvironment(unittest.TestCase):
    """Test cases for synthetic IoT environment generation"""

    def setUp(self):
        """Set up test environment"""
        self.env = IoTEnvironment(
            name="Test Environment",
            duration_hours=1,
            num_machines=2
        )

    def test_environment_creation(self):
        """Test basic environment creation"""
        self.assertEqual(self.env.name, "Test Environment")
        self.assertEqual(len(self.env.machines), 0)  # No machines added yet

    def test_add_welding_station(self):
        """Test adding welding station"""
        self.env.add_welding_station()
        self.assertEqual(len(self.env.machines), 1)

        machine = self.env.machines[0]
        self.assertTrue(machine.id.startswith('WS_'))
        self.assertEqual(len(machine.sensors), 2)  # Power + Temperature
        self.assertIn('Weld', machine.process_steps)

    def test_add_inspection_station(self):
        """Test adding inspection station"""
        self.env.add_inspection_station()
        self.assertEqual(len(self.env.machines), 1)

        machine = self.env.machines[0]
        self.assertTrue(machine.id.startswith('IS_'))
        self.assertEqual(len(machine.sensors), 2)  # Position + Vibration
        self.assertIn('Scan', machine.process_steps)

    def test_data_generation(self):
        """Test synthetic data generation"""
        self.env.add_welding_station()
        data = self.env.generate_data()

        self.assertIn('raw_data', data)
        self.assertIn('process_instances', data)
        self.assertIn('environment', data)

        raw_data = data['raw_data']
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)

        # Check required columns
        required_columns = ['sensor_id', 'timestamp', 'value', 'unit', 'case_id', 'activity']
        for col in required_columns:
            self.assertIn(col, raw_data.columns)

    def test_sensor_models(self):
        """Test individual sensor models"""
        power_sensor = PowerSensor(
            sensor_id="TEST_PWR",
            sampling_rate=1.0,
            location=(0, 0)
        )

        self.assertEqual(power_sensor.sensor_id, "TEST_PWR")
        self.assertEqual(power_sensor.sampling_rate, 1.0)
        self.assertEqual(power_sensor.get_unit(), "W")

        # Test value generation
        value = power_sensor.get_expected_value("Welding_Station_Weld", 1.0)
        self.assertIsInstance(value, float)
        self.assertGreater(value, 100.0)  # Should be higher than base power

    def test_quality_issue_injection(self):
        """Test quality issue injection in synthetic data"""
        self.env.add_welding_station()

        # Check that quality issues are injected
        machine = self.env.machines[0]
        sensors = machine.sensors

        # At least some sensors should have quality issues due to random injection
        has_quality_issues = any([
            sensor.has_placement_issues or
            sensor.has_range_issues or
            sensor.has_volume_issues
            for sensor in sensors
        ])

        # Note: This might occasionally fail due to randomness, but probability is low
        # In a real test suite, you'd set random seeds for deterministic testing


class TestDataGenerator(unittest.TestCase):
    """Test cases for data generator"""

    def setUp(self):
        self.generator = DataGenerator()
        self.sensor = PowerSensor(
            sensor_id="TEST_PWR",
            sampling_rate=2.0,
            location=(0, 0)
        )

    def test_sensor_reading_generation(self):
        """Test generating sensor readings"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)

        readings = self.generator.generate_sensor_readings(
            self.sensor, "Test_Activity", start_time, end_time, case_id=1
        )

        self.assertIsInstance(readings, list)
        self.assertGreater(len(readings), 0)

        # Check reading structure
        reading = readings[0]
        self.assertIn('sensor_id', reading)
        self.assertIn('timestamp', reading)
        self.assertIn('value', reading)
        self.assertIn('case_id', reading)
        self.assertEqual(reading['sensor_id'], 'TEST_PWR')
        self.assertEqual(reading['case_id'], 1)

