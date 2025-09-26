import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .sensor_models import BaseSensor, SensorReading


class DataGenerator:
    """Generates synthetic sensor data with embedded quality issues"""

    def generate_sensor_readings(self, sensor: BaseSensor, activity: str,
                                 start_time: datetime, end_time: datetime,
                                 case_id: int) -> List[Dict[str, Any]]:
        """Generate sensor readings for a specific activity period"""
        readings = []

        duration = (end_time - start_time).total_seconds()
        expected_samples = int(duration * sensor.sampling_rate)

        if expected_samples == 0:
            return readings

        # Generate timestamps
        if sensor.has_volume_issues:
            # C5: Data volume issues - irregular sampling
            timestamps = self._generate_irregular_timestamps(
                start_time, end_time, expected_samples, sensor
            )
        else:
            # Regular sampling
            timestamps = [
                start_time + timedelta(seconds=i / sensor.sampling_rate)
                for i in range(expected_samples)
            ]

        for i, timestamp in enumerate(timestamps):
            # Skip readings due to volume issues
            if sensor.has_volume_issues and np.random.random() < sensor.drop_probability:
                continue

            time_in_activity = (timestamp - start_time).total_seconds()

            # Get expected value
            expected_value = sensor.get_expected_value(activity, time_in_activity)

            # Apply quality issues
            actual_value, quality_flags = self._apply_quality_issues(
                sensor, expected_value, time_in_activity, activity
            )

            # Apply timestamp drift for C5 issues
            if sensor.has_volume_issues:
                timestamp_drift = np.random.normal(0, sensor.timestamp_drift_std)
                timestamp += timedelta(seconds=timestamp_drift)

            reading = {
                'sensor_id': sensor.sensor_id,
                'timestamp': timestamp,
                'value': actual_value,
                'unit': sensor.get_unit(),
                'case_id': case_id,
                'activity': activity,
                'quality_flags': quality_flags
            }

            readings.append(reading)

        return readings

    def _generate_irregular_timestamps(self, start_time: datetime, end_time: datetime,
                                       expected_samples: int, sensor: BaseSensor) -> List[datetime]:
        """Generate irregular timestamps for C5 issues"""
        duration = (end_time - start_time).total_seconds()

        # Generate irregular intervals
        intervals = np.random.exponential(1.0 / sensor.sampling_rate, expected_samples)
        intervals = intervals * (duration / np.sum(intervals))  # Normalize to fit duration

        timestamps = []
        current_time = start_time

        for interval in intervals:
            current_time += timedelta(seconds=interval)
            if current_time <= end_time:
                timestamps.append(current_time)
            else:
                break

        return timestamps

    def _apply_quality_issues(self, sensor: BaseSensor, expected_value: float,
                              time_in_activity: float, activity: str) -> tuple:
        """Apply various quality issues to sensor reading"""
        actual_value = expected_value
        quality_flags = {}

        # C3: Sensor noise and outliers
        if sensor.noise_level > 0:
            noise = np.random.normal(0, sensor.noise_level * abs(expected_value))
            actual_value += noise

            if abs(noise) > 2 * sensor.noise_level * abs(expected_value):
                quality_flags['high_noise'] = True

        # C3: Outliers
        from ..config.settings import QUALITY_CONFIG
        if np.random.random() < QUALITY_CONFIG.sensor_noise['outlier_probability']:
            outlier_magnitude = QUALITY_CONFIG.sensor_noise['outlier_magnitude']
            outlier = np.random.normal(0, outlier_magnitude * abs(expected_value))
            actual_value += outlier
            quality_flags['outlier'] = True

        # C2: Poor sensor placement - overlapping readings
        if sensor.has_placement_issues:
            if np.random.random() < sensor.placement_overlap:
                # Simulate reading from wrong location/activity
                interference = np.random.normal(expected_value * 0.5, abs(expected_value) * 0.2)
                actual_value = (actual_value + interference) / 2
                quality_flags['placement_overlap'] = True

            if np.random.random() < sensor.placement_inconsistency:
                # Inconsistent readings
                inconsistency = np.random.uniform(-0.3, 0.3) * expected_value
                actual_value += inconsistency
                quality_flags['placement_inconsistency'] = True

        # C4: Sensor range too small - clipping and blind spots
        if sensor.has_range_issues:
            # Reduce effective range
            max_range = abs(expected_value) * sensor.range_reduction
            if abs(actual_value) > max_range:
                actual_value = np.sign(actual_value) * max_range
                quality_flags['range_clipping'] = True

            # Blind spots - no reading in certain value ranges
            if np.random.random() < sensor.blind_spots:
                quality_flags['blind_spot'] = True
                return None, quality_flags  # No reading

        # C1: Inadequate sampling rate effects
        if sensor.sampling_rate < 1.0:  # Less than 1 Hz considered inadequate
            quality_flags['inadequate_sampling'] = True
            # Fast changes might be missed - smooth the value
            if time_in_activity > 0:
                smoothing_factor = max(0.1, sensor.sampling_rate / 2.0)
                actual_value = actual_value * smoothing_factor + expected_value * (1 - smoothing_factor)

        return actual_value, quality_flags