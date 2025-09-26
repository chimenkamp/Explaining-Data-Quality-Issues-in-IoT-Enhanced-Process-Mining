import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
from scipy import signal
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Preprocesses raw IoT sensor data while preserving quality issue information"""

    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw sensor data"""

        if raw_data.empty:
            return raw_data

        processed_data = raw_data.copy()

        # Sort by timestamp
        processed_data = processed_data.sort_values(['sensor_id', 'timestamp']).reset_index(drop=True)

        # Handle missing timestamps (C5 issue indicator)
        processed_data = self._handle_missing_timestamps(processed_data)

        # Detect and flag anomalies (preserve for quality analysis)
        processed_data = self._flag_anomalies(processed_data)

        # Apply gentle smoothing (preserve quality indicators)
        processed_data = self._apply_quality_aware_smoothing(processed_data)

        # Add temporal features
        processed_data = self._add_temporal_features(processed_data)

        # Add sensor-specific preprocessing
        processed_data = self._sensor_specific_preprocessing(processed_data)

        return processed_data

    def _handle_missing_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing timestamps while flagging as quality issue"""

        result_data = data.copy()

        for sensor_id in data['sensor_id'].unique():
            sensor_mask = data['sensor_id'] == sensor_id
            sensor_data = data[sensor_mask].copy()

            if len(sensor_data) < 2:
                continue

            # Calculate expected sampling interval
            time_diffs = sensor_data['timestamp'].diff().dt.total_seconds()
            median_interval = time_diffs.median()

            # Detect large gaps
            large_gaps = time_diffs > median_interval * 2

            if large_gaps.any():
                # Mark large gaps in quality flags
                gap_indices = sensor_data[large_gaps].index
                for idx in gap_indices:
                    if 'quality_flags' not in result_data.columns:
                        result_data['quality_flags'] = [{}] * len(result_data)

                    result_data.at[idx, 'quality_flags']['timestamp_gap'] = True
                    result_data.at[idx, 'quality_flags']['gap_duration'] = time_diffs[idx]

        return result_data

    def _flag_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Flag anomalies without removing them (preserve for quality analysis)"""

        result_data = data.copy()

        for sensor_id in data['sensor_id'].unique():
            sensor_mask = data['sensor_id'] == sensor_id
            sensor_data = data[sensor_mask]

            if len(sensor_data) < 10:
                continue

            values = sensor_data['value'].values

            # Statistical anomaly detection
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            anomaly_mask = (values < lower_bound) | (values > upper_bound)
            anomaly_indices = sensor_data[anomaly_mask].index

            # Flag anomalies
            for idx in anomaly_indices:
                if 'quality_flags' not in result_data.columns:
                    result_data['quality_flags'] = [{}] * len(result_data)

                result_data.at[idx, 'quality_flags']['statistical_anomaly'] = True
                result_data.at[idx, 'quality_flags']['anomaly_bounds'] = [lower_bound, upper_bound]

        return result_data

    def _apply_quality_aware_smoothing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing while preserving quality issue signatures"""

        result_data = data.copy()
        result_data['smoothed_value'] = result_data['value'].copy()

        for sensor_id in data['sensor_id'].unique():
            sensor_mask = data['sensor_id'] == sensor_id
            sensor_data = data[sensor_mask]

            if len(sensor_data) < 5:
                continue

            values = sensor_data['value'].values

            # Check if sensor has high noise (preserve some noise for quality analysis)
            noise_flags = []
            if 'quality_flags' in sensor_data.columns:
                noise_flags = [qf.get('high_noise', False) if isinstance(qf, dict) else False
                               for qf in sensor_data['quality_flags']]

            if any(noise_flags):
                # Light smoothing for noisy sensors (preserve some noise signature)
                smoothed = signal.savgol_filter(values, min(7, len(values) // 2 * 2 - 1), 2)
                # Keep 30% of original noise
                smoothed = 0.7 * smoothed + 0.3 * values
            else:
                # Normal smoothing
                smoothed = signal.savgol_filter(values, min(5, len(values) // 2 * 2 - 1), 2)

            # Update smoothed values
            result_data.loc[sensor_mask, 'smoothed_value'] = smoothed

        return result_data

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for better event detection"""

        result_data = data.copy()

        # Add time-based features
        result_data['hour'] = result_data['timestamp'].dt.hour
        result_data['minute'] = result_data['timestamp'].dt.minute
        result_data['day_of_week'] = result_data['timestamp'].dt.dayofweek

        # Add time differences for each sensor
        for sensor_id in data['sensor_id'].unique():
            sensor_mask = data['sensor_id'] == sensor_id
            result_data.loc[sensor_mask, 'time_diff'] = (
                result_data.loc[sensor_mask, 'timestamp'].diff().dt.total_seconds()
            )

        # Add value differences
        for sensor_id in data['sensor_id'].unique():
            sensor_mask = data['sensor_id'] == sensor_id
            result_data.loc[sensor_mask, 'value_diff'] = (
                result_data.loc[sensor_mask, 'smoothed_value'].diff()
            )

        return result_data

    def _sensor_specific_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply sensor-specific preprocessing"""

        result_data = data.copy()

        for sensor_id in data['sensor_id'].unique():
            sensor_mask = data['sensor_id'] == sensor_id
            sensor_data = data[sensor_mask]

            # Determine sensor type from ID
            if 'PWR' in sensor_id:
                # Power sensor specific processing
                result_data.loc[sensor_mask, 'normalized_value'] = self._normalize_power_values(
                    sensor_data['smoothed_value']
                )
            elif 'TEMP' in sensor_id:
                # Temperature sensor specific processing
                result_data.loc[sensor_mask, 'normalized_value'] = self._normalize_temperature_values(
                    sensor_data['smoothed_value']
                )
            elif 'VIB' in sensor_id:
                # Vibration sensor specific processing
                result_data.loc[sensor_mask, 'normalized_value'] = self._normalize_vibration_values(
                    sensor_data['smoothed_value']
                )
            elif 'POS' in sensor_id:
                # Position sensor specific processing
                result_data.loc[sensor_mask, 'normalized_value'] = self._normalize_position_values(
                    sensor_data['smoothed_value']
                )
            else:
                # Default normalization
                result_data.loc[sensor_mask, 'normalized_value'] = (
                        (sensor_data['smoothed_value'] - sensor_data['smoothed_value'].mean()) /
                        (sensor_data['smoothed_value'].std() + 1e-6)
                )

        return result_data

    def _normalize_power_values(self, values: pd.Series) -> pd.Series:
        """Normalize power sensor values"""
        # Power sensors often have baseline + spikes pattern
        baseline = values.quantile(0.1)  # Assume 10th percentile is baseline
        return (values - baseline) / (values.max() - baseline + 1e-6)

    def _normalize_temperature_values(self, values: pd.Series) -> pd.Series:
        """Normalize temperature sensor values"""
        # Temperature typically has a range around ambient
        ambient = values.quantile(0.5)  # Median as ambient
        return (values - ambient) / (values.std() + 1e-6)

    def _normalize_vibration_values(self, values: pd.Series) -> pd.Series:
        """Normalize vibration sensor values"""
        # Vibration is often relative to baseline
        return (values - values.min()) / (values.max() - values.min() + 1e-6)

    def _normalize_position_values(self, values: pd.Series) -> pd.Series:
        """Normalize position sensor values"""
        # Position values should be normalized to [0,1] range
        return (values - values.min()) / (values.max() - values.min() + 1e-6)

