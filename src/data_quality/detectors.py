import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class QualityIssueDetector:
    """Detects various types of data quality issues in IoT sensor data"""

    def __init__(self):
        self.detected_issues = []

    def detect_all_issues(self, raw_data: pd.DataFrame, environment) -> List[Dict[str, Any]]:
        """Detect all types of quality issues in the raw data"""
        self.detected_issues = []

        # Group by sensor for analysis
        for sensor_id in raw_data['sensor_id'].unique():
            sensor_data = raw_data[raw_data['sensor_id'] == sensor_id].copy()
            sensor_data = sensor_data.sort_values('timestamp').reset_index(drop=True)

            # Detect each type of issue
            c1_issues = self._detect_inadequate_sampling(sensor_data, sensor_id)
            c2_issues = self._detect_poor_placement(sensor_data, sensor_id)
            c3_issues = self._detect_noise_outliers(sensor_data, sensor_id)
            c4_issues = self._detect_range_issues(sensor_data, sensor_id)
            c5_issues = self._detect_volume_issues(sensor_data, sensor_id)

            # Combine all issues
            self.detected_issues.extend(c1_issues + c2_issues + c3_issues + c4_issues + c5_issues)

        return self.detected_issues

    def _detect_inadequate_sampling(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect C1: Inadequate sampling rate issues"""
        issues = []

        if len(sensor_data) < 2:
            return issues

        # Calculate sampling intervals
        sensor_data['time_diff'] = sensor_data['timestamp'].diff().dt.total_seconds()
        sampling_intervals = sensor_data['time_diff'].dropna()

        if len(sampling_intervals) == 0:
            return issues

        # Detect irregular sampling
        median_interval = sampling_intervals.median()
        std_interval = sampling_intervals.std()

        # Check for large gaps (missed samples)
        large_gaps = sampling_intervals[sampling_intervals > median_interval + 3 * std_interval]

        if len(large_gaps) > 0:
            issues.append({
                'type': 'C1_inadequate_sampling',
                'sensor_id': sensor_id,
                'description': f'Detected {len(large_gaps)} large sampling gaps',
                'severity': 'high' if len(large_gaps) > len(sampling_intervals) * 0.1 else 'medium',
                'evidence': {
                    'median_interval': median_interval,
                    'max_gap': large_gaps.max(),
                    'gap_count': len(large_gaps),
                    'sampling_rate_estimate': 1.0 / median_interval if median_interval > 0 else 0
                },
                'signatures': {
                    'raw_data': 'irregular_sampling',
                    'events': 'missing_events',
                    'process_model': 'incomplete_paths'
                }
            })

        # Check for overall low sampling rate
        estimated_rate = 1.0 / median_interval if median_interval > 0 else 0
        if estimated_rate < 1.0:  # Less than 1 Hz
            issues.append({
                'type': 'C1_inadequate_sampling',
                'sensor_id': sensor_id,
                'description': f'Low sampling rate: {estimated_rate:.2f} Hz',
                'severity': 'high' if estimated_rate < 0.5 else 'medium',
                'evidence': {
                    'sampling_rate': estimated_rate,
                    'recommended_rate': 2.0
                },
                'signatures': {
                    'raw_data': 'low_frequency',
                    'events': 'missing_short_activities',
                    'process_model': 'oversimplified'
                }
            })

        return issues

    def _detect_poor_placement(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect C2: Poor sensor placement issues"""
        issues = []

        # Check for inconsistent readings within same activities
        activity_groups = sensor_data.groupby('activity')['value']

        inconsistency_score = 0
        overlapping_activities = []

        for activity, values in activity_groups:
            if len(values) < 3:
                continue

            # Calculate coefficient of variation
            cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')

            if cv > 0.5:  # High variability within same activity
                inconsistency_score += cv
                overlapping_activities.append(activity)

        if inconsistency_score > 1.0:
            issues.append({
                'type': 'C2_poor_placement',
                'sensor_id': sensor_id,
                'description': f'High inconsistency in {len(overlapping_activities)} activities',
                'severity': 'high' if inconsistency_score > 2.0 else 'medium',
                'evidence': {
                    'inconsistency_score': inconsistency_score,
                    'affected_activities': overlapping_activities
                },
                'signatures': {
                    'raw_data': 'inconsistent_readings',
                    'events': 'concurrent_events',
                    'process_model': 'parallel_branches'
                }
            })

        # Detect overlapping sensor readings from different activities
        overlaps = self._detect_temporal_overlaps(sensor_data)
        if overlaps:
            issues.append({
                'type': 'C2_poor_placement',
                'sensor_id': sensor_id,
                'description': f'Detected {len(overlaps)} temporal overlaps',
                'severity': 'medium',
                'evidence': {
                    'overlapping_periods': overlaps
                },
                'signatures': {
                    'raw_data': 'overlapping_readings',
                    'events': 'out_of_sequence',
                    'process_model': 'wrong_dependencies'
                }
            })

        return issues

    def _detect_noise_outliers(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect C3: Sensor noise and outlier issues"""
        issues = []

        if len(sensor_data) < 10:
            return issues

        values = sensor_data['value'].values

        # Use IQR method for outlier detection
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outlier_ratio = len(outliers) / len(values)

        if outlier_ratio > 0.05:  # More than 5% outliers
            issues.append({
                'type': 'C3_sensor_noise',
                'sensor_id': sensor_id,
                'description': f'{len(outliers)} outliers ({outlier_ratio:.1%} of readings)',
                'severity': 'high' if outlier_ratio > 0.15 else 'medium',
                'evidence': {
                    'outlier_count': len(outliers),
                    'outlier_ratio': outlier_ratio,
                    'outlier_bounds': [lower_bound, upper_bound]
                },
                'signatures': {
                    'raw_data': 'high_variance',
                    'events': 'erroneous_events',
                    'process_model': 'spaghetti_model'
                }
            })

        # Detect high noise level
        noise_estimate = self._estimate_noise_level(sensor_data)
        if noise_estimate > 0.1:  # High noise threshold
            issues.append({
                'type': 'C3_sensor_noise',
                'sensor_id': sensor_id,
                'description': f'High noise level: {noise_estimate:.3f}',
                'severity': 'medium',
                'evidence': {
                    'noise_level': noise_estimate,
                    'signal_to_noise_ratio': abs(values.mean()) / values.std() if values.std() > 0 else 0
                },
                'signatures': {
                    'raw_data': 'noisy_signal',
                    'events': 'incorrect_labels',
                    'process_model': 'complex_paths'
                }
            })

        return issues

    def _detect_range_issues(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect C4: Sensor range too small issues"""
        issues = []

        values = sensor_data['value'].values

        # Check for value clipping (many readings at min/max)
        value_counts = pd.Series(values).value_counts()
        most_common_value = value_counts.iloc[0]
        total_readings = len(values)

        # If more than 10% of readings are the same value, might indicate clipping
        if most_common_value / total_readings > 0.1:
            clipped_value = value_counts.index[0]
            issues.append({
                'type': 'C4_range_too_small',
                'sensor_id': sensor_id,
                'description': f'Potential clipping at value {clipped_value}',
                'severity': 'medium',
                'evidence': {
                    'clipped_value': clipped_value,
                    'clipped_ratio': most_common_value / total_readings,
                    'clipped_count': most_common_value
                },
                'signatures': {
                    'raw_data': 'clipped_values',
                    'events': 'missing_phases',
                    'process_model': 'incomplete_model'
                }
            })

        # Check for limited dynamic range
        value_range = values.max() - values.min()
        value_std = values.std()

        if value_std > 0 and value_range / value_std < 4:  # Limited range relative to variation
            issues.append({
                'type': 'C4_range_too_small',
                'sensor_id': sensor_id,
                'description': f'Limited dynamic range: {value_range:.2f}',
                'severity': 'medium',
                'evidence': {
                    'value_range': value_range,
                    'range_to_std_ratio': value_range / value_std
                },
                'signatures': {
                    'raw_data': 'compressed_range',
                    'events': 'merged_activities',
                    'process_model': 'wrong_splitting'
                }
            })

        return issues

    def _detect_volume_issues(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect C5: High data volume/velocity issues"""
        issues = []

        # Check for dropped samples
        if len(sensor_data) < 2:
            return issues

        # Analyze timestamp distribution
        sensor_data = sensor_data.sort_values('timestamp')
        time_diffs = sensor_data['timestamp'].diff().dt.total_seconds().dropna()

        if len(time_diffs) == 0:
            return issues

        # Detect irregular timing (sign of processing bottlenecks)
        cv_timing = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 0

        if cv_timing > 1.0:  # High variability in timing
            issues.append({
                'type': 'C5_high_volume',
                'sensor_id': sensor_id,
                'description': f'Irregular timing (CV: {cv_timing:.2f})',
                'severity': 'medium',
                'evidence': {
                    'timing_cv': cv_timing,
                    'mean_interval': time_diffs.mean(),
                    'max_gap': time_diffs.max()
                },
                'signatures': {
                    'raw_data': 'irregular_timestamps',
                    'events': 'timestamp_imprecision',
                    'process_model': 'low_fitness'
                }
            })

        # Check for potential data drops
        expected_samples_per_hour = 3600  # Assume 1 Hz base rate
        actual_duration_hours = (sensor_data['timestamp'].max() - sensor_data['timestamp'].min()).total_seconds() / 3600
        expected_total = expected_samples_per_hour * actual_duration_hours

        if len(sensor_data) < expected_total * 0.8:  # Less than 80% of expected samples
            drop_ratio = 1 - (len(sensor_data) / expected_total)
            issues.append({
                'type': 'C5_high_volume',
                'sensor_id': sensor_id,
                'description': f'Potential data drops: {drop_ratio:.1%}',
                'severity': 'high' if drop_ratio > 0.3 else 'medium',
                'evidence': {
                    'expected_samples': expected_total,
                    'actual_samples': len(sensor_data),
                    'drop_ratio': drop_ratio
                },
                'signatures': {
                    'raw_data': 'missing_readings',
                    'events': 'incomplete_traces',
                    'process_model': 'wrong_variants'
                }
            })

        return issues

    def _detect_temporal_overlaps(self, sensor_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect overlapping readings from different activities"""
        overlaps = []

        # Group consecutive readings by activity
        sensor_data = sensor_data.sort_values('timestamp')
        activity_changes = sensor_data['activity'] != sensor_data['activity'].shift(1)
        activity_groups = activity_changes.cumsum()

        segments = []
        for group_id, group_data in sensor_data.groupby(activity_groups):
            if len(group_data) > 0:
                segments.append({
                    'activity': group_data['activity'].iloc[0],
                    'start': group_data['timestamp'].min(),
                    'end': group_data['timestamp'].max(),
                    'readings': len(group_data)
                })

        # Check for temporal overlaps
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]

            if current['end'] > next_segment['start']:
                overlaps.append({
                    'activities': [current['activity'], next_segment['activity']],
                    'overlap_duration': (current['end'] - next_segment['start']).total_seconds()
                })

        return overlaps

    def _estimate_noise_level(self, sensor_data: pd.DataFrame) -> float:
        """Estimate noise level using first-order differences"""
        if len(sensor_data) < 3:
            return 0.0

        values = sensor_data['value'].values
        first_diff = np.diff(values)

        # Estimate noise as std of high-frequency components
        if len(first_diff) > 1:
            noise_estimate = np.std(first_diff) / np.sqrt(2)
            return noise_estimate / (np.std(values) if np.std(values) > 0 else 1.0)

        return 0.0