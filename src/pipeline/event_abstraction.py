"""Simplified event abstraction with direct mapping"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta


class EventAbstractor:
    """Abstracts raw sensor data to structured events using direct mapping"""

    def __init__(self, predefined_activities: Set[str] = None):
        self.predefined_activities = predefined_activities or {
            'Welding_Position', 'Weld', 'Cool',
            'Inspection_Position', 'Scan', 'Measure', 'Validate',
            'Package_Pick', 'Package', 'Seal'
        }

        self.activity_thresholds = self._initialize_activity_thresholds()

    def abstract_events(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract preprocessed sensor data to structured events.

        :param preprocessed_data: Preprocessed sensor data
        :return: DataFrame of structured events
        """
        if preprocessed_data.empty:
            return pd.DataFrame()

        if 'activity' not in preprocessed_data.columns:
            return self._detect_events_from_patterns(preprocessed_data)

        events = []

        for case_id in preprocessed_data['case_id'].unique():
            case_data = preprocessed_data[preprocessed_data['case_id'] == case_id].copy()
            case_data = case_data.sort_values('timestamp')

            case_events = self._extract_events_from_case(case_data)
            events.extend(case_events)

        if not events:
            return pd.DataFrame()

        events_df = pd.DataFrame(events)
        create_results_order = events_df.sort_values('start_time').reset_index(drop=True)
        return create_results_order

    def _extract_events_from_case(self, case_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract events from case data with activity labels.

        :param case_data: Data for single case
        :return: List of event dictionaries
        """
        events = []

        case_data['activity_change'] = case_data['activity'] != case_data['activity'].shift(1)
        case_data['activity_group'] = case_data['activity_change'].cumsum()

        for group_id, group_data in case_data.groupby('activity_group'):
            activity = group_data['activity'].iloc[0]

            if activity not in self.predefined_activities:
                continue

            start_time = group_data['timestamp'].min()
            end_time = group_data['timestamp'].max()
            duration = (end_time - start_time).total_seconds()

            if duration < 0.5:
                continue

            sensor_ids = group_data['sensor_id'].unique().tolist()

            event = {
                'activity': activity,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'sensor_ids': sensor_ids,
                'num_sensors': len(sensor_ids),
                'avg_value': group_data['value'].mean() if 'value' in group_data.columns else 0,
                'event_type': 'direct_mapping',
                'quality_issues': self._detect_event_quality_issues(group_data)
            }

            events.append(event)

        return events

    def _detect_events_from_patterns(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect events when activity labels are missing using sensor patterns.

        :param preprocessed_data: Preprocessed data without activity labels
        :return: DataFrame of detected events
        """
        events = []

        for sensor_id in preprocessed_data['sensor_id'].unique():
            sensor_data = preprocessed_data[
                preprocessed_data['sensor_id'] == sensor_id
                ].copy().sort_values('timestamp')

            if len(sensor_data) < 5:
                continue

            sensor_events = self._detect_sensor_events_simple(sensor_data, sensor_id)
            events.extend(sensor_events)

        if not events:
            return pd.DataFrame()

        events_df = pd.DataFrame(events)
        events_df = self._merge_concurrent_events_simple(events_df)

        return events_df.sort_values('start_time').reset_index(drop=True)

    def _detect_sensor_events_simple(
            self, sensor_data: pd.DataFrame, sensor_id: str
    ) -> List[Dict[str, Any]]:
        """
        Detect events from sensor using simple thresholding.

        :param sensor_data: Data from single sensor
        :param sensor_id: Sensor identifier
        :return: List of detected events
        """
        events = []

        if 'normalized_value' not in sensor_data.columns:
            return events

        values = sensor_data['normalized_value'].values
        timestamps = sensor_data['timestamp'].values

        threshold_high = np.percentile(values, 75)
        threshold_low = np.percentile(values, 25)

        above_threshold = values > threshold_high
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])

        for start_idx, end_idx in zip(starts, ends):
            if end_idx <= start_idx:
                continue

            duration = (timestamps[end_idx - 1] - timestamps[start_idx]) / np.timedelta64(1, 's')

            if duration >= 1.0:
                activity = self._classify_activity_simple(sensor_id, values[start_idx:end_idx])

                events.append({
                    'sensor_id': sensor_id,
                    'activity': activity,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx - 1],
                    'duration': duration,
                    'avg_value': np.mean(values[start_idx:end_idx]),
                    'event_type': 'threshold_detection',
                    'quality_issues': []
                })

        return events

    def _classify_activity_simple(self, sensor_id: str, values: np.ndarray) -> str:
        """
        Classify activity based on sensor type and value pattern.

        :param sensor_id: Sensor identifier
        :param values: Sensor values
        :return: Activity name
        """
        avg_value = np.mean(values)

        if 'PWR' in sensor_id:
            if avg_value > 0.8:
                return 'Weld'
            elif avg_value > 0.5:
                return 'Package'
            else:
                return 'Welding_Position'
        elif 'TEMP' in sensor_id:
            return 'Weld' if avg_value > 0.6 else 'Cool'
        elif 'VIB' in sensor_id:
            return 'Scan' if avg_value > 0.4 else 'Measure'
        elif 'POS' in sensor_id:
            return 'Inspection_Position'

        return 'Unknown_Activity'

    def _merge_concurrent_events_simple(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge events occurring concurrently.

        :param events_df: DataFrame of events
        :return: Merged events DataFrame
        """
        if events_df.empty:
            return events_df

        events_df = events_df.sort_values('start_time').reset_index(drop=True)

        merged_events = []
        time_window = timedelta(seconds=5)

        i = 0
        while i < len(events_df):
            current_event = events_df.iloc[i].to_dict()
            current_start = current_event['start_time']
            current_end = current_event['end_time']

            concurrent = [current_event]
            j = i + 1

            while j < len(events_df):
                next_event = events_df.iloc[j].to_dict()
                next_start = next_event['start_time']

                if next_start > current_end + time_window:
                    break

                next_end = next_event['end_time']
                if next_start <= current_end + time_window and next_end >= current_start:
                    concurrent.append(next_event)
                    current_end = max(current_end, next_end)

                j += 1

            if len(concurrent) > 1:
                merged_event = self._create_merged_event(concurrent)
                merged_events.append(merged_event)
            else:
                merged_events.append(current_event)

            i = j if j > i else i + 1

        return pd.DataFrame(merged_events)

    def _create_merged_event(self, concurrent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create single event from concurrent events.

        :param concurrent_events: List of concurrent events
        :return: Merged event dictionary
        """
        activities = [e['activity'] for e in concurrent_events]
        activity_counts = pd.Series(activities).value_counts()
        dominant_activity = activity_counts.index[0]

        start_times = [e['start_time'] for e in concurrent_events]
        end_times = [e['end_time'] for e in concurrent_events]

        sensor_ids = []
        for e in concurrent_events:
            if isinstance(e.get('sensor_ids'), list):
                sensor_ids.extend(e['sensor_ids'])
            elif 'sensor_id' in e:
                sensor_ids.append(e['sensor_id'])

        return {
            'activity': dominant_activity,
            'start_time': min(start_times),
            'end_time': max(end_times),
            'duration': (max(end_times) - min(start_times)).total_seconds(),
            'sensor_ids': list(set(sensor_ids)),
            'num_sensors': len(set(sensor_ids)),
            'event_type': 'merged',
            'confidence': activities.count(dominant_activity) / len(activities),
            'quality_issues': []
        }

    def _detect_event_quality_issues(self, event_data: pd.DataFrame) -> List[str]:
        """
        Detect quality issues in event data.

        :param event_data: Data for single event
        :return: List of quality issue identifiers
        """
        issues = []

        if 'quality_flags' in event_data.columns:
            for _, row in event_data.iterrows():
                if isinstance(row['quality_flags'], dict):
                    for flag, value in row['quality_flags'].items():
                        if value:
                            issues.append(flag)

        if len(event_data) < 3:
            issues.append('insufficient_samples')

        return list(set(issues))

    def _initialize_activity_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize activity-specific thresholds.

        :return: Dictionary of activity thresholds
        """
        return {
            'Weld': {'min_value': 0.7, 'min_duration': 5.0},
            'Cool': {'min_value': 0.3, 'min_duration': 8.0},
            'Scan': {'min_value': 0.4, 'min_duration': 3.0},
            'Package': {'min_value': 0.5, 'min_duration': 4.0}
        }