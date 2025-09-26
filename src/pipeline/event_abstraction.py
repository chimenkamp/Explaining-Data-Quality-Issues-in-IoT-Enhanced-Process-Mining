import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks


class EventAbstractor:
    """Abstracts raw sensor data into structured events"""

    def __init__(self, window_size: int = 10, min_duration: float = 1.0):
        self.window_size = window_size
        self.min_duration = min_duration
        self.scaler = StandardScaler()

    def abstract_events(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Abstract preprocessed sensor data into structured events"""

        if preprocessed_data.empty:
            return pd.DataFrame()

        all_events = []

        # Process each sensor separately
        for sensor_id in preprocessed_data['sensor_id'].unique():
            sensor_data = preprocessed_data[
                preprocessed_data['sensor_id'] == sensor_id
                ].copy().sort_values('timestamp')

            if len(sensor_data) < self.window_size:
                continue

            # Detect events for this sensor
            sensor_events = self._detect_sensor_events(sensor_data, sensor_id)
            all_events.extend(sensor_events)

        # Convert to DataFrame
        if not all_events:
            return pd.DataFrame()

        events_df = pd.DataFrame(all_events)

        # Merge overlapping events from different sensors
        events_df = self._merge_concurrent_events(events_df)

        # Add quality issue indicators
        events_df = self._add_quality_indicators(events_df, preprocessed_data)

        return events_df.sort_values('start_time').reset_index(drop=True)

    def _detect_sensor_events(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect events in individual sensor data"""

        events = []

        if 'normalized_value' not in sensor_data.columns:
            return events

        values = sensor_data['normalized_value'].values
        timestamps = sensor_data['timestamp'].values

        # Detect significant changes using multiple methods
        events.extend(self._detect_threshold_events(sensor_data, sensor_id))
        events.extend(self._detect_peak_events(sensor_data, sensor_id))
        events.extend(self._detect_pattern_events(sensor_data, sensor_id))

        return events

    def _detect_threshold_events(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect events based on threshold crossings"""

        events = []
        values = sensor_data['normalized_value'].values
        timestamps = sensor_data['timestamp'].values

        # Define thresholds based on sensor type
        if 'PWR' in sensor_id:
            # Power sensors: detect high power events
            threshold_high = np.percentile(values, 80)
            threshold_low = np.percentile(values, 20)
        elif 'TEMP' in sensor_id:
            # Temperature sensors: detect heating/cooling events
            threshold_high = np.percentile(values, 75)
            threshold_low = np.percentile(values, 25)
        elif 'VIB' in sensor_id:
            # Vibration sensors: detect high vibration events
            threshold_high = np.percentile(values, 85)
            threshold_low = np.percentile(values, 15)
        else:
            # Default thresholds
            threshold_high = np.percentile(values, 80)
            threshold_low = np.percentile(values, 20)

        # Find threshold crossings
        high_events = self._find_threshold_crossings(
            sensor_data, threshold_high, 'high', sensor_id
        )
        low_events = self._find_threshold_crossings(
            sensor_data, threshold_low, 'low', sensor_id
        )

        events.extend(high_events)
        events.extend(low_events)

        return events

    def _find_threshold_crossings(self, sensor_data: pd.DataFrame, threshold: float,
                                  event_type: str, sensor_id: str) -> List[Dict[str, Any]]:
        """Find threshold crossing events"""

        events = []
        values = sensor_data['normalized_value'].values
        timestamps = sensor_data['timestamp'].values

        if event_type == 'high':
            above_threshold = values > threshold
        else:
            above_threshold = values < threshold

        # Find start and end of threshold crossing periods
        crossings = np.diff(above_threshold.astype(int))
        starts = np.where(crossings == 1)[0] + 1
        ends = np.where(crossings == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])

        # Create events
        for start_idx, end_idx in zip(starts, ends):
            if end_idx <= start_idx:
                continue

            duration: float = pd.Timedelta(timestamps[end_idx - 1] - timestamps[start_idx]).total_seconds()

            if duration >= self.min_duration:
                # Determine activity based on sensor type and event characteristics
                activity = self._classify_activity(
                    sensor_id, event_type, values[start_idx:end_idx]
                )

                events.append({
                    'sensor_id': sensor_id,
                    'activity': activity,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx - 1],
                    'duration': duration,
                    'avg_value': np.mean(values[start_idx:end_idx]),
                    'max_value': np.max(values[start_idx:end_idx]),
                    'min_value': np.min(values[start_idx:end_idx]),
                    'event_type': 'threshold_crossing',
                    'threshold': threshold,
                    'crossing_type': event_type
                })

        return events

    def calculate_duration(self,
            timestamps: Union[np.ndarray, pd.Series], start_idx: int, end_idx: int
    ) -> float:
        """
        Calculate duration in seconds between two timestamps.

        :param timestamps: Array or Series of datetime-like objects.
        :param start_idx: Start index.
        :param end_idx: End index.
        :return: Duration in seconds as float.
        """
        if isinstance(timestamps, pd.Series):
            delta = timestamps.iloc[end_idx] - timestamps.iloc[start_idx]
        else:

            if end_idx == len(timestamps):
                end_idx = len(timestamps) - 1

            delta = timestamps[end_idx] - timestamps[start_idx]

        return float(delta / np.timedelta64(1, "s"))

    def _detect_peak_events(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect peak-based events"""

        events = []
        values = sensor_data['normalized_value'].values
        timestamps = sensor_data['timestamp'].values

        # Find peaks
        peaks, properties = find_peaks(
            values,
            height=np.percentile(values, 70),
            distance=max(5, len(values) // 20),  # Minimum distance between peaks
            width=2
        )

        # Create peak events
        for peak_idx in peaks:
            # Estimate event boundaries around peak
            start_idx = max(0, peak_idx - 5)
            end_idx = min(len(values), peak_idx + 5)

            # Refine boundaries based on value changes
            while start_idx > 0 and values[start_idx] > values[start_idx - 1]:
                start_idx -= 1
            while end_idx < len(values) - 1 and values[end_idx] > values[end_idx + 1]:
                end_idx += 1

            duration: float = self.calculate_duration(timestamps, start_idx, end_idx)

            if duration >= self.min_duration:
                activity = self._classify_peak_activity(sensor_id, values[peak_idx])

                events.append({
                    'sensor_id': sensor_id,
                    'activity': activity,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'duration': duration,
                    'avg_value': np.mean(values[start_idx:end_idx + 1]),
                    'max_value': values[peak_idx],
                    'peak_time': timestamps[peak_idx],
                    'event_type': 'peak',
                    'peak_prominence': properties.get('prominences', [0])[0] if len(
                        properties.get('prominences', [])) > 0 else 0
                })

        return events

    def _detect_pattern_events(self, sensor_data: pd.DataFrame, sensor_id: str) -> List[Dict[str, Any]]:
        """Detect pattern-based events using clustering"""

        events = []

        if len(sensor_data) < self.window_size * 2:
            return events

        # Create sliding windows of sensor values
        values = sensor_data['normalized_value'].values
        timestamps = sensor_data['timestamp'].values

        windows = []
        window_starts = []

        for i in range(len(values) - self.window_size + 1):
            window = values[i:i + self.window_size]
            windows.append(window)
            window_starts.append(i)

        if len(windows) < 3:
            return events

        # Cluster similar patterns
        windows_array = np.array(windows)

        # Use DBSCAN for pattern clustering
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(windows_array[~np.isnan(windows_array).any(axis=1)]
)
        labels = clustering.labels_

        # Find distinct patterns (non-noise clusters)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]

            if len(cluster_indices) < 2:
                continue

            # Merge consecutive windows of same pattern into events
            consecutive_groups = []
            current_group = [cluster_indices[0]]

            for i in range(1, len(cluster_indices)):
                if cluster_indices[i] - cluster_indices[i - 1] <= 2:  # Allow small gaps
                    current_group.append(cluster_indices[i])
                else:
                    consecutive_groups.append(current_group)
                    current_group = [cluster_indices[i]]
            consecutive_groups.append(current_group)

            # Create events from consecutive groups
            for group in consecutive_groups:
                if len(group) < 2:
                    continue

                start_window_idx = min(group)
                end_window_idx = max(group)

                start_idx = window_starts[start_window_idx]
                end_idx = window_starts[end_window_idx] + self.window_size - 1

                duration: float = (timestamps[end_idx] - timestamps[start_idx]) / np.timedelta64(1, "s")

                if duration >= self.min_duration:
                    # Get representative pattern
                    pattern_values = values[start_idx:end_idx + 1]
                    activity = self._classify_pattern_activity(sensor_id, pattern_values, label)

                    events.append({
                        'sensor_id': sensor_id,
                        'activity': activity,
                        'start_time': timestamps[start_idx],
                        'end_time': timestamps[end_idx],
                        'duration': duration,
                        'avg_value': np.mean(pattern_values),
                        'pattern_label': int(label),
                        'event_type': 'pattern',
                        'pattern_strength': len(group) / len(cluster_indices)
                    })

        return events

    def _classify_activity(self, sensor_id: str, event_type: str, values: np.ndarray) -> str:
        """Classify activity based on sensor type and event characteristics"""

        if 'PWR' in sensor_id:
            if event_type == 'high':
                avg_value = np.mean(values)
                if avg_value > 0.8:
                    return 'Welding_Station_Weld'
                elif avg_value > 0.5:
                    return 'Packaging_Station_Seal'
                else:
                    return 'Machine_Active'
            else:
                return 'Machine_Idle'

        elif 'TEMP' in sensor_id:
            if event_type == 'high':
                return 'Welding_Station_Weld'
            else:
                return 'Welding_Station_Cool'

        elif 'VIB' in sensor_id:
            if event_type == 'high':
                return 'Inspection_Station_Scan'
            else:
                return 'Machine_Stable'

        elif 'POS' in sensor_id:
            return 'Position_Change'

        return f'Activity_{event_type}'

    def _classify_peak_activity(self, sensor_id: str, peak_value: float) -> str:
        """Classify activity based on peak characteristics"""

        if 'PWR' in sensor_id:
            if peak_value > 0.9:
                return 'Welding_Station_Weld'
            elif peak_value > 0.6:
                return 'Machine_Startup'
            else:
                return 'Machine_Active'

        elif 'TEMP' in sensor_id:
            return 'Welding_Station_Weld'

        elif 'VIB' in sensor_id:
            return 'Inspection_Station_Measure'

        elif 'POS' in sensor_id:
            return 'Packaging_Station_Position'

        return 'Peak_Activity'

    def _classify_pattern_activity(self, sensor_id: str, pattern_values: np.ndarray, label: int) -> str:
        """Classify activity based on pattern characteristics"""

        pattern_std = np.std(pattern_values)
        pattern_trend = np.polyfit(range(len(pattern_values)), pattern_values, 1)[0]

        if 'PWR' in sensor_id:
            if pattern_std > 0.3:
                return 'Welding_Station_Weld'
            elif abs(pattern_trend) > 0.1:
                return 'Machine_Transition'
            else:
                return 'Machine_Steady'

        elif 'TEMP' in sensor_id:
            if pattern_trend > 0.1:
                return 'Welding_Station_Heat'
            elif pattern_trend < -0.1:
                return 'Welding_Station_Cool'
            else:
                return 'Temperature_Stable'

        return f'Pattern_{label}'

    def _merge_concurrent_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Merge events that occur concurrently across different sensors"""

        if events_df.empty:
            return events_df

        # Sort by start time
        events_df = events_df.sort_values('start_time').reset_index(drop=True)

        merged_events = []

        # Group events by case_id if available, otherwise use time windows
        time_window = timedelta(seconds=10)  # 10 second window for merging

        i = 0
        while i < len(events_df):
            current_event = events_df.iloc[i].to_dict()
            current_start = current_event['start_time']
            current_end = current_event['end_time']

            # Find concurrent events
            concurrent_events = [current_event]
            j = i + 1

            while j < len(events_df):
                next_event = events_df.iloc[j].to_dict()
                next_start = next_event['start_time']

                # Stop if next event starts too far in the future
                if next_start > current_end + time_window:
                    break

                # Check for temporal overlap
                next_end = next_event['end_time']
                if (next_start <= current_end + time_window and
                        next_end >= current_start - time_window):
                    concurrent_events.append(next_event)
                    current_end = max(current_end, next_end)
                    j += 1
                else:
                    j += 1

            # Create merged event if multiple sensors involved
            if len(concurrent_events) > 1:
                merged_event = self._create_merged_event(concurrent_events)
                merged_events.append(merged_event)
            else:
                merged_events.append(current_event)

            i = j if j > i else i + 1

        return pd.DataFrame(merged_events)

    def _create_merged_event(self, concurrent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a merged event from concurrent sensor events"""

        # Determine dominant activity
        activities = [event['activity'] for event in concurrent_events]
        activity_counts = pd.Series(activities).value_counts()
        dominant_activity = activity_counts.index[0]

        # Merge temporal bounds
        start_times = [event['start_time'] for event in concurrent_events]
        end_times = [event['end_time'] for event in concurrent_events]

        merged_start = min(start_times)
        merged_end = max(end_times)

        # Combine sensor information
        sensor_ids = [event['sensor_id'] for event in concurrent_events]

        merged_event = {
            'activity': dominant_activity,
            'start_time': merged_start,
            'end_time': merged_end,
            'duration': (merged_end - merged_start).total_seconds(),
            'sensor_ids': sensor_ids,
            'num_sensors': len(sensor_ids),
            'event_type': 'merged',
            'constituent_events': concurrent_events,
            'confidence': len([a for a in activities if a == dominant_activity]) / len(activities)
        }

        # Add aggregated values
        if all('avg_value' in event for event in concurrent_events):
            merged_event['avg_value'] = np.mean([event['avg_value'] for event in concurrent_events])

        return merged_event

    def _add_quality_indicators(self, events_df: pd.DataFrame,
                                preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Add quality indicators to events based on underlying sensor data"""

        if events_df.empty:
            return events_df

        events_df = events_df.copy()
        events_df['quality_issues'] = [[] for _ in range(len(events_df))]

        for idx, event in events_df.iterrows():
            quality_issues = []

            # Check for quality issues in underlying sensor data
            if 'sensor_ids' in event:
                sensor_ids = event['sensor_ids']
            else:
                sensor_ids = [event['sensor_id']]

            for sensor_id in sensor_ids:
                sensor_data = preprocessed_data[
                    (preprocessed_data['sensor_id'] == sensor_id) &
                    (preprocessed_data['timestamp'] >= event['start_time']) &
                    (preprocessed_data['timestamp'] <= event['end_time'])
                    ]

                # Check for quality flags in sensor data
                if 'quality_flags' in sensor_data.columns:
                    for _, row in sensor_data.iterrows():
                        if isinstance(row['quality_flags'], dict):
                            for flag, value in row['quality_flags'].items():
                                if value:
                                    quality_issues.append(f"{sensor_id}_{flag}")

            # Add event-level quality assessments
            if event.get('duration', 0) < self.min_duration / 2:
                quality_issues.append('event_too_short')

            if event.get('confidence', 1.0) < 0.5:
                quality_issues.append('low_confidence')

            events_df.at[idx, 'quality_issues'] = quality_issues

        return events_df