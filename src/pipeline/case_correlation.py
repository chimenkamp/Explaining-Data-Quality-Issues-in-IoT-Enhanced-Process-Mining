import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx


class CaseCorrelator:
    """Correlates events into process instances (cases)"""

    def __init__(self, case_timeout: float = 300.0, min_events: int = 2):
        self.case_timeout = case_timeout  # seconds
        self.min_events = min_events

    def correlate_cases(self, structured_events: pd.DataFrame) -> pd.DataFrame:
        """Correlate structured events into process instances"""

        if structured_events.empty:
            return pd.DataFrame()

        # Sort events by start time
        events = structured_events.sort_values('start_time').reset_index(drop=True)

        # Apply different correlation strategies
        cases = []

        # Strategy 1: Temporal proximity correlation
        temporal_cases = self._correlate_by_temporal_proximity(events)
        cases.extend(temporal_cases)

        # Strategy 2: Activity sequence correlation
        sequence_cases = self._correlate_by_activity_sequence(events)
        cases.extend(sequence_cases)

        # Strategy 3: Resource-based correlation (same machine/station)
        resource_cases = self._correlate_by_resource_usage(events)
        cases.extend(resource_cases)

        # Merge and deduplicate cases
        merged_cases = self._merge_overlapping_cases(cases)

        # Create case instances DataFrame
        case_instances = self._create_case_instances(merged_cases, events)

        # Add quality assessment for cases
        case_instances = self._assess_case_quality(case_instances)

        return case_instances

    def _correlate_by_temporal_proximity(self, events: pd.DataFrame) -> List[List[int]]:
        """Correlate events based on temporal proximity"""

        cases = []
        used_events = set()

        for i, event in events.iterrows():
            if i in used_events:
                continue

            current_case = [i]
            current_end_time = event['end_time']

            # Look for subsequent events within timeout window
            for j, next_event in events.iloc[i + 1:].iterrows():
                if j in used_events:
                    continue

                time_gap = (next_event['start_time'] - current_end_time).total_seconds()

                if time_gap <= self.case_timeout:
                    current_case.append(j)
                    current_end_time = max(current_end_time, next_event['end_time'])
                    used_events.add(j)
                elif time_gap > self.case_timeout * 2:
                    # Gap too large, stop looking
                    break

            if len(current_case) >= self.min_events:
                cases.append(current_case)
                used_events.update(current_case)

        return cases

    def _correlate_by_activity_sequence(self, events: pd.DataFrame) -> List[List[int]]:
        """Correlate events based on typical activity sequences"""

        # Define expected activity sequences for different processes
        expected_sequences = [
            ['Welding_Station_Position', 'Welding_Station_Weld', 'Welding_Station_Cool'],
            ['Inspection_Station_Position', 'Inspection_Station_Scan', 'Inspection_Station_Measure',
             'Inspection_Station_Validate'],
            ['Packaging_Station_Pick', 'Packaging_Station_Package', 'Packaging_Station_Label',
             'Packaging_Station_Seal'],
            ['Machine_Startup', 'Machine_Active', 'Machine_Idle'],
            ['Position_Change', 'Machine_Active', 'Machine_Steady']
        ]

        cases = []

        for sequence in expected_sequences:
            sequence_cases = self._find_sequence_instances(events, sequence)
            cases.extend(sequence_cases)

        return cases

    def _find_sequence_instances(self, events: pd.DataFrame, sequence: List[str]) -> List[List[int]]:
        """Find instances of a specific activity sequence"""

        cases = []
        sequence_length = len(sequence)

        if sequence_length == 0:
            return cases

        # Find all events matching the first activity in sequence
        first_activity = sequence[0]
        start_events = events[events['activity'].str.contains(first_activity, na=False)]

        for start_idx, start_event in start_events.iterrows():
            current_case = [start_idx]
            current_time = start_event['end_time']
            sequence_pos = 1

            # Look for subsequent activities in sequence
            remaining_events = events[events.index > start_idx]

            for event_idx, event in remaining_events.iterrows():
                if sequence_pos >= sequence_length:
                    break

                time_gap = (event['start_time'] - current_time).total_seconds()

                if time_gap > self.case_timeout:
                    break

                expected_activity = sequence[sequence_pos]
                if expected_activity in event['activity']:
                    current_case.append(event_idx)
                    current_time = event['end_time']
                    sequence_pos += 1

                    if sequence_pos >= sequence_length:
                        # Complete sequence found
                        break

            # Add case if we found a reasonable portion of the sequence
            if len(current_case) >= max(2, sequence_length // 2):
                cases.append(current_case)

        return cases

    def _correlate_by_resource_usage(self, events: pd.DataFrame) -> List[List[int]]:
        """Correlate events based on resource (machine/station) usage"""

        cases = []

        # Group events by resource/machine
        for activity_prefix in ['Welding_Station', 'Inspection_Station', 'Packaging_Station']:
            station_events = events[events['activity'].str.startswith(activity_prefix, na=False)]

            if len(station_events) < 2:
                continue

            # Use temporal clustering within each station
            station_cases = self._temporal_cluster_events(station_events)
            cases.extend(station_cases)

        return cases

    def _temporal_cluster_events(self, station_events: pd.DataFrame) -> List[List[int]]:
        """Cluster events temporally within a station"""

        if len(station_events) < 2:
            return []

        # Create time-based features for clustering
        times = station_events['start_time'].values
        time_numeric = np.array([(t - times[0]).total_seconds() for t in times]).reshape(-1, 1)

        # Use DBSCAN for temporal clustering
        clustering = DBSCAN(
            eps=self.case_timeout,  # Maximum time gap within cluster
            min_samples=self.min_events
        ).fit(time_numeric)

        cases = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise points
                continue

            cluster_indices = station_events.iloc[clustering.labels_ == label].index.tolist()
            if len(cluster_indices) >= self.min_events:
                cases.append(cluster_indices)

        return cases

    def _merge_overlapping_cases(self, all_cases: List[List[int]]) -> List[List[int]]:
        """Merge cases that have significant overlap in events"""

        if not all_cases:
            return []

        merged_cases = []
        used_cases = set()

        for i, case1 in enumerate(all_cases):
            if i in used_cases:
                continue

            current_case = set(case1)

            # Look for overlapping cases
            for j, case2 in enumerate(all_cases[i + 1:], i + 1):
                if j in used_cases:
                    continue

                overlap = len(set(case1) & set(case2))
                union_size = len(set(case1) | set(case2))

                # Merge if significant overlap (Jaccard similarity > 0.3)
                if overlap > 0 and overlap / union_size > 0.3:
                    current_case.update(case2)
                    used_cases.add(j)

            if len(current_case) >= self.min_events:
                merged_cases.append(list(current_case))
                used_cases.add(i)

        return merged_cases

    def _create_case_instances(self, cases: List[List[int]], events: pd.DataFrame) -> pd.DataFrame:
        """Create case instances DataFrame from correlated events"""

        case_instances = []

        for case_id, event_indices in enumerate(cases):
            if not event_indices:
                continue

            case_events = events.iloc[event_indices].sort_values('start_time')

            # Calculate case statistics
            start_time = case_events['start_time'].min()
            end_time = case_events['end_time'].max()
            duration = (end_time - start_time).total_seconds()

            # Extract activity sequence
            activity_sequence = case_events['activity'].tolist()

            # Identify unique sensors involved
            sensors_involved = []
            for _, event in case_events.iterrows():
                if 'sensor_ids' in event and event['sensor_ids']:
                    sensors_involved.extend(event['sensor_ids'])
                elif 'sensor_id' in event and event['sensor_id']:
                    sensors_involved.append(event['sensor_id'])
            sensors_involved = list(set(sensors_involved))

            # Calculate quality metrics
            quality_issues = []
            for _, event in case_events.iterrows():
                if 'quality_issues' in event and event['quality_issues']:
                    quality_issues.extend(event['quality_issues'])

            case_instance = {
                'case_id': f"case_{case_id:04d}",
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'num_events': len(case_events),
                'activity_sequence': activity_sequence,
                'sensors_involved': sensors_involved,
                'num_sensors': len(sensors_involved),
                'event_indices': event_indices,
                'quality_issues': list(set(quality_issues)),
                'num_quality_issues': len(set(quality_issues))
            }

            case_instances.append(case_instance)

        return pd.DataFrame(case_instances)

    def _assess_case_quality(self, case_instances: pd.DataFrame) -> pd.DataFrame:
        """Assess quality of case correlation"""

        if case_instances.empty:
            return case_instances

        case_instances = case_instances.copy()
        case_instances['case_quality_score'] = 0.0
        case_instances['case_quality_issues'] = [[] for _ in range(len(case_instances))]

        for idx, case in case_instances.iterrows():
            quality_score = 1.0
            quality_issues = []

            # Check for minimum events
            if case['num_events'] < self.min_events:
                quality_score -= 0.3
                quality_issues.append('insufficient_events')

            # Check for reasonable duration
            if case['duration'] < 10:  # Less than 10 seconds
                quality_score -= 0.2
                quality_issues.append('case_too_short')
            elif case['duration'] > 3600:  # More than 1 hour
                quality_score -= 0.1
                quality_issues.append('case_too_long')

            # Check for activity sequence coherence
            activity_sequence = case['activity_sequence']
            if len(set(activity_sequence)) == 1:
                # Only one type of activity - might be fragmented
                quality_score -= 0.2
                quality_issues.append('single_activity_type')

            # Check for sensor coverage
            if case['num_sensors'] < 2:
                quality_score -= 0.1
                quality_issues.append('limited_sensor_coverage')

            # Factor in underlying event quality issues
            if case['num_quality_issues'] > 0:
                quality_penalty = min(0.3, case['num_quality_issues'] * 0.05)
                quality_score -= quality_penalty
                quality_issues.append('underlying_event_quality_issues')

            case_instances.at[idx, 'case_quality_score'] = max(0.0, quality_score)
            case_instances.at[idx, 'case_quality_issues'] = quality_issues

        return case_instances