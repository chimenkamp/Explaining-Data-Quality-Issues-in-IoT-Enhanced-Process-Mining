"""Simplified case correlation using ground truth case IDs"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


class CaseCorrelator:
    """Correlates events into process instances using case IDs"""

    def __init__(self, case_timeout: float = 300.0, min_events: int = 2):
        self.case_timeout = case_timeout
        self.min_events = min_events

    def correlate_cases(self, structured_events: pd.DataFrame) -> pd.DataFrame:
        """
        Correlate structured events into process instances.

        :param structured_events: DataFrame of structured events
        :return: DataFrame of case instances
        """
        if structured_events.empty:
            return pd.DataFrame()

        case_instances = []

        case_id_col = self._find_case_id_column(structured_events)

        if case_id_col:
            case_instances = self._correlate_with_case_ids(structured_events, case_id_col)
        else:
            case_instances = self._correlate_without_case_ids(structured_events)

        if not case_instances:
            return pd.DataFrame()

        case_instances_df = pd.DataFrame(case_instances)
        case_instances_df = self._assess_case_quality(case_instances_df)
        start_time_ordered = case_instances_df.sort_values('start_time').reset_index(drop=True)
        return start_time_ordered

    def _find_case_id_column(self, events: pd.DataFrame) -> str:
        """
        Find case ID column in events DataFrame.

        :param events: Events DataFrame
        :return: Name of case ID column or empty string
        """
        possible_columns = ['case_id', 'case:concept:name', 'caseid', 'case']

        for col in possible_columns:
            if col in events.columns:
                return col

        return ''

    def _correlate_with_case_ids(
            self, events: pd.DataFrame, case_id_col: str
    ) -> List[Dict[str, Any]]:
        """
        Correlate events using existing case IDs.

        :param events: Events DataFrame
        :param case_id_col: Name of case ID column
        :return: List of case instances
        """
        case_instances = []

        for case_id in events[case_id_col].unique():
            case_events = events[events[case_id_col] == case_id].copy()
            case_events = case_events.sort_values('start_time')

            if len(case_events) < self.min_events:
                continue

            case_instance = self._create_case_instance(case_id, case_events)
            case_instances.append(case_instance)

        return case_instances

    def _correlate_without_case_ids(self, events: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Correlate events when case IDs are not available.

        :param events: Events DataFrame
        :return: List of case instances
        """
        case_instances = []
        events = events.sort_values('start_time').reset_index(drop=True)

        used_indices = set()
        case_id = 0

        for i in range(len(events)):
            if i in used_indices:
                continue

            case_id += 1
            current_case_indices = [i]
            current_end_time = events.iloc[i]['end_time']

            for j in range(i + 1, len(events)):
                if j in used_indices:
                    continue

                next_event = events.iloc[j]
                time_gap = (next_event['start_time'] - current_end_time).total_seconds()

                if time_gap <= self.case_timeout:
                    current_case_indices.append(j)
                    current_end_time = max(current_end_time, next_event['end_time'])
                    used_indices.add(j)
                elif time_gap > self.case_timeout * 2:
                    break

            if len(current_case_indices) >= self.min_events:
                case_events = events.iloc[current_case_indices]
                case_instance = self._create_case_instance(f"case_{case_id:04d}", case_events)
                case_instances.append(case_instance)
                used_indices.update(current_case_indices)

        return case_instances

    def _create_case_instance(
            self, case_id: Any, case_events: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create case instance from events.

        :param case_id: Case identifier
        :param case_events: Events belonging to this case
        :return: Case instance dictionary
        """
        start_time = case_events['start_time'].min()
        end_time = case_events['end_time'].max()
        duration = (end_time - start_time).total_seconds()

        activity_sequence = case_events['activity'].tolist()

        sensors_involved = []
        for _, event in case_events.iterrows():
            if 'sensor_ids' in event and isinstance(event['sensor_ids'], list):
                sensors_involved.extend(event['sensor_ids'])
            elif 'sensor_id' in event and pd.notna(event['sensor_id']):
                sensors_involved.append(event['sensor_id'])
        sensors_involved = list(set(sensors_involved))

        quality_issues = []
        for _, event in case_events.iterrows():
            if 'quality_issues' in event and isinstance(event['quality_issues'], list):
                quality_issues.extend(event['quality_issues'])
        quality_issues = list(set(quality_issues))

        return {
            'case_id': str(case_id),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'num_events': len(case_events),
            'activity_sequence': activity_sequence,
            'sensors_involved': sensors_involved,
            'num_sensors': len(sensors_involved),
            'event_indices': case_events.index.tolist(),
            'quality_issues': quality_issues,
            'num_quality_issues': len(quality_issues)
        }

    def _assess_case_quality(self, case_instances: pd.DataFrame) -> pd.DataFrame:
        """
        Assess quality of case correlation.

        :param case_instances: DataFrame of case instances
        :return: Case instances with quality scores
        """
        if case_instances.empty:
            return case_instances

        case_instances = case_instances.copy()
        case_instances['case_quality_score'] = 0.0
        case_instances['case_quality_issues'] = [[] for _ in range(len(case_instances))]

        for idx, case in case_instances.iterrows():
            quality_score = 1.0
            quality_issues = []

            if case['num_events'] < self.min_events:
                quality_score -= 0.3
                quality_issues.append('insufficient_events')

            if case['duration'] < 5:
                quality_score -= 0.2
                quality_issues.append('case_too_short')
            elif case['duration'] > 3600:
                quality_score -= 0.1
                quality_issues.append('case_too_long')

            activity_sequence = case['activity_sequence']
            if len(set(activity_sequence)) == 1:
                quality_score -= 0.2
                quality_issues.append('single_activity_type')

            if case['num_sensors'] < 1:
                quality_score -= 0.3
                quality_issues.append('no_sensor_coverage')

            if case['num_quality_issues'] > 0:
                quality_penalty = min(0.3, case['num_quality_issues'] * 0.05)
                quality_score -= quality_penalty
                quality_issues.append('underlying_event_quality_issues')

            case_instances.at[idx, 'case_quality_score'] = max(0.0, quality_score)
            case_instances.at[idx, 'case_quality_issues'] = quality_issues

        return case_instances