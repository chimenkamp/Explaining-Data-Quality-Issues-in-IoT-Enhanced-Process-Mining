"""Synthetic IoT environment with ground truth Petri net"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.sim import play_out

from .sensor_models import BaseSensor, PowerSensor, TemperatureSensor, VibrationSensor, PositionSensor
from .data_generator import DataGenerator
from .ground_truth_model import (
    create_manufacturing_ground_truth,
    get_activity_sensor_mapping,
    get_activity_duration_ranges
)
from ..config.settings import QUALITY_CONFIG


@dataclass
class IoTEnvironment:
    """Synthetic IoT environment with ground truth process model"""

    def __init__(self, name: str, duration_hours: int = 8, num_cases: int = 20):
        self.name = name
        self.duration_hours = duration_hours
        self.num_cases = num_cases
        self.start_time = datetime.now()

        self.ground_truth_net, self.initial_marking, self.final_marking = create_manufacturing_ground_truth()
        self.activity_sensor_mapping = get_activity_sensor_mapping()
        self.activity_duration_ranges = get_activity_duration_ranges()

        self.sensors = self._create_sensors()
        self.data_generator = DataGenerator()

    def _create_sensors(self) -> Dict[str, BaseSensor]:
        """
        Create sensors with quality issues.

        :return: Dictionary mapping sensor IDs to sensor objects
        """
        sensors = {}

        sensor_types = {
            'PWR': PowerSensor,
            'TEMP': TemperatureSensor,
            'VIB': VibrationSensor,
            'POS': PositionSensor
        }

        for sensor_type, sensor_class in sensor_types.items():
            sensor_id = f"{sensor_type}_01"
            sensor = sensor_class(
                sensor_id=sensor_id,
                sampling_rate=self._get_sampling_rate_with_issues(),
                location=(0, 0),
                noise_level=self._get_noise_level()
            )
            self._apply_quality_issues(sensor)
            sensors[sensor_id] = sensor

        return sensors

    def generate_data(self) -> Dict[str, Any]:
        """
        Generate synthetic data based on ground truth Petri net.

        :return: Dictionary with raw_data, process_instances, and ground_truth
        """
        event_log = play_out(
            self.ground_truth_net, self.initial_marking, self.final_marking,
        )

        all_raw_data = []
        process_instances = []

        for case_idx, trace in enumerate(event_log):
            case_id = case_idx + 1
            case_data = self._generate_case_from_trace(case_id, trace)

            all_raw_data.extend(case_data['raw_readings'])
            process_instances.append(case_data['process_instance'])

        raw_data_df = pd.DataFrame(all_raw_data)
        if not raw_data_df.empty:
            raw_data_df = raw_data_df.sort_values('timestamp').reset_index(drop=True)

        return {
            'raw_data': raw_data_df,
            'process_instances': process_instances,
            'ground_truth': {
                'petri_net': self.ground_truth_net,
                'initial_marking': self.initial_marking,
                'final_marking': self.final_marking
            },
            'environment': self
        }

    def _generate_case_from_trace(self, case_id: int, trace: Any) -> Dict[str, Any]:
        """
        Generate IoT data for a single trace.

        :param case_id: Case identifier
        :param trace: Trace from event log
        :return: Dictionary with raw readings and process instance
        """
        raw_readings = []
        events = []

        current_time = self.start_time + timedelta(minutes=case_id * 3)

        for event in trace:
            activity = event['concept:name']

            min_dur, max_dur = self.activity_duration_ranges.get(activity, (5.0, 10.0))
            duration = np.random.uniform(min_dur, max_dur)

            step_start = current_time
            step_end = current_time + timedelta(seconds=duration)

            sensor_types = self.activity_sensor_mapping.get(activity, ['PWR'])

            for sensor_type in sensor_types:
                sensor_id = f"{sensor_type}_01"
                if sensor_id in self.sensors:
                    sensor = self.sensors[sensor_id]
                    readings = self.data_generator.generate_sensor_readings(
                        sensor, activity, step_start, step_end, case_id
                    )
                    raw_readings.extend(readings)

            events.append({
                'case_id': case_id,
                'activity': activity,
                'start_time': step_start,
                'end_time': step_end
            })

            current_time = step_end + timedelta(seconds=np.random.uniform(0.5, 2.0))

        return {
            'raw_readings': raw_readings,
            'process_instance': {
                'case_id': case_id,
                'start_time': events[0]['start_time'] if events else current_time,
                'end_time': events[-1]['end_time'] if events else current_time,
                'events': events,
                'ground_truth_trace': [e['concept:name'] for e in trace]
            }
        }

    def _get_sampling_rate_with_issues(self) -> float:
        """
        Get sampling rate with potential C1 issues.

        :return: Sampling rate in Hz
        """
        base_rate = 2.0

        if np.random.random() < QUALITY_CONFIG.inadequate_sampling_rate['probability']:
            min_rate = QUALITY_CONFIG.inadequate_sampling_rate['min_sampling_rate']
            max_rate = QUALITY_CONFIG.inadequate_sampling_rate['max_sampling_rate']
            return np.random.uniform(min_rate, max_rate)

        return base_rate + np.random.normal(0, 0.2)

    def _get_noise_level(self) -> float:
        """
        Get noise level with potential C3 issues.

        :return: Noise level
        """
        base_noise = 0.02

        if np.random.random() < QUALITY_CONFIG.sensor_noise['probability']:
            return base_noise + np.random.uniform(0.05, 0.2)

        return base_noise

    def _apply_quality_issues(self, sensor: BaseSensor) -> None:
        """
        Apply quality issues to sensor.

        :param sensor: Sensor to modify
        """
        if np.random.random() < QUALITY_CONFIG.poor_sensor_placement['probability']:
            sensor.has_placement_issues = True
            sensor.placement_overlap = QUALITY_CONFIG.poor_sensor_placement['overlap_factor']
            sensor.placement_inconsistency = QUALITY_CONFIG.poor_sensor_placement['inconsistency_factor']

        if np.random.random() < QUALITY_CONFIG.sensor_range_too_small['probability']:
            sensor.has_range_issues = True
            sensor.range_reduction = QUALITY_CONFIG.sensor_range_too_small['range_reduction_factor']
            sensor.blind_spots = QUALITY_CONFIG.sensor_range_too_small['blind_spot_fraction']

        if np.random.random() < QUALITY_CONFIG.high_data_volume['probability']:
            sensor.has_volume_issues = True
            sensor.drop_probability = QUALITY_CONFIG.high_data_volume['drop_probability']
            sensor.timestamp_drift_std = QUALITY_CONFIG.high_data_volume['timestamp_drift_std']

    def add_welding_station(self) -> None:
        """Compatibility method - does nothing in new implementation"""
        pass

    def add_inspection_station(self) -> None:
        """Compatibility method - does nothing in new implementation"""
        pass

    def add_packaging_station(self) -> None:
        """Compatibility method - does nothing in new implementation"""
        pass