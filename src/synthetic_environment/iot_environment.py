import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from .sensor_models import BaseSensor, PowerSensor, TemperatureSensor, VibrationSensor, PositionSensor
from .data_generator import DataGenerator
from ..config.settings import QUALITY_CONFIG


@dataclass
class Machine:
    """Represents a machine in the IoT environment"""
    id: str
    name: str
    location: tuple
    sensors: List[BaseSensor]
    process_steps: List[str]
    cycle_time: float  # seconds


class IoTEnvironment:
    """Synthetic IoT environment generator for manufacturing processes"""

    def __init__(self, name: str, duration_hours: int = 8, num_machines: int = 3):
        self.name = name
        self.duration = timedelta(hours=duration_hours)
        self.start_time = datetime.now()
        self.end_time = self.start_time + self.duration
        self.machines: List[Machine] = []
        self.data_generator = DataGenerator()

    def add_welding_station(self):
        """Add a welding station with power and temperature sensors"""
        machine_id = f"WS_{len(self.machines) + 1:02d}"

        # Create sensors with potential quality issues
        power_sensor = PowerSensor(
            sensor_id=f"{machine_id}_PWR",
            sampling_rate=self._get_sampling_rate_with_issues(),
            location=(0, len(self.machines)),
            noise_level=self._get_noise_level()
        )

        temp_sensor = TemperatureSensor(
            sensor_id=f"{machine_id}_TEMP",
            sampling_rate=self._get_sampling_rate_with_issues(),
            location=(0, len(self.machines)),
            noise_level=self._get_noise_level()
        )

        # Apply quality issues
        self._apply_quality_issues(power_sensor)
        self._apply_quality_issues(temp_sensor)

        machine = Machine(
            id=machine_id,
            name=f"Welding Station {len(self.machines) + 1}",
            location=(0, len(self.machines)),
            sensors=[power_sensor, temp_sensor],
            process_steps=['Position', 'Weld', 'Cool'],
            cycle_time=45.0
        )

        self.machines.append(machine)

    def add_inspection_station(self):
        """Add an inspection station with vision and position sensors"""
        machine_id = f"IS_{len(self.machines) + 1:02d}"

        position_sensor = PositionSensor(
            sensor_id=f"{machine_id}_POS",
            sampling_rate=self._get_sampling_rate_with_issues(),
            location=(1, len(self.machines)),
            noise_level=self._get_noise_level()
        )

        vibration_sensor = VibrationSensor(
            sensor_id=f"{machine_id}_VIB",
            sampling_rate=self._get_sampling_rate_with_issues(),
            location=(1, len(self.machines)),
            noise_level=self._get_noise_level()
        )

        self._apply_quality_issues(position_sensor)
        self._apply_quality_issues(vibration_sensor)

        machine = Machine(
            id=machine_id,
            name=f"Inspection Station {len(self.machines) + 1}",
            location=(1, len(self.machines)),
            sensors=[position_sensor, vibration_sensor],
            process_steps=['Position', 'Scan', 'Measure', 'Validate'],
            cycle_time=30.0
        )

        self.machines.append(machine)

    def add_packaging_station(self):
        """Add a packaging station with multiple sensor types"""
        machine_id = f"PS_{len(self.machines) + 1:02d}"

        position_sensor = PositionSensor(
            sensor_id=f"{machine_id}_POS",
            sampling_rate=self._get_sampling_rate_with_issues(),
            location=(2, len(self.machines)),
            noise_level=self._get_noise_level()
        )

        power_sensor = PowerSensor(
            sensor_id=f"{machine_id}_PWR",
            sampling_rate=self._get_sampling_rate_with_issues(),
            location=(2, len(self.machines)),
            noise_level=self._get_noise_level()
        )

        self._apply_quality_issues(position_sensor)
        self._apply_quality_issues(power_sensor)

        machine = Machine(
            id=machine_id,
            name=f"Packaging Station {len(self.machines) + 1}",
            location=(2, len(self.machines)),
            sensors=[position_sensor, power_sensor],
            process_steps=['Pick', 'Package', 'Label', 'Seal'],
            cycle_time=25.0
        )

        self.machines.append(machine)

    def generate_data(self) -> Dict[str, Any]:
        """Generate synthetic IoT data with embedded quality issues"""
        all_raw_data = []
        process_instances = []

        current_time = self.start_time
        case_id = 0

        while current_time < self.end_time:
            # Generate a process instance (case)
            case_id += 1
            case_data = self._generate_case_instance(case_id, current_time)

            all_raw_data.extend(case_data['raw_readings'])
            process_instances.append(case_data['process_instance'])

            # Move to next case start time
            inter_arrival_time = np.random.exponential(60.0)  # Average 1 minute between cases
            current_time += timedelta(seconds=inter_arrival_time)

        # Convert to DataFrame
        raw_data_df = pd.DataFrame(all_raw_data)
        raw_data_df = raw_data_df.sort_values('timestamp').reset_index(drop=True)

        return {
            'raw_data': raw_data_df,
            'process_instances': process_instances,
            'environment': self
        }

    def _generate_case_instance(self, case_id: int, start_time: datetime) -> Dict[str, Any]:
        """Generate a single process instance (case) across all machines"""
        raw_readings = []
        events = []

        current_time = start_time

        for machine in self.machines:
            machine_start = current_time

            for step_idx, step in enumerate(machine.process_steps):
                step_duration = machine.cycle_time / len(machine.process_steps)
                step_duration += np.random.normal(0, step_duration * 0.1)  # Add variability

                step_start = current_time
                step_end = current_time + timedelta(seconds=step_duration)

                # Generate sensor readings for this step
                for sensor in machine.sensors:
                    readings = self.data_generator.generate_sensor_readings(
                        sensor, step, step_start, step_end, case_id
                    )
                    raw_readings.extend(readings)

                # Create process event
                events.append({
                    'case_id': case_id,
                    'activity': f"{machine.name}_{step}",
                    'start_time': step_start,
                    'end_time': step_end,
                    'machine_id': machine.id,
                    'step_index': step_idx
                })

                current_time = step_end

            # Add some delay between machines
            current_time += timedelta(seconds=np.random.exponential(10.0))

        return {
            'raw_readings': raw_readings,
            'process_instance': {
                'case_id': case_id,
                'start_time': start_time,
                'end_time': current_time,
                'events': events
            }
        }

    def _get_sampling_rate_with_issues(self) -> float:
        """Get sampling rate, potentially with C1 issues"""
        base_rate = 2.0  # 2 Hz default

        if np.random.random() < QUALITY_CONFIG.inadequate_sampling_rate['probability']:
            # Apply C1: Inadequate sampling rate
            min_rate = QUALITY_CONFIG.inadequate_sampling_rate['min_sampling_rate']
            max_rate = QUALITY_CONFIG.inadequate_sampling_rate['max_sampling_rate']
            return np.random.uniform(min_rate, max_rate)

        return base_rate + np.random.normal(0, 0.2)

    def _get_noise_level(self) -> float:
        """Get noise level, potentially with C3 issues"""
        base_noise = 0.02

        if np.random.random() < QUALITY_CONFIG.sensor_noise['probability']:
            # Apply C3: High noise
            return base_noise + np.random.uniform(0.05, 0.2)

        return base_noise

    def _apply_quality_issues(self, sensor: BaseSensor):
        """Apply various quality issues to a sensor"""

        # C2: Poor sensor placement
        if np.random.random() < QUALITY_CONFIG.poor_sensor_placement['probability']:
            sensor.has_placement_issues = True
            sensor.placement_overlap = QUALITY_CONFIG.poor_sensor_placement['overlap_factor']
            sensor.placement_inconsistency = QUALITY_CONFIG.poor_sensor_placement['inconsistency_factor']

        # C4: Sensor range too small
        if np.random.random() < QUALITY_CONFIG.sensor_range_too_small['probability']:
            sensor.has_range_issues = True
            sensor.range_reduction = QUALITY_CONFIG.sensor_range_too_small['range_reduction_factor']
            sensor.blind_spots = QUALITY_CONFIG.sensor_range_too_small['blind_spot_fraction']

        # C5: High data volume issues
        if np.random.random() < QUALITY_CONFIG.high_data_volume['probability']:
            sensor.has_volume_issues = True
            sensor.drop_probability = QUALITY_CONFIG.high_data_volume['drop_probability']
            sensor.timestamp_drift_std = QUALITY_CONFIG.high_data_volume['timestamp_drift_std']