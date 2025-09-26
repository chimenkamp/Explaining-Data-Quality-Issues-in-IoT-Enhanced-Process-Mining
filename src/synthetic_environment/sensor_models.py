import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class SensorReading:
    """Individual sensor reading"""
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    case_id: int
    activity: str
    quality_flags: Dict[str, Any]


class BaseSensor(ABC):
    """Base class for all sensor types"""

    def __init__(self, sensor_id: str, sampling_rate: float, location: Tuple[float, float],
                 noise_level: float = 0.01):
        self.sensor_id = sensor_id
        self.sampling_rate = sampling_rate
        self.location = location
        self.noise_level = noise_level

        # Quality issue flags
        self.has_placement_issues = False
        self.has_range_issues = False
        self.has_volume_issues = False

        # Quality issue parameters
        self.placement_overlap = 0.0
        self.placement_inconsistency = 0.0
        self.range_reduction = 1.0
        self.blind_spots = 0.0
        self.drop_probability = 0.0
        self.timestamp_drift_std = 0.0

    @abstractmethod
    def get_expected_value(self, activity: str, time_in_activity: float) -> float:
        """Get expected sensor value for given activity and time"""
        pass

    @abstractmethod
    def get_unit(self) -> str:
        """Get sensor measurement unit"""
        pass


class PowerSensor(BaseSensor):
    """Power consumption sensor"""

    def get_expected_value(self, activity: str, time_in_activity: float) -> float:
        """Power consumption patterns based on activity"""
        base_power = 100.0  # Watts

        if 'Weld' in activity:
            # High power spike during welding
            return base_power + 1500.0 + 200.0 * np.sin(time_in_activity * 10)
        elif 'Position' in activity:
            return base_power + 50.0
        elif 'Package' in activity or 'Seal' in activity:
            return base_power + 300.0
        elif 'Pick' in activity:
            return base_power + 150.0
        else:
            return base_power + np.random.normal(0, 10)

    def get_unit(self) -> str:
        return "W"


class TemperatureSensor(BaseSensor):
    """Temperature sensor"""

    def get_expected_value(self, activity: str, time_in_activity: float) -> float:
        """Temperature patterns based on activity"""
        ambient_temp = 22.0  # Celsius

        if 'Weld' in activity:
            # Temperature rise during welding
            return ambient_temp + 150.0 * (1 - np.exp(-time_in_activity / 5.0))
        elif 'Cool' in activity:
            # Exponential cooling
            return ambient_temp + 100.0 * np.exp(-time_in_activity / 10.0)
        else:
            return ambient_temp + np.random.normal(0, 2)

    def get_unit(self) -> str:
        return "°C"


class VibrationSensor(BaseSensor):
    """Vibration sensor (accelerometer)"""

    def get_expected_value(self, activity: str, time_in_activity: float) -> float:
        """Vibration patterns based on activity"""
        base_vibration = 0.1  # g

        if 'Scan' in activity or 'Measure' in activity:
            # High precision movement causes specific vibration
            return base_vibration + 0.5 * np.sin(time_in_activity * 15)
        elif 'Position' in activity:
            return base_vibration + 0.3
        elif 'Package' in activity:
            return base_vibration + 0.8 * np.random.random()
        else:
            return base_vibration + np.random.normal(0, 0.05)

    def get_unit(self) -> str:
        return "g"


class PositionSensor(BaseSensor):
    """Position sensor (encoder/GPS-like)"""

    def get_expected_value(self, activity: str, time_in_activity: float) -> float:
        """Position patterns based on activity"""
        if 'Position' in activity:
            # Movement to target position
            target_pos = 100.0 + hash(activity) % 500
            return target_pos * (1 - np.exp(-time_in_activity / 2.0))
        elif 'Pick' in activity:
            return 50.0 + 10 * np.sin(time_in_activity * 5)
        else:
            # Stable position with small drift
            return 100.0 + np.random.normal(0, 1)

    def get_unit(self) -> str:
        return "mm"