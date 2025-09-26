"""
Controlled injection of quality issues for testing and validation
"""
import numpy as np
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta


class QualityIssueInjector:
    """Injects specific quality issues into synthetic data for testing"""

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def inject_c1_inadequate_sampling(self, timestamps: List[datetime],
                                      severity: str = 'medium') -> List[datetime]:
        """Inject C1 inadequate sampling by removing timestamps"""

        if severity == 'high':
            removal_ratio = 0.4  # Remove 40% of samples
        elif severity == 'medium':
            removal_ratio = 0.2  # Remove 20% of samples
        else:  # low
            removal_ratio = 0.1  # Remove 10% of samples

        num_to_remove = int(len(timestamps) * removal_ratio)
        indices_to_remove = random.sample(range(len(timestamps)), num_to_remove)

        return [ts for i, ts in enumerate(timestamps) if i not in indices_to_remove]

    def inject_c3_sensor_noise(self, values: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Inject C3 sensor noise and outliers"""

        noisy_values = values.copy()

        if severity == 'high':
            noise_std = 0.2 * np.std(values)
            outlier_ratio = 0.15
            outlier_magnitude = 5.0
        elif severity == 'medium':
            noise_std = 0.1 * np.std(values)
            outlier_ratio = 0.08
            outlier_magnitude = 3.0
        else:  # low
            noise_std = 0.05 * np.std(values)
            outlier_ratio = 0.03
            outlier_magnitude = 2.0

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, len(values))
        noisy_values += noise

        # Add outliers
        num_outliers = int(len(values) * outlier_ratio)
        outlier_indices = random.sample(range(len(values)), num_outliers)

        for idx in outlier_indices:
            outlier_sign = random.choice([-1, 1])
            outlier_value = outlier_sign * outlier_magnitude * np.std(values)
            noisy_values[idx] += outlier_value

        return noisy_values

    def inject_c4_range_clipping(self, values: np.ndarray, severity: str = 'medium') -> np.ndarray:
        """Inject C4 range limitations by clipping values"""

        clipped_values = values.copy()

        if severity == 'high':
            clip_ratio = 0.3  # Clip to 30% of original range
        elif severity == 'medium':
            clip_ratio = 0.6  # Clip to 60% of original range
        else:  # low
            clip_ratio = 0.8  # Clip to 80% of original range

        value_min, value_max = values.min(), values.max()
        value_range = value_max - value_min
        center = (value_min + value_max) / 2

        new_range = value_range * clip_ratio
        new_min = center - new_range / 2
        new_max = center + new_range / 2

        return np.clip(clipped_values, new_min, new_max)