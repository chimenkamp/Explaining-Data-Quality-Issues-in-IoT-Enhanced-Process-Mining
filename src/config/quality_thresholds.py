"""
Quality detection thresholds and configuration parameters
"""


class QualityThresholds:
    """Configurable thresholds for quality issue detection"""

    # C1: Inadequate Sampling Rate thresholds
    INADEQUATE_SAMPLING = {
        'min_acceptable_rate_hz': 1.0,
        'gap_detection_multiplier': 3.0,  # Gap > median_interval * multiplier
        'low_rate_threshold_hz': 0.5
    }

    # C2: Poor Sensor Placement thresholds
    POOR_PLACEMENT = {
        'inconsistency_cv_threshold': 0.5,  # Coefficient of variation threshold
        'overlap_detection_seconds': 10.0,
        'placement_confidence_threshold': 0.7
    }

    # C3: Sensor Noise & Outliers thresholds
    SENSOR_NOISE = {
        'outlier_iqr_multiplier': 1.5,  # Standard IQR multiplier for outliers
        'extreme_outlier_iqr_multiplier': 3.0,
        'outlier_ratio_threshold': 0.05,  # 5% outlier ratio threshold
        'noise_level_threshold': 0.1
    }

    # C4: Sensor Range Too Small thresholds
    RANGE_TOO_SMALL = {
        'clipping_ratio_threshold': 0.1,  # 10% of readings at same value
        'range_to_std_threshold': 4.0,
        'blind_spot_detection_threshold': 0.05
    }

    # C5: High Data Volume thresholds
    HIGH_VOLUME = {
        'timing_cv_threshold': 1.0,  # Coefficient of variation for timing
        'drop_ratio_threshold': 0.2,  # 20% data drop threshold
        'expected_samples_per_hour': 3600  # Baseline expectation
    }

    # Process Mining thresholds
    PROCESS_MINING = {
        'fitness_threshold': 0.6,
        'precision_threshold': 0.7,
        'complexity_threshold': 0.8,
        'min_case_events': 2,
        'case_timeout_seconds': 300.0
    }
