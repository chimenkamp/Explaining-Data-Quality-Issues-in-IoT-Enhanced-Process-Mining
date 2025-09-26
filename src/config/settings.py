"""
Configuration settings for the IoT Data Quality Pipeline
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np


@dataclass
class QualityIssueConfig:
    """Configuration for different types of quality issues"""

    # C1: Inadequate Sampling Rate
    inadequate_sampling_rate: Dict[str, Any] = field(default_factory=lambda: {
        'probability': 0.15,
        'min_sampling_rate': 0.1,  # Hz
        'max_sampling_rate': 0.5  # Hz
    })

    # C2: Poor Sensor Placement
    poor_sensor_placement: Dict[str, Any] = field(default_factory=lambda: {
        'probability': 0.10,
        'overlap_factor': 0.3,
        'inconsistency_factor': 0.4
    })

    # C3: Sensor Noise & Outliers
    sensor_noise: Dict[str, Any] = field(default_factory=lambda: {
        'probability': 0.25,
        'noise_std': 0.1,
        'outlier_probability': 0.05,
        'outlier_magnitude': 3.0
    })

    # C4: Sensor Range Too Small
    sensor_range_too_small: Dict[str, Any] = field(default_factory=lambda: {
        'probability': 0.12,
        'blind_spot_fraction': 0.2,
        'range_reduction_factor': 0.6
    })

    # C5: High Data Volume/Velocity
    high_data_volume: Dict[str, Any] = field(default_factory=lambda: {
        'probability': 0.08,
        'drop_probability': 0.15,
        'timestamp_drift_std': 2.0  # seconds
    })


@dataclass
class PipelineConfig:
    """Configuration for pipeline stages"""

    # Event abstraction parameters
    event_window_size: int = 10
    min_event_duration: float = 1.0

    # Case correlation parameters
    case_timeout: float = 300.0  # 5 minutes
    min_case_events: int = 2

    # Process mining parameters
    noise_threshold: float = 0.05
    dependency_threshold: float = 0.1

    # Probabilistic reasoning
    prior_probabilities: Dict[str, float] = field(default_factory=lambda: {
        'C1_inadequate_sampling': 0.2,
        'C2_poor_placement': 0.15,
        'C3_sensor_noise': 0.3,
        'C4_range_too_small': 0.2,
        'C5_high_volume': 0.15
    })


@dataclass
class QualityThresholds:
    """Thresholds for quality issue detection"""

    # C1: Inadequate Sampling Rate thresholds
    inadequate_sampling: Dict[str, float] = field(default_factory=lambda: {
        'min_acceptable_rate_hz': 1.0,
        'gap_detection_multiplier': 3.0,  # Gap > median_interval * multiplier
        'low_rate_threshold_hz': 0.5
    })

    # C2: Poor Sensor Placement thresholds
    poor_placement: Dict[str, float] = field(default_factory=lambda: {
        'inconsistency_cv_threshold': 0.5,  # Coefficient of variation threshold
        'overlap_detection_seconds': 10.0,
        'placement_confidence_threshold': 0.7
    })

    # C3: Sensor Noise & Outliers thresholds
    sensor_noise: Dict[str, float] = field(default_factory=lambda: {
        'outlier_iqr_multiplier': 1.5,  # Standard IQR multiplier for outliers
        'extreme_outlier_iqr_multiplier': 3.0,
        'outlier_ratio_threshold': 0.05,  # 5% outlier ratio threshold
        'noise_level_threshold': 0.1
    })

    # C4: Sensor Range Too Small thresholds
    range_too_small: Dict[str, float] = field(default_factory=lambda: {
        'clipping_ratio_threshold': 0.1,  # 10% of readings at same value
        'range_to_std_threshold': 4.0,
        'blind_spot_detection_threshold': 0.05
    })

    # C5: High Data Volume thresholds
    high_volume: Dict[str, float] = field(default_factory=lambda: {
        'timing_cv_threshold': 1.0,  # Coefficient of variation for timing
        'drop_ratio_threshold': 0.2,  # 20% data drop threshold
        'expected_samples_per_hour': 3600  # Baseline expectation
    })

    # Process Mining thresholds
    process_mining: Dict[str, float] = field(default_factory=lambda: {
        'fitness_threshold': 0.6,
        'precision_threshold': 0.7,
        'complexity_threshold': 0.8,
        'min_case_events': 2,
        'case_timeout_seconds': 300.0
    })


@dataclass
class SensorConfig:
    """Configuration for sensor models"""

    # Default sensor parameters
    default_sampling_rate: float = 2.0  # Hz
    default_noise_level: float = 0.02

    # Sensor type specific configurations
    power_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'base_power_w': 100.0,
        'welding_spike_w': 1500.0,
        'normal_variation_w': 10.0,
        'unit': 'W'
    })

    temperature_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'ambient_temp_c': 22.0,
        'welding_temp_rise_c': 150.0,
        'cooling_time_constant': 10.0,
        'unit': '°C'
    })

    vibration_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'base_vibration_g': 0.1,
        'scan_vibration_g': 0.5,
        'random_variation_g': 0.05,
        'unit': 'g'
    })

    position_sensor: Dict[str, Any] = field(default_factory=lambda: {
        'base_position_mm': 100.0,
        'movement_range_mm': 500.0,
        'positioning_time_s': 2.0,
        'unit': 'mm'
    })


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""

    # Color palettes
    quality_colors: Dict[str, str] = field(default_factory=lambda: {
        'C1_inadequate_sampling': '#FF6B6B',
        'C2_poor_placement': '#4ECDC4',
        'C3_sensor_noise': '#45B7D1',
        'C4_range_too_small': '#96CEB4',
        'C5_high_volume': '#FFEAA7',
        'normal': '#DDD6FE',
        'high_quality': '#10B981',
        'medium_quality': '#F59E0B',
        'low_quality': '#EF4444'
    })

    severity_colors: Dict[str, str] = field(default_factory=lambda: {
        'high': '#EF4444',
        'medium': '#F59E0B',
        'low': '#10B981'
    })

    # Plot settings
    figure_size: tuple = (12, 8)
    dpi: int = 150
    font_size: int = 12


@dataclass
class ExportConfig:
    """Configuration for data export"""

    # File formats
    supported_formats: List[str] = field(default_factory=lambda: [
        'json', 'csv', 'excel', 'pickle'
    ])

    # Export settings
    json_indent: int = 2
    csv_separator: str = ','
    excel_engine: str = 'openpyxl'

    # Compression settings
    compress_large_files: bool = True
    compression_threshold_mb: float = 10.0


@dataclass
class LoggingConfig:
    """Configuration for logging"""

    # Logging levels
    default_level: str = 'INFO'
    file_level: str = 'DEBUG'
    console_level: str = 'INFO'

    # Log format
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'

    # File settings
    max_file_size_mb: float = 10.0
    backup_count: int = 5
    log_filename: str = 'iot_pipeline.log'


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""

    # Memory monitoring
    memory_warning_threshold_mb: float = 1000.0
    memory_critical_threshold_mb: float = 2000.0

    # Timing thresholds
    slow_operation_threshold_seconds: float = 10.0
    very_slow_operation_threshold_seconds: float = 60.0

    # Throughput expectations
    min_expected_throughput_records_per_second: float = 100.0
    target_throughput_records_per_second: float = 1000.0


@dataclass
class ValidationConfig:
    """Configuration for validation and testing"""

    # Test data generation
    test_data_seed: int = 42
    test_scenarios: List[str] = field(default_factory=lambda: [
        'clean_data', 'c1_sampling_issues', 'c3_noise_issues', 'mixed_issues'
    ])

    # Validation thresholds
    min_acceptable_precision: float = 0.7
    min_acceptable_recall: float = 0.7
    min_acceptable_f1: float = 0.7

    # Performance benchmarks
    max_acceptable_processing_time_per_1k_records: float = 5.0  # seconds
    max_acceptable_memory_per_1k_records_mb: float = 50.0


# Global configuration instances
QUALITY_CONFIG = QualityIssueConfig()
PIPELINE_CONFIG = PipelineConfig()
QUALITY_THRESHOLDS = QualityThresholds()
SENSOR_CONFIG = SensorConfig()
VISUALIZATION_CONFIG = VisualizationConfig()
EXPORT_CONFIG = ExportConfig()
LOGGING_CONFIG = LoggingConfig()
PERFORMANCE_CONFIG = PerformanceConfig()
VALIDATION_CONFIG = ValidationConfig()

# Configuration registry for easy access
CONFIG_REGISTRY = {
    'quality': QUALITY_CONFIG,
    'pipeline': PIPELINE_CONFIG,
    'thresholds': QUALITY_THRESHOLDS,
    'sensors': SENSOR_CONFIG,
    'visualization': VISUALIZATION_CONFIG,
    'export': EXPORT_CONFIG,
    'logging': LOGGING_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'validation': VALIDATION_CONFIG
}


def get_config(config_name: str = None):
    """
    Get configuration object by name

    Args:
        config_name: Name of configuration to retrieve.
                    If None, returns entire registry.

    Returns:
        Configuration object or registry
    """
    if config_name is None:
        return CONFIG_REGISTRY

    return CONFIG_REGISTRY.get(config_name.lower())


def update_config(config_name: str, **kwargs):
    """
    Update configuration parameters

    Args:
        config_name: Name of configuration to update
        **kwargs: Configuration parameters to update
    """
    config = get_config(config_name)
    if config is not None:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: {config_name} config does not have parameter '{key}'")
    else:
        print(f"Warning: Configuration '{config_name}' not found")


def reset_config(config_name: str = None):
    """
    Reset configuration to default values

    Args:
        config_name: Name of configuration to reset.
                    If None, resets all configurations.
    """
    global CONFIG_REGISTRY

    if config_name is None:
        # Reset all configurations
        CONFIG_REGISTRY = {
            'quality': QualityIssueConfig(),
            'pipeline': PipelineConfig(),
            'thresholds': QualityThresholds(),
            'sensors': SensorConfig(),
            'visualization': VisualizationConfig(),
            'export': ExportConfig(),
            'logging': LoggingConfig(),
            'performance': PerformanceConfig(),
            'validation': ValidationConfig()
        }
    else:
        # Reset specific configuration
        config_classes = {
            'quality': QualityIssueConfig,
            'pipeline': PipelineConfig,
            'thresholds': QualityThresholds,
            'sensors': SensorConfig,
            'visualization': VisualizationConfig,
            'export': ExportConfig,
            'logging': LoggingConfig,
            'performance': PerformanceConfig,
            'validation': ValidationConfig
        }

        config_class = config_classes.get(config_name.lower())
        if config_class:
            CONFIG_REGISTRY[config_name.lower()] = config_class()
        else:
            print(f"Warning: Configuration '{config_name}' not found")


def print_config_summary():
    """Print summary of all configurations"""

    print("IoT Data Quality Pipeline - Configuration Summary")
    print("=" * 60)

    for name, config in CONFIG_REGISTRY.items():
        print(f"\n{name.upper()} CONFIG:")
        print("-" * 30)

        # Print first few attributes of each config
        attrs = [attr for attr in dir(config) if not attr.startswith('_')][:5]
        for attr in attrs:
            value = getattr(config, attr)
            if isinstance(value, dict):
                print(f"  {attr}: {list(value.keys())[:3]}...")
            elif isinstance(value, list):
                print(f"  {attr}: [{len(value)} items]")
            else:
                print(f"  {attr}: {value}")

        if len([attr for attr in dir(config) if not attr.startswith('_')]) > 5:
            print(f"  ... and {len([attr for attr in dir(config) if not attr.startswith('_')]) - 5} more")


# Environment-specific configurations
def get_development_config():
    """Get configuration optimized for development"""
    config = QualityIssueConfig()
    # Increase quality issue probabilities for better testing
    config.inadequate_sampling_rate['probability'] = 0.3
    config.sensor_noise['probability'] = 0.4
    config.poor_sensor_placement['probability'] = 0.2
    return config


def get_production_config():
    """Get configuration optimized for production"""
    config = QualityIssueConfig()
    # Use more conservative probabilities for production
    config.inadequate_sampling_rate['probability'] = 0.1
    config.sensor_noise['probability'] = 0.15
    config.poor_sensor_placement['probability'] = 0.05
    return config


def get_testing_config():
    """Get configuration optimized for testing"""
    config = QualityIssueConfig()
    # Use high probabilities to ensure issues are generated for testing
    config.inadequate_sampling_rate['probability'] = 0.5
    config.sensor_noise['probability'] = 0.6
    config.poor_sensor_placement['probability'] = 0.4
    config.sensor_range_too_small['probability'] = 0.3
    config.high_data_volume['probability'] = 0.2
    return config


# Utility functions for configuration management
def save_config_to_file(filename: str, config_name: str = None):
    """Save configuration to file"""
    import json

    if config_name:
        config = get_config(config_name)
        if config is None:
            raise ValueError(f"Configuration '{config_name}' not found")
        config_data = {config_name: config.__dict__}
    else:
        config_data = {name: config.__dict__ for name, config in CONFIG_REGISTRY.items()}

    with open(filename, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)


def load_config_from_file(filename: str):
    """Load configuration from file"""
    import json

    with open(filename, 'r') as f:
        config_data = json.load(f)

    for config_name, config_dict in config_data.items():
        config = get_config(config_name)
        if config:
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)


if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()