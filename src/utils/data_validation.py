import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta


class DataValidator:
    """Validates data integrity throughout the pipeline"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}

    def validate_raw_data(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate raw sensor data"""

        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }

        if raw_data.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Raw data is empty')
            return validation_result

        # Check required columns
        required_columns = ['sensor_id', 'timestamp', 'value', 'unit', 'case_id', 'activity']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]

        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')

        # Check data types
        if 'timestamp' in raw_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(raw_data['timestamp']):
                validation_result['warnings'].append('Timestamp column is not datetime type')

        if 'value' in raw_data.columns:
            if not pd.api.types.is_numeric_dtype(raw_data['value']):
                validation_result['errors'].append('Value column contains non-numeric data')
                validation_result['is_valid'] = False

        # Check for missing values
        missing_counts = raw_data.isnull().sum()
        critical_missing = missing_counts[missing_counts > 0]

        if len(critical_missing) > 0:
            validation_result['warnings'].append(f'Missing values detected: {dict(critical_missing)}')

        # Statistical validation
        if 'value' in raw_data.columns and pd.api.types.is_numeric_dtype(raw_data['value']):
            values = raw_data['value']

            validation_result['statistics'] = {
                'value_count': len(values),
                'value_mean': float(values.mean()),
                'value_std': float(values.std()),
                'value_min': float(values.min()),
                'value_max': float(values.max()),
                'unique_sensors': raw_data['sensor_id'].nunique() if 'sensor_id' in raw_data.columns else 0,
                'time_span_hours': self._calculate_time_span_hours(raw_data) if 'timestamp' in raw_data.columns else 0
            }

            # Check for suspicious values
            if values.min() == values.max():
                validation_result['warnings'].append('All sensor values are identical')

            # Check for extreme outliers
            Q1, Q3 = values.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            extreme_outliers = ((values < Q1 - 3 * IQR) | (values > Q3 + 3 * IQR)).sum()

            if extreme_outliers > len(values) * 0.1:  # More than 10% extreme outliers
                validation_result['warnings'].append(
                    f'{extreme_outliers} extreme outliers detected ({extreme_outliers / len(values):.1%})')

        return validation_result

    def validate_pipeline_consistency(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency across pipeline stages"""

        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stage_consistency': {}
        }

        raw_data = pipeline_results.get('raw_data', pd.DataFrame())
        structured_events = pipeline_results.get('structured_events', pd.DataFrame())
        process_instances = pipeline_results.get('process_instances', pd.DataFrame())

        # Check data flow consistency
        if not raw_data.empty and not structured_events.empty:
            # Check if events span reasonable time range of raw data
            if 'timestamp' in raw_data.columns and 'start_time' in structured_events.columns:
                raw_time_span = (raw_data['timestamp'].max() - raw_data['timestamp'].min()).total_seconds()
                event_time_span = (
                            structured_events['end_time'].max() - structured_events['start_time'].min()).total_seconds()

                if event_time_span > raw_time_span * 1.5:  # Events span much longer than raw data
                    validation_result['warnings'].append('Event time span exceeds raw data time span significantly')

        # Check case consistency
        if not structured_events.empty and not process_instances.empty:
            if 'case_id' in structured_events.columns and 'case_id' in process_instances.columns:
                event_cases = set(structured_events['case_id'].unique())
                instance_cases = set(process_instances['case_id'].unique())

                missing_in_instances = event_cases - instance_cases
                if missing_in_instances:
                    validation_result['warnings'].append(
                        f'{len(missing_in_instances)} cases have events but no process instances')

        # Check quality issue consistency
        quality_issues = pipeline_results.get('quality_issues', [])
        if quality_issues:
            # Check that quality issues have required fields
            required_fields = ['type', 'description', 'confidence']
            for i, issue in enumerate(quality_issues):
                missing_fields = [field for field in required_fields if field not in issue]
                if missing_fields:
                    validation_result['warnings'].append(f'Quality issue {i} missing fields: {missing_fields}')

        return validation_result

    def _calculate_time_span_hours(self, data: pd.DataFrame) -> float:
        """Calculate time span in hours"""
        if 'timestamp' not in data.columns or data.empty:
            return 0.0

        time_span = data['timestamp'].max() - data['timestamp'].min()
        return time_span.total_seconds() / 3600

