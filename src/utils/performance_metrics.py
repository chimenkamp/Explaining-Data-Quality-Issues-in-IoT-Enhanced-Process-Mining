import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging


class PerformanceMetrics:
    """Tracks and analyzes pipeline performance metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'stage_timings': {},
            'memory_usage': {},
            'data_throughput': {},
            'quality_detection_efficiency': {}
        }
        self.start_times = {}

    def start_stage_timing(self, stage_name: str):
        """Start timing a pipeline stage"""
        self.start_times[stage_name] = time.time()

    def end_stage_timing(self, stage_name: str, data_size: int = None):
        """End timing a pipeline stage"""
        if stage_name not in self.start_times:
            self.logger.warning(f"No start time recorded for stage {stage_name}")
            return

        elapsed_time = time.time() - self.start_times[stage_name]

        self.metrics['stage_timings'][stage_name] = {
            'elapsed_time_seconds': elapsed_time,
            'data_size': data_size,
            'throughput_records_per_second': data_size / elapsed_time if data_size and elapsed_time > 0 else None
        }

        # Record memory usage
        process = psutil.Process()
        memory_info = process.memory_info()

        self.metrics['memory_usage'][stage_name] = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024  # Virtual Memory Size in MB
        }

        del self.start_times[stage_name]

    def record_quality_detection_metrics(self, stage_name: str,
                                         issues_detected: List[Dict[str, Any]],
                                         data_size: int):
        """Record quality detection efficiency metrics"""

        if not issues_detected:
            detection_rate = 0.0
            avg_confidence = 0.0
        else:
            detection_rate = len(issues_detected) / data_size if data_size > 0 else 0.0
            confidences = [issue.get('confidence', 0) for issue in issues_detected]
            avg_confidence = np.mean(confidences) if confidences else 0.0

        self.metrics['quality_detection_efficiency'][stage_name] = {
            'issues_detected': len(issues_detected),
            'detection_rate': detection_rate,
            'average_confidence': avg_confidence,
            'high_confidence_issues': len([i for i in issues_detected if i.get('confidence', 0) > 0.8])
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""

        summary = {
            'total_pipeline_time': sum([
                timing['elapsed_time_seconds']
                for timing in self.metrics['stage_timings'].values()
            ]),
            'peak_memory_usage_mb': max([
                memory['rss_mb']
                for memory in self.metrics['memory_usage'].values()
            ]) if self.metrics['memory_usage'] else 0,
            'stage_performance': {},
            'quality_detection_summary': {}
        }

        # Stage performance breakdown
        for stage, timing in self.metrics['stage_timings'].items():
            stage_perf = {
                'time_seconds': timing['elapsed_time_seconds'],
                'time_percentage': timing['elapsed_time_seconds'] / summary['total_pipeline_time'] * 100 if summary[
                                                                                                                'total_pipeline_time'] > 0 else 0
            }

            if timing['throughput_records_per_second']:
                stage_perf['throughput_records_per_sec'] = timing['throughput_records_per_second']

            if stage in self.metrics['memory_usage']:
                stage_perf['memory_mb'] = self.metrics['memory_usage'][stage]['rss_mb']

            summary['stage_performance'][stage] = stage_perf

        # Quality detection summary
        total_issues = sum([
            metrics['issues_detected']
            for metrics in self.metrics['quality_detection_efficiency'].values()
        ])

        total_high_conf_issues = sum([
            metrics['high_confidence_issues']
            for metrics in self.metrics['quality_detection_efficiency'].values()
        ])

        summary['quality_detection_summary'] = {
            'total_issues_detected': total_issues,
            'high_confidence_issues': total_high_conf_issues,
            'high_confidence_ratio': total_high_conf_issues / total_issues if total_issues > 0 else 0
        }

        return summary

    def export_metrics_to_csv(self, filename: str):
        """Export performance metrics to CSV"""

        # Create DataFrame with stage metrics
        stage_data = []

        for stage in self.metrics['stage_timings']:
            row = {'stage': stage}

            if stage in self.metrics['stage_timings']:
                timing = self.metrics['stage_timings'][stage]
                row.update({
                    'elapsed_time_seconds': timing['elapsed_time_seconds'],
                    'data_size': timing.get('data_size', 0),
                    'throughput_records_per_second': timing.get('throughput_records_per_second', 0)
                })

            if stage in self.metrics['memory_usage']:
                memory = self.metrics['memory_usage'][stage]
                row.update({
                    'memory_rss_mb': memory['rss_mb'],
                    'memory_vms_mb': memory['vms_mb']
                })

            if stage in self.metrics['quality_detection_efficiency']:
                quality = self.metrics['quality_detection_efficiency'][stage]
                row.update({
                    'issues_detected': quality['issues_detected'],
                    'detection_rate': quality['detection_rate'],
                    'average_confidence': quality['average_confidence']
                })

            stage_data.append(row)

        df = pd.DataFrame(stage_data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Performance metrics exported to {filename}")

