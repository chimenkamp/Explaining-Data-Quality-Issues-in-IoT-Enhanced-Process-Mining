import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import networkx as nx
from collections import defaultdict


class VisualizationHelper:
    """Helper utilities for creating advanced visualizations"""

    def __init__(self):
        # Set up color palettes
        self.quality_colors = {
            'C1_inadequate_sampling': '#FF6B6B',
            'C2_poor_placement': '#4ECDC4',
            'C3_sensor_noise': '#45B7D1',
            'C4_range_too_small': '#96CEB4',
            'C5_high_volume': '#FFEAA7'
        }

        self.severity_colors = {
            'high': '#EF4444',
            'medium': '#F59E0B',
            'low': '#10B981'
        }

    def create_sensor_heatmap(self, raw_data: pd.DataFrame, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create heatmap showing sensor activity and quality issues"""

        if raw_data.empty:
            return go.Figure()

        # Create sensor-time matrix
        raw_data['hour'] = raw_data['timestamp'].dt.hour
        sensor_activity = raw_data.groupby(['sensor_id', 'hour'])['value'].mean().reset_index()
        sensor_pivot = sensor_activity.pivot(index='sensor_id', columns='hour', values='value')

        # Create base heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sensor_pivot.values,
            x=sensor_pivot.columns,
            y=sensor_pivot.index,
            colorscale='Viridis',
            name='Sensor Activity'
        ))

        # Add quality issue annotations
        issue_sensors = [issue.get('sensor_id') for issue in quality_issues if issue.get('sensor_id')]

        # Highlight problematic sensors
        for i, sensor in enumerate(sensor_pivot.index):
            if sensor in issue_sensors:
                fig.add_shape(
                    type="rect",
                    x0=-0.5, y0=i - 0.5, x1=len(sensor_pivot.columns) - 0.5, y1=i + 0.5,
                    line=dict(color="red", width=3),
                    fillcolor="rgba(255,0,0,0.1)"
                )

        fig.update_layout(
            title='Sensor Activity Heatmap with Quality Issues',
            xaxis_title='Hour of Day',
            yaxis_title='Sensor ID'
        )

        return fig

    def create_issue_correlation_network(self, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create network graph showing relationships between quality issues"""

        # Build network based on co-occurring issues
        G = nx.Graph()

        # Group issues by sensor or time proximity
        sensor_issues = defaultdict(list)
        for issue in quality_issues:
            sensor_id = issue.get('sensor_id', 'unknown')
            sensor_issues[sensor_id].append(issue['type'])

        # Add nodes for each issue type
        issue_types = list(set([issue['type'] for issue in quality_issues]))
        for issue_type in issue_types:
            G.add_node(issue_type)

        # Add edges for co-occurring issues
        for sensor, issues in sensor_issues.items():
            if len(issues) > 1:
                for i in range(len(issues)):
                    for j in range(i + 1, len(issues)):
                        if G.has_edge(issues[i], issues[j]):
                            G[issues[i]][issues[j]]['weight'] += 1
                        else:
                            G.add_edge(issues[i], issues[j], weight=1)

        if G.number_of_nodes() == 0:
            return go.Figure()

        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_colors = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_colors.append(self.quality_colors.get(node, '#888'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title='Quality Issue Correlation Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Connected nodes represent issues that co-occur",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#888", size=10)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig

    def create_quality_evolution_chart(self, pipeline_results: Dict[str, Any]) -> go.Figure:
        """Create chart showing how quality issues evolve through pipeline stages"""

        propagation_results = pipeline_results.get('propagation_results', {})

        stages = ['preprocessing', 'event_abstraction', 'case_correlation', 'process_mining']
        issue_evolution = defaultdict(list)

        # Track issue evolution through stages
        for stage in stages:
            if stage in propagation_results:
                issues = propagation_results[stage].get('propagated_issues', [])

                stage_issue_counts = defaultdict(int)
                for issue in issues:
                    stage_issue_counts[issue['type']] += 1

                # Add counts for each issue type
                for issue_type in self.quality_colors.keys():
                    issue_evolution[issue_type].append(stage_issue_counts.get(issue_type, 0))

        # Create stacked bar chart
        fig = go.Figure()

        for issue_type, counts in issue_evolution.items():
            if any(counts):  # Only add if there are non-zero counts
                fig.add_trace(go.Bar(
                    name=issue_type,
                    x=stages,
                    y=counts,
                    marker_color=self.quality_colors.get(issue_type, '#888')
                ))

        fig.update_layout(
            title='Quality Issue Evolution Through Pipeline Stages',
            xaxis_title='Pipeline Stage',
            yaxis_title='Number of Issues',
            barmode='stack'
        )

        return fig

    def create_confidence_distribution(self, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create distribution plot of quality issue confidence scores"""

        if not quality_issues:
            return go.Figure()

        confidences = [issue.get('confidence', 0) for issue in quality_issues]
        issue_types = [issue['type'] for issue in quality_issues]

        # Create violin plot
        fig = go.Figure()

        for issue_type in set(issue_types):
            type_confidences = [conf for conf, itype in zip(confidences, issue_types) if itype == issue_type]

            fig.add_trace(go.Violin(
                y=type_confidences,
                name=issue_type,
                box_visible=True,
                meanline_visible=True,
                fillcolor=self.quality_colors.get(issue_type, '#888'),
                opacity=0.7
            ))

        fig.update_layout(
            title='Confidence Score Distribution by Issue Type',
            yaxis_title='Confidence Score',
            xaxis_title='Issue Type'
        )

        return fig

