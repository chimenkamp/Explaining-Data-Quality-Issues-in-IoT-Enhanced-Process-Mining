import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')


class EnhancedVisualizer:
    """Creates enhanced visualizations with quality issue information"""

    def __init__(self):
        self.color_palette = {
            'C1_inadequate_sampling': '#FF6B6B',
            'C2_poor_placement': '#4ECDC4',
            'C3_sensor_noise': '#45B7D1',
            'C4_range_too_small': '#96CEB4',
            'C5_high_volume': '#FFEAA7',
            'normal': '#DDD6FE',
            'high_quality': '#10B981',
            'medium_quality': '#F59E0B',
            'low_quality': '#EF4444'
        }

    def create_enhanced_visualization(self, process_model: Dict[str, Any],
                                      quality_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive visualization with quality annotations"""

        visualizations = {}

        # 1. Process Model Visualization with Quality Overlay
        visualizations['process_model'] = self._create_process_model_viz(
            process_model, quality_issues
        )

        # 2. Quality Issues Timeline
        visualizations['quality_timeline'] = self._create_quality_timeline(quality_issues)

        # 3. Quality Impact Dashboard
        visualizations['quality_dashboard'] = self._create_quality_dashboard(
            process_model, quality_issues
        )

        # 4. Information Gain Visualization
        visualizations['information_gain'] = self._create_information_gain_viz(quality_issues)

        # 5. Quality Propagation Flow
        visualizations['propagation_flow'] = self._create_propagation_flow_viz(quality_issues)

        return visualizations

    def _create_process_model_viz(self, process_model: Dict[str, Any],
                                  quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create process model visualization with quality annotations"""

        if not process_model or 'model' not in process_model:
            return go.Figure()

        model = process_model['model']
        graph = model['process_graph']

        if graph.number_of_nodes() == 0:
            return go.Figure()

        # Calculate layout
        try:
            pos = nx.spring_layout(graph, k=3, iterations=50)
        except:
            pos = {node: (i, 0) for i, node in enumerate(graph.nodes())}

        # Create traces for edges
        edge_traces = []

        for edge in graph.edges(data=True):
            source, target, data = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]

            relation_type = data.get('relation_type', 'causality')

            if relation_type == 'causality':
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=2, color='#374151'),
                    showlegend=False
                )
            else:  # parallel
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#9CA3AF', dash='dash'),
                    showlegend=False
                )

            edge_traces.append(edge_trace)

        # Create node traces with quality information
        node_trace = self._create_quality_annotated_nodes(
            graph, pos, quality_issues, model
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title="Process Model with Quality Issues",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node colors indicate quality issues. Hover for details.",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#888", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig

    def _create_quality_annotated_nodes(self, graph: nx.DiGraph, pos: Dict,
                                        quality_issues: List[Dict[str, Any]],
                                        model: Dict[str, Any]) -> go.Scatter:
        """Create node trace with quality annotations"""

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        hover_text = []

        # Map activities to quality issues
        activity_quality_map = defaultdict(list)
        for issue in quality_issues:
            # Extract affected activities from signatures or stage effects
            if 'stage_effects' in issue:
                effects = issue['stage_effects']
                if any(key in effects for key in ['missing_events', 'false_events', 'event_fragmentation']):
                    # This issue affects event abstraction, so it impacts activities
                    activity_quality_map['all_activities'].append(issue)

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Determine node color based on quality issues
            node_issues = activity_quality_map.get(node, []) + activity_quality_map.get('all_activities', [])

            if node_issues:
                # Color based on most severe issue
                severities = [issue.get('severity', 'medium') for issue in node_issues]
                if 'high' in severities:
                    color = self.color_palette['low_quality']
                elif 'medium' in severities:
                    color = self.color_palette['medium_quality']
                else:
                    color = self.color_palette['high_quality']
            else:
                color = self.color_palette['normal']

            node_color.append(color)

            # Create hover text
            hover_info = f"<b>{node}</b><br>"
            hover_info += f"Activity Frequency: {model['activity_frequencies'].get(node, 0):.2f}<br>"

            if node_issues:
                hover_info += f"<br><b>Quality Issues ({len(node_issues)}):</b><br>"
                for issue in node_issues[:3]:  # Show top 3 issues
                    hover_info += f"• {issue['type']}: {issue['description']}<br>"
                if len(node_issues) > 3:
                    hover_info += f"• ... and {len(node_issues) - 3} more<br>"
            else:
                hover_info += "<br>No quality issues detected"

            hover_text.append(hover_info)
            node_text.append(node)

        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=hover_text,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color=node_color,
                line=dict(width=2, color='#1F2937')
            ),
            textfont=dict(size=10, color='black'),
            name="Activities"
        )

    def _create_quality_timeline(self, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create timeline visualization of quality issues"""

        if not quality_issues:
            return go.Figure()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Quality Issues Over Time', 'Confidence Levels'],
            vertical_spacing=0.1
        )

        # Extract issue data
        issue_types = []
        confidences = []
        timestamps = []
        severities = []

        for i, issue in enumerate(quality_issues):
            issue_types.append(issue['type'])
            confidences.append(issue.get('confidence', 0.5))
            timestamps.append(i)  # Use index as proxy for time
            severities.append(issue.get('severity', 'medium'))

        # Create issue type timeline
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=issue_types,
                mode='markers',
                marker=dict(
                    size=10,
                    color=[self.color_palette.get(issue_type, '#888') for issue_type in issue_types],
                    line=dict(width=1, color='black')
                ),
                text=[f"Confidence: {conf:.2f}" for conf in confidences],
                hovertemplate="<b>%{y}</b><br>Issue #: %{x}<br>%{text}<extra></extra>",
                name="Quality Issues"
            ),
            row=1, col=1
        )

        # Create confidence timeline
        color_map = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        colors = [color_map.get(sev, 'gray') for sev in severities]

        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=confidences,
                marker_color=colors,
                name="Confidence",
                hovertemplate="Issue #: %{x}<br>Confidence: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )

        fig.update_layout(
            title="Quality Issues Timeline Analysis",
            height=600,
            showlegend=True
        )

        fig.update_xaxes(title_text="Issue Sequence", row=2, col=1)
        fig.update_yaxes(title_text="Issue Type", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)

        return fig

    def _create_quality_dashboard(self, process_model: Dict[str, Any],
                                  quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create comprehensive quality dashboard"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Quality Issue Distribution',
                'Severity Breakdown',
                'Process Model Metrics',
                'Information Gain Analysis'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # 1. Quality Issue Distribution (Pie Chart)
        if quality_issues:
            issue_counts = defaultdict(int)
            for issue in quality_issues:
                issue_counts[issue['type']] += 1

            fig.add_trace(
                go.Pie(
                    labels=list(issue_counts.keys()),
                    values=list(issue_counts.values()),
                    name="Issue Distribution"
                ),
                row=1, col=1
            )

        # 2. Severity Breakdown (Bar Chart)
        if quality_issues:
            severity_counts = defaultdict(int)
            for issue in quality_issues:
                severity_counts[issue.get('severity', 'medium')] += 1

            fig.add_trace(
                go.Bar(
                    x=list(severity_counts.keys()),
                    y=list(severity_counts.values()),
                    marker_color=['green', 'orange', 'red'],
                    name="Severity"
                ),
                row=1, col=2
            )

        # 3. Process Model Metrics (Bar Chart)
        if process_model and 'metrics' in process_model:
            metrics = process_model['metrics']
            metric_names = ['Fitness', 'Precision', 'Complexity']
            metric_values = [
                metrics.get('fitness', 0),
                metrics.get('precision', 0),
                metrics.get('complexity', 0)
            ]

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color=['#10B981', '#F59E0B', '#EF4444'],
                    name="Model Metrics"
                ),
                row=2, col=1
            )

        # 4. Information Gain Analysis (Scatter Plot)
        if quality_issues:
            x_vals = []
            y_vals = []
            hover_text = []

            for issue in quality_issues:
                info_gain = issue.get('information_gain', {})
                interpretability = info_gain.get('interpretability_gain', 0)
                actionability = info_gain.get('actionability_gain', 0)

                x_vals.append(interpretability)
                y_vals.append(actionability)
                hover_text.append(f"{issue['type']}<br>Confidence: {issue.get('confidence', 0):.2f}")

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(size=10, opacity=0.7),
                    text=hover_text,
                    hovertemplate="%{text}<br>Interpretability: %{x:.2f}<br>Actionability: %{y:.2f}<extra></extra>",
                    name="Info Gain"
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="IoT Data Quality Dashboard",
            height=800,
            showlegend=False
        )

        fig.update_xaxes(title_text="Interpretability Gain", row=2, col=2)
        fig.update_yaxes(title_text="Actionability Gain", row=2, col=2)

        return fig

    def _create_information_gain_viz(self, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create information gain visualization"""

        if not quality_issues:
            return go.Figure()

        # Extract information gain data
        issue_types = []
        interpretability = []
        actionability = []
        explainability = []

        for issue in quality_issues:
            info_gain = issue.get('information_gain', {})
            issue_types.append(issue['type'])
            interpretability.append(info_gain.get('interpretability_gain', 0))
            actionability.append(info_gain.get('actionability_gain', 0))
            explainability.append(info_gain.get('explainability_gain', 0))

        fig = go.Figure()

        # Add traces for each gain type
        fig.add_trace(go.Bar(
            name='Interpretability',
            x=issue_types,
            y=interpretability,
            marker_color='#3B82F6'
        ))

        fig.add_trace(go.Bar(
            name='Actionability',
            x=issue_types,
            y=actionability,
            marker_color='#10B981'
        ))

        fig.add_trace(go.Bar(
            name='Explainability',
            x=issue_types,
            y=explainability,
            marker_color='#F59E0B'
        ))

        fig.update_layout(
            title='Information Gain by Quality Issue Type',
            barmode='group',
            xaxis_title='Quality Issue Type',
            yaxis_title='Information Gain Score',
            height=500
        )

        return fig

    def _create_propagation_flow_viz(self, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """Create quality issue propagation flow visualization"""

        if not quality_issues:
            return go.Figure()

        # Create Sankey diagram showing issue propagation through pipeline stages
        stages = ['Raw Data', 'Preprocessing', 'Event Abstraction', 'Case Correlation', 'Process Mining',
                  'Visualization']

        # Build source, target, and value lists for Sankey
        sources = []
        targets = []
        values = []
        labels = []

        # Add stage labels
        stage_indices = {stage: i for i, stage in enumerate(stages)}
        labels.extend(stages)

        # Add issue type labels
        issue_types = list(set(issue['type'] for issue in quality_issues))
        issue_indices = {issue_type: len(stages) + i for i, issue_type in enumerate(issue_types)}
        labels.extend(issue_types)

        # Create flows
        for issue in quality_issues:
            issue_type = issue['type']
            issue_idx = issue_indices[issue_type]

            # Flow from issue type to affected stages
            propagated_prob = issue.get('propagated_probability', issue.get('confidence', 0.5))

            # Simulate propagation through stages (in practice, this would use real propagation data)
            for i in range(len(stages) - 1):
                sources.append(issue_idx)
                targets.append(stage_indices[stages[i]])
                values.append(propagated_prob * 10)  # Scale for visibility

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="blue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])

        fig.update_layout(
            title_text="Quality Issue Propagation Flow",
            font_size=10
        )

        return fig