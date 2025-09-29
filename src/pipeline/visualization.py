"""Enhanced visualizations working with Petri net models"""
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict


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
        """
        Create comprehensive visualization with quality annotations.

        :param process_model: Process model dictionary
        :param quality_issues: List of quality issues
        :return: Dictionary of visualizations
        """
        visualizations = {}

        visualizations['process_model'] = self._create_process_model_viz(
            process_model, quality_issues
        )

        visualizations['petri_net_quality_overlay'] = self._create_petri_net_quality_overlay(
            process_model, quality_issues
        )

        visualizations['quality_timeline'] = self._create_quality_timeline(quality_issues)

        visualizations['quality_dashboard'] = self._create_quality_dashboard(
            process_model, quality_issues
        )

        visualizations['information_gain'] = self._create_information_gain_viz(quality_issues)

        return visualizations

    def _create_process_model_viz(self, process_model: Dict[str, Any],
                                  quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """
        Create process model visualization from Petri net.

        :param process_model: Process model with Petri net
        :param quality_issues: List of quality issues
        :return: Plotly figure
        """
        if not process_model or 'model' not in process_model:
            return go.Figure()

        model = process_model['model']

        graph = self._build_graph_from_petri_net(model)

        if graph.number_of_nodes() == 0:
            return self._create_empty_figure()

        try:
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        except:
            pos = {node: (i % 5, i // 5) for i, node in enumerate(graph.nodes())}

        edge_traces = self._create_edge_traces(graph, pos)
        node_trace = self._create_node_trace(graph, pos, quality_issues, model)

        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title="Process Model with Quality Issues",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )

        return fig

    def _build_graph_from_petri_net(self, model: Dict[str, Any]) -> nx.DiGraph:
        """
        Build NetworkX graph from Petri net structure.

        :param model: Model dictionary with activities and arcs
        :return: NetworkX directed graph
        """
        graph = nx.DiGraph()

        activities = model.get('activities', [])

        for activity in activities:
            if activity:
                graph.add_node(activity, node_type='activity')

        transitions = model.get('transitions', [])
        arcs = model.get('arcs', [])

        activity_to_transitions = {}
        for trans in transitions:
            if hasattr(trans, 'label') and trans.label:
                activity_to_transitions[trans.label] = trans

        for arc in arcs:
            if hasattr(arc, 'source') and hasattr(arc, 'target'):
                source_label = arc.source.label if hasattr(arc.source, 'label') else None
                target_label = arc.target.label if hasattr(arc.target, 'label') else None

                if source_label and target_label and source_label in activities and target_label in activities:
                    graph.add_edge(source_label, target_label)

        if graph.number_of_edges() == 0 and len(activities) > 1:
            for i in range(len(activities) - 1):
                graph.add_edge(activities[i], activities[i + 1])

        return graph

    def _create_edge_traces(self, graph: nx.DiGraph, pos: Dict) -> List[go.Scatter]:
        """
        Create edge traces for graph visualization.

        :param graph: NetworkX graph
        :param pos: Node positions
        :return: List of Plotly scatter traces
        """
        edge_traces = []

        for edge in graph.edges():
            source, target = edge
            if source not in pos or target not in pos:
                continue

            x0, y0 = pos[source]
            x1, y1 = pos[target]

            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#374151'),
                showlegend=False,
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)

        return edge_traces

    def _create_node_trace(self, graph: nx.DiGraph, pos: Dict,
                          quality_issues: List[Dict[str, Any]],
                          model: Dict[str, Any]) -> go.Scatter:
        """
        Create node trace with quality annotations.

        :param graph: NetworkX graph
        :param pos: Node positions
        :param quality_issues: List of quality issues
        :param model: Model dictionary
        :return: Plotly scatter trace
        """
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        hover_text = []

        activity_quality_map = self._map_quality_to_activities(quality_issues)

        for node in graph.nodes():
            if node not in pos:
                continue

            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_issues = activity_quality_map.get(node, [])

            if node_issues:
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

            hover_info = f"<b>{node}</b><br>"

            if node_issues:
                hover_info += f"<br><b>Quality Issues ({len(node_issues)}):</b><br>"
                for issue in node_issues[:3]:
                    hover_info += f"• {issue['type']}: {issue.get('description', '')}<br>"
                if len(node_issues) > 3:
                    hover_info += f"• ... and {len(node_issues) - 3} more<br>"
            else:
                hover_info += "<br>No quality issues detected"

            hover_text.append(hover_info)
            node_text.append(node[:15])

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
            textfont=dict(size=9, color='black'),
            name="Activities"
        )

    def _map_quality_to_activities(self, quality_issues: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Map quality issues to activities.

        :param quality_issues: List of quality issues
        :return: Dictionary mapping activity names to quality issues
        """
        activity_quality_map = defaultdict(list)

        for issue in quality_issues:
            activity_quality_map['all_activities'].append(issue)

        return activity_quality_map

    def _create_quality_timeline(self, quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """
        Create timeline visualization of quality issues.

        :param quality_issues: List of quality issues
        :return: Plotly figure
        """
        if not quality_issues:
            return self._create_empty_figure()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Quality Issues Over Time', 'Confidence Levels'],
            vertical_spacing=0.1
        )

        issue_types = []
        confidences = []
        timestamps = []
        severities = []

        for i, issue in enumerate(quality_issues):
            issue_types.append(issue['type'])
            confidences.append(issue.get('confidence', 0.5))
            timestamps.append(i)
            severities.append(issue.get('severity', 'medium'))

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
        """
        Create comprehensive quality dashboard.

        :param process_model: Process model dictionary
        :param quality_issues: List of quality issues
        :return: Plotly figure
        """
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

        if process_model and 'metrics' in process_model:
            metrics = process_model['metrics']
            metric_names = ['Fitness', 'Precision']
            metric_values = [
                metrics.get('fitness', 0),
                metrics.get('precision', 0)
            ]

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    marker_color=['#10B981', '#F59E0B'],
                    name="Model Metrics"
                ),
                row=2, col=1
            )

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
        """
        Create information gain visualization.

        :param quality_issues: List of quality issues
        :return: Plotly figure
        """
        if not quality_issues:
            return self._create_empty_figure()

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

    def _create_empty_figure(self) -> go.Figure:
        """
        Create empty figure with message.

        :return: Empty Plotly figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    def _create_petri_net_quality_overlay(self, process_model: Dict[str, Any],
                                          quality_issues: List[Dict[str, Any]]) -> go.Figure:
        """
        Create Petri net visualization with quality overlays for process analysts.

        :param process_model: Process model with Petri net
        :param quality_issues: List of quality issues
        :return: Interactive Plotly figure with quality annotations
        """
        if not process_model or 'model' not in process_model:
            return self._create_empty_figure()

        model = process_model['model']
        net = model.get('petri_net')

        if not net:
            return self._create_empty_figure()

        activity_quality_map = self._analyze_quality_by_activity(quality_issues, model)
        sensor_activity_map = self._extract_sensor_activity_mapping(quality_issues)

        layout_info = self._calculate_petri_net_layout(net)

        fig = go.Figure()

        self._add_places_to_figure(fig, layout_info['places'])
        self._add_arcs_to_figure(fig, layout_info['arcs'])
        self._add_transitions_to_figure(fig, layout_info['transitions'], activity_quality_map)
        self._add_sensor_overlays_to_figure(fig, layout_info['transitions'],
                                           sensor_activity_map, activity_quality_map)

        self._add_quality_legend(fig, quality_issues)

        fig.update_layout(
            title={
                'text': "Process Model with Data Quality Insights<br><sub>Colors indicate quality issues - Hover for details</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[layout_info['x_range'][0] - 1, layout_info['x_range'][1] + 1]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[layout_info['y_range'][0] - 2, layout_info['y_range'][1] + 2]
            ),
            height=800,
            width=1200,
            plot_bgcolor='#F9FAFB'
        )

        return fig

    def _analyze_quality_by_activity(self, quality_issues: List[Dict[str, Any]],
                                     model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze quality issues and map them to activities.

        :param quality_issues: List of quality issues
        :param model: Model dictionary
        :return: Dictionary mapping activities to quality analysis
        """
        activity_quality = {}

        conformance_issues = [qi for qi in quality_issues if 'conformance' in qi.get('type', '')]
        data_issues = [qi for qi in quality_issues if 'conformance' not in qi.get('type', '')]

        for activity in model.get('activities', []):
            activity_quality[activity] = {
                'issues': [],
                'sensors': set(),
                'severity': 'none',
                'confidence': 0.0,
                'impact_description': []
            }

        for issue in data_issues:
            sensor_id = issue.get('sensor_id', '')
            issue_type = issue.get('type', '')

            affected_activities = self._infer_affected_activities(issue, model.get('activities', []))

            for activity in affected_activities:
                if activity in activity_quality:
                    activity_quality[activity]['issues'].append(issue)
                    if sensor_id:
                        activity_quality[activity]['sensors'].add(sensor_id)

                    current_severity = activity_quality[activity]['severity']
                    issue_severity = issue.get('severity', 'low')

                    severity_rank = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
                    if severity_rank.get(issue_severity, 0) > severity_rank.get(current_severity, 0):
                        activity_quality[activity]['severity'] = issue_severity

                    activity_quality[activity]['confidence'] = max(
                        activity_quality[activity]['confidence'],
                        issue.get('confidence', 0)
                    )

                    impact = self._describe_issue_impact(issue, activity)
                    if impact:
                        activity_quality[activity]['impact_description'].append(impact)

        return activity_quality

    def _infer_affected_activities(self, issue: Dict[str, Any], activities: List[str]) -> List[str]:
        """
        Infer which activities are affected by a quality issue.

        :param issue: Quality issue
        :param activities: List of all activities
        :return: List of affected activity names
        """
        affected = []
        sensor_id = issue.get('sensor_id', '')

        if not sensor_id:
            return activities[:3] if len(activities) > 0 else []

        if 'PWR' in sensor_id:
            affected = [a for a in activities if any(x in a for x in ['Weld', 'Package', 'Seal'])]
        elif 'TEMP' in sensor_id:
            affected = [a for a in activities if any(x in a for x in ['Weld', 'Cool'])]
        elif 'VIB' in sensor_id:
            affected = [a for a in activities if any(x in a for x in ['Scan', 'Measure', 'Inspect'])]
        elif 'POS' in sensor_id:
            affected = [a for a in activities if 'Position' in a or 'Pick' in a]

        return affected if affected else activities[:1]

    def _describe_issue_impact(self, issue: Dict[str, Any], activity: str) -> str:
        """
        Describe how an issue impacts a specific activity.

        :param issue: Quality issue
        :param activity: Activity name
        :return: Impact description
        """
        issue_type = issue.get('type', '')
        confidence = issue.get('confidence', 0)

        descriptions = {
            'C1_inadequate_sampling': f"Low sampling rate may miss {activity} events",
            'C2_poor_placement': f"Sensor overlap causes ambiguous {activity} detection",
            'C3_sensor_noise': f"Noise creates false {activity} events",
            'C4_range_too_small': f"Sensor range insufficient for {activity} detection",
            'C5_high_volume': f"Data drops may lose {activity} events"
        }

        desc = descriptions.get(issue_type, f"Quality issue affects {activity}")
        return f"{desc} (conf: {confidence:.2f})"

    def _extract_sensor_activity_mapping(self, quality_issues: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract which sensors are related to which activities.

        :param quality_issues: List of quality issues
        :return: Dictionary mapping sensor IDs to activity lists
        """
        sensor_map = defaultdict(set)

        for issue in quality_issues:
            sensor_id = issue.get('sensor_id', '')
            if not sensor_id:
                continue

            if 'PWR' in sensor_id:
                sensor_map[sensor_id].update(['Weld', 'Package', 'Seal'])
            elif 'TEMP' in sensor_id:
                sensor_map[sensor_id].update(['Weld', 'Cool'])
            elif 'VIB' in sensor_id:
                sensor_map[sensor_id].update(['Scan', 'Measure'])
            elif 'POS' in sensor_id:
                sensor_map[sensor_id].update(['Position', 'Pick'])

        return {k: list(v) for k, v in sensor_map.items()}

    def _calculate_petri_net_layout(self, net) -> Dict[str, Any]:
        """
        Calculate layout for Petri net visualization.

        :param net: Petri net object
        :return: Layout information dictionary
        """
        transitions = [t for t in net.transitions]
        places = [p for p in net.places]

        transition_positions = {}
        place_positions = {}

        labeled_transitions = [t for t in transitions if t.label]
        unlabeled_transitions = [t for t in transitions if not t.label]

        x_spacing = 3
        y_spacing = 2

        for i, trans in enumerate(labeled_transitions):
            x = i * x_spacing
            y = 0
            transition_positions[trans] = {'x': x, 'y': y, 'label': trans.label}

        for i, place in enumerate(places):
            if str(place.name) == 'source':
                place_positions[place] = {'x': -2, 'y': 0, 'label': 'START'}
            elif str(place.name) == 'sink':
                place_positions[place] = {'x': len(labeled_transitions) * x_spacing + 1, 'y': 0, 'label': 'END'}
            else:
                x = (i % 8) * x_spacing - 1.5
                y = (i // 8) * y_spacing + 1
                place_positions[place] = {'x': x, 'y': y, 'label': ''}

        arc_data = []
        for arc in net.arcs:
            source_pos = transition_positions.get(arc.source) or place_positions.get(arc.source)
            target_pos = transition_positions.get(arc.target) or place_positions.get(arc.target)

            if source_pos and target_pos:
                arc_data.append({
                    'x0': source_pos['x'],
                    'y0': source_pos['y'],
                    'x1': target_pos['x'],
                    'y1': target_pos['y']
                })

        all_x = [p['x'] for p in list(transition_positions.values()) + list(place_positions.values())]
        all_y = [p['y'] for p in list(transition_positions.values()) + list(place_positions.values())]

        return {
            'transitions': transition_positions,
            'places': place_positions,
            'arcs': arc_data,
            'x_range': [min(all_x) if all_x else 0, max(all_x) if all_x else 1],
            'y_range': [min(all_y) if all_y else 0, max(all_y) if all_y else 1]
        }

    def _add_places_to_figure(self, fig: go.Figure, places: Dict) -> None:
        """
        Add places (circles) to Petri net visualization.

        :param fig: Plotly figure
        :param places: Place position information
        """
        place_x = []
        place_y = []
        place_text = []
        place_hover = []

        for place, pos in places.items():
            place_x.append(pos['x'])
            place_y.append(pos['y'])
            place_text.append(pos['label'])
            place_hover.append(f"Place: {place.name}<br>Type: Control Flow")

        fig.add_trace(go.Scatter(
            x=place_x,
            y=place_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color='#E5E7EB',
                symbol='circle',
                line=dict(color='#6B7280', width=2)
            ),
            text=place_text,
            textposition='top center',
            textfont=dict(size=10, color='#374151'),
            hovertext=place_hover,
            hoverinfo='text',
            name='Places',
            showlegend=False
        ))

    def _add_arcs_to_figure(self, fig: go.Figure, arcs: List[Dict]) -> None:
        """
        Add arcs (arrows) to Petri net visualization.

        :param fig: Plotly figure
        :param arcs: Arc information
        """
        for arc in arcs:
            fig.add_annotation(
                x=arc['x1'],
                y=arc['y1'],
                ax=arc['x0'],
                ay=arc['y0'],
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='#9CA3AF',
                opacity=0.6
            )

    def _add_transitions_to_figure(self, fig: go.Figure, transitions: Dict,
                                   activity_quality: Dict[str, Dict]) -> None:
        """
        Add transitions (rectangles) with quality color coding.

        :param fig: Plotly figure
        :param transitions: Transition position information
        :param activity_quality: Quality analysis per activity
        """
        for trans, pos in transitions.items():
            if not pos['label']:
                continue

            activity = pos['label']
            quality_info = activity_quality.get(activity, {})

            severity = quality_info.get('severity', 'none')
            severity_colors = {
                'high': '#EF4444',
                'medium': '#F59E0B',
                'low': '#10B981',
                'none': '#3B82F6'
            }
            color = severity_colors.get(severity, '#3B82F6')

            issues = quality_info.get('issues', [])
            sensors = quality_info.get('sensors', set())
            impact_desc = quality_info.get('impact_description', [])

            hover_text = f"<b>{activity}</b><br>"
            hover_text += f"<b>Quality Status:</b> {severity.upper()}<br>"

            if issues:
                hover_text += f"<br><b>Issues Detected ({len(issues)}):</b><br>"
                for issue in issues[:3]:
                    hover_text += f"• {issue['type']}: {issue.get('description', 'N/A')[:50]}<br>"

            if sensors:
                hover_text += f"<br><b>Related Sensors:</b> {', '.join(sensors)}<br>"

            if impact_desc:
                hover_text += f"<br><b>Impact:</b><br>"
                for imp in impact_desc[:2]:
                    hover_text += f"• {imp}<br>"

            if not issues:
                hover_text += "<br>✓ No quality issues detected"

            fig.add_trace(go.Scatter(
                x=[pos['x']],
                y=[pos['y']],
                mode='markers+text',
                marker=dict(
                    size=40,
                    color=color,
                    symbol='square',
                    line=dict(
                        color='#1F2937',
                        width=3 if severity == 'high' else 2
                    )
                ),
                text=activity[:12],
                textposition='middle center',
                textfont=dict(size=9, color='white', family='Arial Black'),
                hovertext=hover_text,
                hoverinfo='text',
                name=activity,
                showlegend=False
            ))

    def _add_sensor_overlays_to_figure(self, fig: go.Figure, transitions: Dict,
                                       sensor_map: Dict[str, List[str]],
                                       activity_quality: Dict[str, Dict]) -> None:
        """
        Add sensor nodes and connections to show data flow.

        :param fig: Plotly figure
        :param transitions: Transition positions
        :param sensor_map: Sensor to activity mapping
        :param activity_quality: Quality information per activity
        """
        sensor_positions = {}
        sensor_y_offset = -3

        sensor_x_positions = {}
        current_x = 0
        for sensor_id in sensor_map.keys():
            sensor_x_positions[sensor_id] = current_x
            current_x += 2.5

        for sensor_id, activities in sensor_map.items():
            x_pos = sensor_x_positions[sensor_id]
            sensor_positions[sensor_id] = {'x': x_pos, 'y': sensor_y_offset}

            sensor_issues = [
                issue for activity in activities
                for issue in activity_quality.get(activity, {}).get('issues', [])
                if issue.get('sensor_id') == sensor_id
            ]

            sensor_color = '#EF4444' if any(i.get('severity') == 'high' for i in sensor_issues) else '#F59E0B' if sensor_issues else '#10B981'

            hover_text = f"<b>Sensor: {sensor_id}</b><br>"
            hover_text += f"<b>Monitors:</b> {', '.join(activities)}<br>"

            if sensor_issues:
                hover_text += f"<br><b>Issues ({len(sensor_issues)}):</b><br>"
                for issue in sensor_issues[:2]:
                    hover_text += f"• {issue['type']}<br>"
            else:
                hover_text += "<br>✓ Operating normally"

            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[sensor_y_offset],
                mode='markers+text',
                marker=dict(
                    size=30,
                    color=sensor_color,
                    symbol='diamond',
                    line=dict(color='#1F2937', width=2)
                ),
                text=sensor_id,
                textposition='bottom center',
                textfont=dict(size=8, color='#374151'),
                hovertext=hover_text,
                hoverinfo='text',
                name=sensor_id,
                showlegend=False
            ))

            for activity in activities:
                trans_pos = None
                for trans, pos in transitions.items():
                    if pos['label'] == activity:
                        trans_pos = pos
                        break

                if trans_pos:
                    fig.add_annotation(
                        x=trans_pos['x'],
                        y=trans_pos['y'],
                        ax=x_pos,
                        ay=sensor_y_offset,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.8,
                        arrowwidth=1,
                        arrowcolor=sensor_color,
                        opacity=0.4,
                        standoff=10
                    )

    def _add_quality_legend(self, fig: go.Figure, quality_issues: List[Dict[str, Any]]) -> None:
        """
        Add legend explaining quality indicators.

        :param fig: Plotly figure
        :param quality_issues: List of quality issues
        """
        legend_items = [
            {'color': '#EF4444', 'label': 'High Severity', 'symbol': 'square'},
            {'color': '#F59E0B', 'label': 'Medium Severity', 'symbol': 'square'},
            {'color': '#10B981', 'label': 'Low/No Issues', 'symbol': 'square'},
            {'color': '#3B82F6', 'label': 'Normal Activity', 'symbol': 'square'},
            {'color': '#9CA3AF', 'label': 'Sensor', 'symbol': 'diamond'}
        ]

        for i, item in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=item['color'], symbol=item['symbol']),
                name=item['label'],
                showlegend=True
            ))