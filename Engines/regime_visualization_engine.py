#!/usr/bin/env python3
"""
Regime Visualization Engine - Generates comprehensive reports and visualizations
for regime analysis results.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Add the current directory to Python path for imports
sys.path.append('.')

warnings.filterwarnings('ignore')

@dataclass
class RegimeVisualizationConfig:
    """Configuration for regime visualization engine"""
    # Paths
    data_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Data"
    results_path: str = r"C:\Users\andre\OneDrive\Desktop\Mastercode\Backtesting\Engines\Results"
    
    # Visualization settings
    save_png: bool = True
    save_html: bool = True
    save_pdf: bool = False
    create_interactive_charts: bool = True
    create_summary_dashboard: bool = True
    
    # Chart settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "default"
    
    # Color schemes
    regime_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

class RegimeVisualizationEngine:
    """
    Regime visualization engine that generates comprehensive reports and visualizations
    for regime analysis results.
    """
    
    def __init__(self, config: RegimeVisualizationConfig):
        self.config = config
        self.setup_logging()
        self.setup_plotting()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.results_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"regime_visualization_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def setup_plotting(self):
        """Setup plotting configuration."""
        try:
            plt.style.use(self.config.style)
        except:
            # Fallback to default style if custom style not available
            plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_regime_report(self, regime_result, 
                             data_file: str, output_path: str = None) -> str:
        """Generate comprehensive regime analysis report"""
        try:
            if output_path is None:
                output_path = Path(self.config.results_path) / "RegimeVisualizationEngine" / data_file.replace('.csv', '') / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Generating regime report for {data_file}...")
            
            # Convert RegimeDetectionResult to dictionary format
            regime_dict = {
                'regime_mapping': regime_result.regime_mapping,
                'regime_statistics': regime_result.regime_statistics,
                'regime_transitions': regime_result.regime_transitions,
                'feature_importance': regime_result.feature_importance,
                'baseline_conditions': regime_result.baseline_conditions
            }
            
            # Generate individual visualizations
            self._create_regime_timeline_chart(regime_dict, output_path)
            self._create_regime_distribution_chart(regime_dict, output_path)
            self._create_regime_characteristics_heatmap(regime_dict, output_path)
            self._create_regime_transitions_chart(regime_dict, output_path)
            self._create_feature_importance_chart(regime_dict, output_path)
            self._create_baseline_conditions_chart(regime_dict, output_path)
            
            # Generate summary dashboard
            if self.config.create_summary_dashboard:
                self._create_summary_dashboard(regime_dict, data_file, output_path)
            
            # Generate text report
            self._generate_text_report(regime_dict, data_file, output_path)
            
            self.logger.info(f"Regime report generated for {data_file} at {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating regime report for {data_file}: {e}")
            raise
    
    def _create_regime_timeline_chart(self, regime_result: Dict[str, Any], output_path: Path):
        """Create regime timeline chart"""
        try:
            regime_mapping = regime_result.get('regime_mapping', {})
            if not regime_mapping:
                return
            
            # Prepare data for timeline
            timeline_data = []
            for timestamp, regime in regime_mapping.items():
                timeline_data.append({
                    'timestamp': timestamp,
                    'regime_id': regime['regime_id'],
                    'regime_type': regime['regime_type'],
                    'regime_name': regime['regime_name']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('timestamp')
            
            # Create timeline chart
            fig = go.Figure()
            
            # Add regime segments
            for regime_type in timeline_df['regime_type'].unique():
                regime_data = timeline_df[timeline_df['regime_type'] == regime_type]
                color = self.config.regime_colors[hash(regime_type) % len(self.config.regime_colors)]
                
                fig.add_trace(go.Scatter(
                    x=regime_data['timestamp'],
                    y=[regime_type] * len(regime_data),
                    mode='markers',
                    marker=dict(size=8, color=color),
                    name=regime_type,
                    hovertemplate=f'<b>{regime_type}</b><br>' +
                                 'Date: %{x}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_layout(
                title='Market Regime Timeline',
                xaxis_title='Date',
                yaxis_title='Regime Type',
                height=400,
                showlegend=True
            )
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "regime_timeline.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "regime_timeline.png"
                fig.write_image(png_file, width=1200, height=400, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating regime timeline chart: {e}")
    
    def _create_regime_distribution_chart(self, regime_result: Dict[str, Any], output_path: Path):
        """Create regime distribution chart"""
        try:
            regime_stats = regime_result.get('regime_statistics', {})
            if not regime_stats:
                return
            
            # Extract regime information
            regime_names = []
            regime_percentages = []
            regime_counts = []
            
            for regime_key, stats in regime_stats.items():
                regime_names.append(regime_key)
                regime_percentages.append(stats['percentage_of_data'])
                regime_counts.append(stats['n_observations'])
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=regime_names, 
                values=regime_percentages,
                hovertemplate='<b>%{label}</b><br>' +
                             'Percentage: %{percent}<br>' +
                             'Count: %{text}<br>' +
                             '<extra></extra>',
                text=regime_counts
            )])
            
            fig.update_layout(
                title='Market Regime Distribution',
                height=500
            )
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "regime_distribution.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "regime_distribution.png"
                fig.write_image(png_file, width=800, height=500, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating regime distribution chart: {e}")
    
    def _create_regime_characteristics_heatmap(self, regime_result: Dict[str, Any], output_path: Path):
        """Create regime characteristics heatmap"""
        try:
            regime_stats = regime_result.get('regime_statistics', {})
            if not regime_stats:
                return
            
            # Prepare data for heatmap
            feature_names = []
            regime_names = []
            char_matrix = []
            
            # Get all unique features
            all_features = set()
            for regime_key, stats in regime_stats.items():
                if 'characteristics' in stats:
                    all_features.update(stats['characteristics'].keys())
            
            feature_names = sorted(list(all_features))
            regime_names = list(regime_stats.keys())
            
            # Create characteristic matrix
            for regime_name in regime_names:
                regime_row = []
                for feature_name in feature_names:
                    if feature_name in regime_stats[regime_name].get('characteristics', {}):
                        char_matrix.append([
                            regime_name,
                            feature_name,
                            regime_stats[regime_name]['characteristics'][feature_name]['mean']
                        ])
            
            if char_matrix:
                char_df = pd.DataFrame(char_matrix, columns=['Regime', 'Feature', 'Value'])
                char_pivot = char_df.pivot(index='Regime', columns='Feature', values='Value')
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=char_pivot.values,
                    x=char_pivot.columns,
                    y=char_pivot.index,
                    colorscale='RdYlBu',
                    hovertemplate='<b>%{y}</b><br>' +
                                 'Feature: %{x}<br>' +
                                 'Value: %{z:.4f}<br>' +
                                 '<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Regime Characteristics Heatmap',
                    xaxis_title='Features',
                    yaxis_title='Regimes',
                    height=600
                )
                
                # Save chart
                if self.config.save_html:
                    html_file = output_path / "regime_characteristics_heatmap.html"
                    fig.write_html(html_file)
                
                if self.config.save_png:
                    png_file = output_path / "regime_characteristics_heatmap.png"
                    fig.write_image(png_file, width=1200, height=600, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating regime characteristics heatmap: {e}")
    
    def _create_regime_transitions_chart(self, regime_result: Dict[str, Any], output_path: Path):
        """Create regime transitions chart"""
        try:
            transitions = regime_result.get('regime_transitions', [])
            if not transitions:
                return
            
            # Create transition matrix
            unique_regimes = set()
            for transition in transitions:
                unique_regimes.add(transition['from_regime'])
                unique_regimes.add(transition['to_regime'])
            
            transition_matrix = np.zeros((len(unique_regimes), len(unique_regimes)))
            regime_id_map = {regime_id: i for i, regime_id in enumerate(sorted(unique_regimes))}
            
            for transition in transitions:
                from_idx = regime_id_map[transition['from_regime']]
                to_idx = regime_id_map[transition['to_regime']]
                transition_matrix[from_idx, to_idx] += 1
            
            # Create heatmap
            regime_labels = [f"Regime_{rid}" for rid in sorted(unique_regimes)]
            
            fig = go.Figure(data=go.Heatmap(
                z=transition_matrix,
                x=regime_labels,
                y=regime_labels,
                colorscale='Blues',
                hovertemplate='<b>From: %{y}</b><br>' +
                             'To: %{x}<br>' +
                             'Count: %{z}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title='Regime Transition Matrix',
                xaxis_title='To Regime',
                yaxis_title='From Regime',
                height=500
            )
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "regime_transitions.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "regime_transitions.png"
                fig.write_image(png_file, width=800, height=500, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating regime transitions chart: {e}")
    
    def _create_feature_importance_chart(self, regime_result: Dict[str, Any], output_path: Path):
        """Create feature importance chart"""
        try:
            feature_importance = regime_result.get('feature_importance', {})
            if not feature_importance:
                return
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            # Create bar chart
            fig = go.Figure(data=go.Bar(
                x=feature_names, 
                y=importance_values,
                hovertemplate='<b>%{x}</b><br>' +
                             'Importance: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title='Feature Importance for Regime Detection',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                xaxis_tickangle=-45,
                height=500
            )
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "feature_importance.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "feature_importance.png"
                fig.write_image(png_file, width=1200, height=500, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance chart: {e}")
    
    def _create_baseline_conditions_chart(self, regime_result: Dict[str, Any], output_path: Path):
        """Create baseline conditions chart"""
        try:
            baseline_conditions = regime_result.get('baseline_conditions', {})
            if not baseline_conditions:
                return
            
            # Create baseline conditions visualization
            fig = go.Figure()
            
            for regime_type, conditions in baseline_conditions.items():
                if 'avg_characteristics' in conditions:
                    characteristics = conditions['avg_characteristics']
                    
                    # Create radar chart for each regime
                    categories = list(characteristics.keys())
                    values = [characteristics[cat]['mean'] for cat in categories]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=regime_type,
                        hovertemplate='<b>%{theta}</b><br>' +
                                     'Value: %{r:.4f}<br>' +
                                     '<extra></extra>'
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max([max([conditions['avg_characteristics'][cat]['mean'] 
                                          for cat in conditions['avg_characteristics']]) 
                                    for conditions in baseline_conditions.values() 
                                    if 'avg_characteristics' in conditions])]
                    )
                ),
                title='Baseline Conditions by Regime Type',
                height=600
            )
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "baseline_conditions.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "baseline_conditions.png"
                fig.write_image(png_file, width=800, height=600, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating baseline conditions chart: {e}")
    
    def _create_summary_dashboard(self, regime_result: Dict[str, Any], data_file: str, output_path: Path):
        """Create summary dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Regime Distribution', 'Feature Importance', 
                              'Regime Timeline', 'Baseline Conditions'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatterpolar"}]]
            )
            
            # Add regime distribution
            regime_stats = regime_result.get('regime_statistics', {})
            if regime_stats:
                regime_names = list(regime_stats.keys())
                regime_percentages = [stats['percentage_of_data'] for stats in regime_stats.values()]
                
                fig.add_trace(
                    go.Pie(labels=regime_names, values=regime_percentages, name="Distribution"),
                    row=1, col=1
                )
            
            # Add feature importance
            feature_importance = regime_result.get('feature_importance', {})
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                feature_names = [f[0] for f in sorted_features]
                importance_values = [f[1] for f in sorted_features]
                
                fig.add_trace(
                    go.Bar(x=feature_names, y=importance_values, name="Importance"),
                    row=1, col=2
                )
            
            # Add regime timeline
            regime_mapping = regime_result.get('regime_mapping', {})
            if regime_mapping:
                timeline_data = []
                for timestamp, regime in regime_mapping.items():
                    timeline_data.append({
                        'timestamp': timestamp,
                        'regime_type': regime['regime_type']
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df = timeline_df.sort_values('timestamp')
                
                for regime_type in timeline_df['regime_type'].unique():
                    regime_data = timeline_df[timeline_df['regime_type'] == regime_type]
                    fig.add_trace(
                        go.Scatter(
                            x=regime_data['timestamp'],
                            y=[regime_type] * len(regime_data),
                            mode='markers',
                            name=regime_type,
                            showlegend=False
                        ),
                        row=2, col=1
                    )
            
            fig.update_layout(
                title=f'Regime Analysis Summary Dashboard - {data_file}',
                height=800,
                showlegend=True
            )
            
            # Save dashboard
            if self.config.save_html:
                html_file = output_path / "summary_dashboard.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "summary_dashboard.png"
                fig.write_image(png_file, width=1600, height=800, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating summary dashboard: {e}")
    
    def _generate_text_report(self, regime_result: Dict[str, Any], data_file: str, output_path: Path):
        """Generate text report"""
        try:
            report_file = output_path / "regime_analysis_report.txt"
            
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MARKET REGIME ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Data File: {data_file}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Regime statistics
                regime_stats = regime_result.get('regime_statistics', {})
                f.write("REGIME STATISTICS\n")
                f.write("-" * 40 + "\n")
                for regime_key, stats in regime_stats.items():
                    f.write(f"\n{regime_key}:\n")
                    f.write(f"  Observations: {stats['n_observations']}\n")
                    f.write(f"  Percentage: {stats['percentage_of_data']:.2f}%\n")
                    f.write(f"  Avg Duration: {stats['avg_duration_days']:.1f} days\n")
                
                # Feature importance
                feature_importance = regime_result.get('feature_importance', {})
                f.write("\n\nFEATURE IMPORTANCE\n")
                f.write("-" * 40 + "\n")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features:
                    f.write(f"{feature}: {importance:.4f}\n")
                
                # Regime transitions
                transitions = regime_result.get('regime_transitions', [])
                f.write("\n\nREGIME TRANSITIONS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Transitions: {len(transitions)}\n")
                for transition in transitions:
                    f.write(f"  {transition['from_regime_type']} -> {transition['to_regime_type']} "
                           f"({transition['transition_date']})\n")
                
                # Baseline conditions
                baseline_conditions = regime_result.get('baseline_conditions', {})
                f.write("\n\nBASELINE CONDITIONS\n")
                f.write("-" * 40 + "\n")
                for regime_type, conditions in baseline_conditions.items():
                    f.write(f"\n{regime_type}:\n")
                    if 'avg_characteristics' in conditions:
                        for feature, stats in conditions['avg_characteristics'].items():
                            f.write(f"  {feature}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"Text report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating text report: {e}")
    
    def create_regime_overlay_visualization(self, overlay_result: Dict[str, Any], 
                                          output_path: str = None) -> str:
        """Create visualization for regime overlay results"""
        try:
            if output_path is None:
                output_path = Path(self.config.results_path) / "RegimeOverlayVisualization" / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Creating regime overlay visualizations...")
            
            # Create overlay performance chart
            self._create_overlay_performance_chart(overlay_result, output_path)
            
            # Create overlay consistency chart
            self._create_overlay_consistency_chart(overlay_result, output_path)
            
            # Create overlay summary dashboard
            self._create_overlay_dashboard(overlay_result, output_path)
            
            self.logger.info(f"Regime overlay visualizations created at {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating regime overlay visualizations: {e}")
            raise
    
    def _create_overlay_performance_chart(self, overlay_result: Dict[str, Any], output_path: Path):
        """Create overlay performance chart"""
        try:
            regime_performance = overlay_result.get('regime_performance', {})
            if not regime_performance:
                return
            
            # Create performance comparison chart
            regimes = list(regime_performance.keys())
            avg_returns = [regime_performance[r]['avg_return'] for r in regimes]
            avg_sharpe = [regime_performance[r]['avg_sharpe'] for r in regimes]
            avg_drawdown = [regime_performance[r]['avg_drawdown'] for r in regimes]
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Average Returns by Regime', 'Average Sharpe Ratio by Regime', 
                              'Average Max Drawdown by Regime'),
                vertical_spacing=0.1
            )
            
            # Returns chart
            fig.add_trace(
                go.Bar(x=regimes, y=avg_returns, name='Returns', 
                      hovertemplate='<b>%{x}</b><br>Return: %{y:.4f}<br><extra></extra>'),
                row=1, col=1
            )
            
            # Sharpe ratio chart
            fig.add_trace(
                go.Bar(x=regimes, y=avg_sharpe, name='Sharpe Ratio',
                      hovertemplate='<b>%{x}</b><br>Sharpe: %{y:.4f}<br><extra></extra>'),
                row=2, col=1
            )
            
            # Drawdown chart
            fig.add_trace(
                go.Bar(x=regimes, y=avg_drawdown, name='Max Drawdown',
                      hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.4f}<br><extra></extra>'),
                row=3, col=1
            )
            
            fig.update_layout(height=900, title='Regime Performance Analysis')
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "overlay_performance.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "overlay_performance.png"
                fig.write_image(png_file, width=1200, height=900, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating overlay performance chart: {e}")
    
    def _create_overlay_consistency_chart(self, overlay_result: Dict[str, Any], output_path: Path):
        """Create overlay consistency chart"""
        try:
            regime_performance = overlay_result.get('regime_performance', {})
            if not regime_performance:
                return
            
            # Create consistency chart
            regimes = list(regime_performance.keys())
            consistency_scores = [regime_performance[r]['return_consistency'] for r in regimes]
            n_results = [regime_performance[r]['n_results'] for r in regimes]
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Regime Performance Consistency', 'Number of Results by Regime'),
                vertical_spacing=0.1
            )
            
            # Consistency chart
            fig.add_trace(
                go.Bar(x=regimes, y=consistency_scores, name='Consistency',
                      hovertemplate='<b>%{x}</b><br>Consistency: %{y:.4f}<br><extra></extra>'),
                row=1, col=1
            )
            
            # Results count chart
            fig.add_trace(
                go.Bar(x=regimes, y=n_results, name='Results Count',
                      hovertemplate='<b>%{x}</b><br>Count: %{y}<br><extra></extra>'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title='Regime Performance Consistency Analysis')
            
            # Save chart
            if self.config.save_html:
                html_file = output_path / "overlay_consistency.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "overlay_consistency.png"
                fig.write_image(png_file, width=1200, height=600, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating overlay consistency chart: {e}")
    
    def _create_overlay_dashboard(self, overlay_result: Dict[str, Any], output_path: Path):
        """Create overlay summary dashboard"""
        try:
            # Create comprehensive dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Performance by Regime', 'Consistency by Regime', 
                              'Results Distribution', 'Regime Conditions'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "scatter"}]]
            )
            
            regime_performance = overlay_result.get('regime_performance', {})
            if regime_performance:
                regimes = list(regime_performance.keys())
                
                # Performance chart
                avg_returns = [regime_performance[r]['avg_return'] for r in regimes]
                fig.add_trace(
                    go.Bar(x=regimes, y=avg_returns, name="Returns"),
                    row=1, col=1
                )
                
                # Consistency chart
                consistency_scores = [regime_performance[r]['return_consistency'] for r in regimes]
                fig.add_trace(
                    go.Bar(x=regimes, y=consistency_scores, name="Consistency"),
                    row=1, col=2
                )
                
                # Results distribution
                n_results = [regime_performance[r]['n_results'] for r in regimes]
                fig.add_trace(
                    go.Pie(labels=regimes, values=n_results, name="Distribution"),
                    row=2, col=1
                )
            
            fig.update_layout(
                title='Regime Overlay Analysis Dashboard',
                height=800,
                showlegend=True
            )
            
            # Save dashboard
            if self.config.save_html:
                html_file = output_path / "overlay_dashboard.html"
                fig.write_html(html_file)
            
            if self.config.save_png:
                png_file = output_path / "overlay_dashboard.png"
                fig.write_image(png_file, width=1600, height=800, scale=2)
            
        except Exception as e:
            self.logger.error(f"Error creating overlay dashboard: {e}")
