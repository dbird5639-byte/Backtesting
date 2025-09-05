#!/usr/bin/env python3
"""
Enhanced Visualization Engine

This engine provides comprehensive visualization capabilities for backtesting results,
including heatmaps, charts, and regime overlays. It organizes results by strategy
and data file for easy analysis.

Features:
- Comprehensive chart generation (candlestick, line, bar charts)
- Heatmap generation for performance metrics
- Regime overlay visualization
- Strategy comparison charts
- Risk analysis visualizations
- Performance attribution charts
- Interactive plotly charts
- Organized output by strategy and data file
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import warnings
warnings.filterwarnings('ignore')

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from plotly.offline import plot
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization libraries not available. Install matplotlib, seaborn, plotly for full functionality.")

# Import base engine
from .core_engine import CoreEngine, EngineConfig

@dataclass
class VisualizationConfig(EngineConfig):
    """Configuration for enhanced visualization engine"""
    # Visualization settings
    chart_types: List[str] = field(default_factory=lambda: [
        'candlestick', 'line', 'bar', 'heatmap', 'scatter', 'histogram'
    ])
    
    # Chart dimensions
    chart_width: int = 1200
    chart_height: int = 800
    
    # Color schemes
    color_scheme: str = 'plotly'  # plotly, seaborn, custom
    
    # Regime overlay settings
    enable_regime_overlay: bool = True
    regime_alpha: float = 0.3
    regime_colors: Dict[str, str] = field(default_factory=lambda: {
        'Bull': '#00ff00',
        'Bear': '#ff0000',
        'Sideways': '#ffff00',
        'High_Volatility': '#ff8000',
        'Low_Volatility': '#0080ff'
    })
    
    # Heatmap settings
    heatmap_metrics: List[str] = field(default_factory=lambda: [
        'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'
    ])
    
    # Output organization
    organize_by_strategy: bool = True
    organize_by_data_file: bool = True
    create_summary_dashboard: bool = True
    
    # File formats
    save_png: bool = True
    save_html: bool = True
    save_pdf: bool = False

class EnhancedVisualizationEngine(CoreEngine):
    """Enhanced visualization engine with comprehensive charting capabilities"""
    
    def __init__(self, config: VisualizationConfig):
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization libraries not available. Please install matplotlib, seaborn, plotly.")
        
        # Setup visualization style
        self.setup_visualization_style()
    
    def setup_visualization_style(self):
        """Setup visualization style and color schemes"""
        if self.config.color_scheme == 'seaborn':
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        elif self.config.color_scheme == 'plotly':
            # Use plotly default colors
            pass
        else:
            # Custom color scheme
            plt.style.use('default')
    
    def create_candlestick_chart(self, data: pd.DataFrame, title: str, 
                                regime_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create candlestick chart with regime overlay"""
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ))
        
        # Add regime overlay if available
        if regime_data is not None and self.config.enable_regime_overlay:
            self.add_regime_overlay(fig, regime_data)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            width=self.config.chart_width,
            height=self.config.chart_height,
            showlegend=True
        )
        
        return fig
    
    def create_performance_chart(self, data: pd.DataFrame, title: str) -> go.Figure:
        """Create performance chart showing returns and cumulative returns"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Daily Returns', 'Cumulative Returns'],
            vertical_spacing=0.1
        )
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        
        # Add daily returns
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                mode='lines',
                name='Daily Returns',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Add cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=self.config.chart_width,
            height=self.config.chart_height,
            showlegend=True
        )
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, title: str) -> go.Figure:
        """Create heatmap for performance metrics"""
        # Prepare data for heatmap
        heatmap_data = data[self.config.heatmap_metrics].values
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=self.config.heatmap_metrics,
            y=data.index,
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            width=self.config.chart_width,
            height=self.config.chart_height
        )
        
        return fig
    
    def create_regime_heatmap(self, regime_data: pd.DataFrame, title: str) -> go.Figure:
        """Create heatmap showing regime transitions"""
        # Create regime transition matrix
        regimes = regime_data['regime'].unique()
        transition_matrix = np.zeros((len(regimes), len(regimes)))
        
        for i, from_regime in enumerate(regimes):
            for j, to_regime in enumerate(regimes):
                transitions = ((regime_data['regime'].shift(1) == from_regime) & 
                             (regime_data['regime'] == to_regime)).sum()
                transition_matrix[i, j] = transitions
        
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix,
            x=regimes,
            y=regimes,
            colorscale='Blues',
            showscale=True,
            text=transition_matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title=title,
            width=self.config.chart_width,
            height=self.config.chart_height
        )
        
        return fig
    
    def create_strategy_comparison_chart(self, results: List[Dict[str, Any]], title: str) -> go.Figure:
        """Create chart comparing multiple strategies"""
        fig = go.Figure()
        
        for result in results:
            strategy_name = result.get('strategy_name', 'Unknown')
            returns = result.get('returns', [])
            
            if returns:
                cumulative_returns = (1 + pd.Series(returns)).cumprod()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name=strategy_name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Cumulative Returns',
            width=self.config.chart_width,
            height=self.config.chart_height,
            showlegend=True
        )
        
        return fig
    
    def create_risk_analysis_chart(self, data: pd.DataFrame, title: str) -> go.Figure:
        """Create risk analysis chart showing drawdowns and risk metrics"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Price', 'Drawdown', 'Volatility'],
            vertical_spacing=0.1
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown chart
        returns = data['close'].pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Volatility chart
        volatility = returns.rolling(window=20).std()
        fig.add_trace(
            go.Scatter(
                x=volatility.index,
                y=volatility,
                mode='lines',
                name='Volatility',
                line=dict(color='orange', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=title,
            width=self.config.chart_width,
            height=self.config.chart_height,
            showlegend=True
        )
        
        return fig
    
    def add_regime_overlay(self, fig: go.Figure, regime_data: pd.DataFrame):
        """Add regime overlay to existing chart"""
        if regime_data is None or 'regime' not in regime_data.columns:
            return
        
        # Get unique regimes
        regimes = regime_data['regime'].unique()
        
        for regime in regimes:
            if pd.isna(regime):
                continue
            
            regime_mask = regime_data['regime'] == regime
            regime_periods = regime_data[regime_mask]
            
            if len(regime_periods) == 0:
                continue
            
            # Add regime background
            color = self.config.regime_colors.get(regime, '#808080')
            
            for _, period in regime_periods.iterrows():
                fig.add_vrect(
                    x0=period.name,
                    x1=period.name + timedelta(days=1),
                    fillcolor=color,
                    opacity=self.config.regime_alpha,
                    layer="below",
                    line_width=0,
                    annotation_text=regime,
                    annotation_position="top left"
                )
    
    def create_summary_dashboard(self, results: List[Dict[str, Any]], 
                               regime_data: Optional[Dict[str, pd.DataFrame]] = None) -> go.Figure:
        """Create comprehensive summary dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Strategy Performance', 'Risk Metrics', 'Regime Analysis', 'Heatmap'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Strategy Performance
        for result in results:
            strategy_name = result.get('strategy_name', 'Unknown')
            returns = result.get('returns', [])
            
            if returns:
                cumulative_returns = (1 + pd.Series(returns)).cumprod()
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns,
                        mode='lines',
                        name=strategy_name
                    ),
                    row=1, col=1
                )
        
        # Risk Metrics
        risk_metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        strategy_names = [r.get('strategy_name', 'Unknown') for r in results]
        
        for i, metric in enumerate(risk_metrics):
            values = [r.get(metric, 0) for r in results]
            fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=values,
                    name=metric
                ),
                row=1, col=2
            )
        
        # Regime Analysis (if available)
        if regime_data:
            regime_counts = {}
            for file_name, data in regime_data.items():
                if 'regime' in data.columns:
                    counts = data['regime'].value_counts()
                    regime_counts[file_name] = counts
            
            if regime_counts:
                # Create regime distribution chart
                all_regimes = set()
                for counts in regime_counts.values():
                    all_regimes.update(counts.index)
                
                for regime in all_regimes:
                    values = [counts.get(regime, 0) for counts in regime_counts.values()]
                    fig.add_trace(
                        go.Bar(
                            x=list(regime_counts.keys()),
                            y=values,
                            name=f'Regime: {regime}'
                        ),
                        row=2, col=1
                    )
        
        # Heatmap
        if results:
            heatmap_data = []
            for result in results:
                row = [result.get(metric, 0) for metric in self.config.heatmap_metrics]
                heatmap_data.append(row)
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=self.config.heatmap_metrics,
                    y=strategy_names,
                    colorscale='RdYlGn',
                    showscale=True
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Comprehensive Trading Analysis Dashboard",
            width=self.config.chart_width * 1.5,
            height=self.config.chart_height * 1.5,
            showlegend=True
        )
        
        return fig
    
    def organize_results_by_strategy(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize results by strategy name"""
        organized = {}
        
        for result in results:
            strategy_name = result.get('strategy_name', 'Unknown')
            if strategy_name not in organized:
                organized[strategy_name] = []
            organized[strategy_name].append(result)
        
        return organized
    
    def organize_results_by_data_file(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize results by data file"""
        organized = {}
        
        for result in results:
            data_file = result.get('data_file', 'Unknown')
            if data_file not in organized:
                organized[data_file] = []
            organized[data_file].append(result)
        
        return organized
    
    def save_chart(self, fig: go.Figure, filepath: str, title: str):
        """Save chart in multiple formats"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.config.save_html:
            html_path = f"{filepath}.html"
            fig.write_html(html_path)
            self.logger.info(f"Chart saved as HTML: {html_path}")
        
        if self.config.save_png:
            png_path = f"{filepath}.png"
            fig.write_image(png_path, width=self.config.chart_width, height=self.config.chart_height)
            self.logger.info(f"Chart saved as PNG: {png_path}")
        
        if self.config.save_pdf:
            pdf_path = f"{filepath}.pdf"
            fig.write_image(pdf_path, width=self.config.chart_width, height=self.config.chart_height)
            self.logger.info(f"Chart saved as PDF: {pdf_path}")
    
    async def run_visualization_analysis(self, results: List[Dict[str, Any]], 
                                       regime_data: Optional[Dict[str, pd.DataFrame]] = None) -> str:
        """Run comprehensive visualization analysis"""
        self.logger.info("Starting enhanced visualization analysis...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"{self.config.results_path}/visualizations_{timestamp}"
        
        # Create main visualization directory
        os.makedirs(base_path, exist_ok=True)
        
        # Organize results
        if self.config.organize_by_strategy:
            strategy_results = self.organize_results_by_strategy(results)
            
            for strategy_name, strategy_data in strategy_results.items():
                strategy_path = f"{base_path}/by_strategy/{strategy_name}"
                os.makedirs(strategy_path, exist_ok=True)
                
                # Create strategy-specific charts
                self.create_strategy_charts(strategy_data, strategy_path, strategy_name, regime_data)
        
        if self.config.organize_by_data_file:
            file_results = self.organize_results_by_data_file(results)
            
            for data_file, file_data in file_results.items():
                file_name = Path(data_file).stem
                file_path = f"{base_path}/by_data_file/{file_name}"
                os.makedirs(file_path, exist_ok=True)
                
                # Create file-specific charts
                self.create_file_charts(file_data, file_path, file_name, regime_data)
        
        # Create summary dashboard
        if self.config.create_summary_dashboard:
            dashboard = self.create_summary_dashboard(results, regime_data)
            self.save_chart(dashboard, f"{base_path}/summary_dashboard", "Summary Dashboard")
        
        self.logger.info(f"Visualization analysis complete. Results saved to: {base_path}")
        return base_path
    
    def create_strategy_charts(self, results: List[Dict[str, Any]], path: str, 
                             strategy_name: str, regime_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Create charts for a specific strategy"""
        for result in results:
            data_file = result.get('data_file', 'Unknown')
            file_name = Path(data_file).stem
            
            # Create performance chart
            if 'data' in result:
                data = result['data']
                performance_chart = self.create_performance_chart(data, f"{strategy_name} - {file_name}")
                self.save_chart(performance_chart, f"{path}/performance_{file_name}", 
                              f"{strategy_name} Performance - {file_name}")
                
                # Create risk analysis chart
                risk_chart = self.create_risk_analysis_chart(data, f"{strategy_name} - Risk Analysis - {file_name}")
                self.save_chart(risk_chart, f"{path}/risk_analysis_{file_name}", 
                              f"{strategy_name} Risk Analysis - {file_name}")
                
                # Add regime overlay if available
                if regime_data and data_file in regime_data:
                    regime_chart = self.create_candlestick_chart(data, f"{strategy_name} - {file_name}", 
                                                               regime_data[data_file])
                    self.save_chart(regime_chart, f"{path}/candlestick_regime_{file_name}", 
                                  f"{strategy_name} with Regime Overlay - {file_name}")
    
    def create_file_charts(self, results: List[Dict[str, Any]], path: str, 
                          file_name: str, regime_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Create charts for a specific data file"""
        # Create strategy comparison chart
        comparison_chart = self.create_strategy_comparison_chart(results, f"Strategy Comparison - {file_name}")
        self.save_chart(comparison_chart, f"{path}/strategy_comparison", 
                       f"Strategy Comparison - {file_name}")
        
        # Create heatmap
        if results:
            results_df = pd.DataFrame(results)
            heatmap_chart = self.create_heatmap(results_df, f"Performance Heatmap - {file_name}")
            self.save_chart(heatmap_chart, f"{path}/performance_heatmap", 
                          f"Performance Heatmap - {file_name}")
        
        # Create regime heatmap if available
        if regime_data and file_name in regime_data:
            regime_heatmap = self.create_regime_heatmap(regime_data[file_name], f"Regime Analysis - {file_name}")
            self.save_chart(regime_heatmap, f"{path}/regime_heatmap", 
                          f"Regime Analysis - {file_name}")

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = VisualizationConfig(
        data_path="./Data",
        results_path="./Results/Visualizations",
        save_csv=True,
        save_png=True,
        save_html=True,
        enable_regime_overlay=True
    )
    
    # Create engine
    engine = EnhancedVisualizationEngine(config)
    
    print("Enhanced Visualization Engine created successfully!")
    print("Ready to generate comprehensive charts and visualizations.")
