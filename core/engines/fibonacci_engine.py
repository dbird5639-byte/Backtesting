"""
Fibonacci Analysis Engine

This engine provides comprehensive Fibonacci and Gann analysis tools including:
- Fibonacci retracements, extensions, and channels
- Fibonacci time zones and circles
- Speed resistance arcs and fans
- Gann squares, fans, and boxes
- Advanced geometric analysis for trading
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
import json
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.base.base_engine import BaseEngine, EngineConfig, BacktestResult
from core.base.base_strategy import BaseStrategy, StrategyConfig, Signal, Trade
from core.base.base_data_handler import BaseDataHandler, DataConfig
from core.base.base_risk_manager import BaseRiskManager, RiskConfig

warnings.filterwarnings('ignore')

@dataclass
class FibonacciConfig:
    """Configuration for Fibonacci analysis"""
    # Fibonacci ratios
    fibonacci_ratios: List[float] = field(default_factory=lambda: [
        0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618
    ])
    
    # Extension ratios
    extension_ratios: List[float] = field(default_factory=lambda: [
        1.272, 1.618, 2.000, 2.618, 3.000, 3.618, 4.236
    ])
    
    # Time analysis
    time_analysis_enabled: bool = True
    time_zone_ratios: List[float] = field(default_factory=lambda: [
        0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618
    ])
    
    # Gann analysis
    gann_analysis_enabled: bool = True
    gann_square_size: int = 9
    gann_angle_step: float = 1.0
    
    # Speed resistance
    speed_resistance_enabled: bool = True
    speed_angles: List[float] = field(default_factory=lambda: [
        1, 2, 3, 4, 5, 6, 7, 8
    ])
    
    # Visualization
    generate_charts: bool = True
    save_charts: bool = True
    chart_format: str = "html"  # "html", "png", "pdf"

@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels"""
    swing_high: float
    swing_low: float
    retracement_levels: Dict[str, float]
    extension_levels: Dict[str, float]
    time_zones: List[datetime]
    support_resistance: List[float]

@dataclass
class GannAnalysis:
    """Gann analysis results"""
    gann_square: np.ndarray
    gann_angles: List[float]
    gann_fan_levels: Dict[str, List[float]]
    gann_box_levels: Dict[str, List[float]]
    time_price_relationships: Dict[str, Any]

class FibonacciEngine(BaseEngine):
    """
    Comprehensive Fibonacci and Gann analysis engine
    """
    
    def __init__(self, config: FibonacciConfig):
        super().__init__(config)
        self.config = config
        self.setup_logging()
        self.fibonacci_results = []
        self.gann_results = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate data for Fibonacci analysis"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                col_mapping = {
                    'time': 'timestamp', 'date': 'timestamp',
                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                }
                data = data.rename(columns=col_mapping)
            
            # Convert timestamp and sort
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp')
            
            # Calculate additional features for analysis
            data = self._calculate_analysis_features(data)
            
            self.logger.info(f"Loaded data for Fibonacci analysis: {len(data)} rows")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _calculate_analysis_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features needed for Fibonacci analysis"""
        try:
            # Price changes
            data['price_change'] = data['close'].pct_change()
            data['high_low_range'] = data['high'] - data['low']
            data['open_close_range'] = abs(data['close'] - data['open'])
            
            # Moving averages for trend identification
            data['ma_20'] = data['close'].rolling(window=20).mean()
            data['ma_50'] = data['close'].rolling(window=50).mean()
            data['ma_200'] = data['close'].rolling(window=200).mean()
            
            # Volatility measures
            data['volatility'] = data['price_change'].rolling(window=20).std()
            data['atr'] = self._calculate_atr(data, window=14)
            
            # Trend strength
            data['trend_strength'] = abs(data['ma_20'] - data['ma_50']) / data['ma_20']
            
            # Support and resistance levels
            data['support_level'] = data['low'].rolling(window=20).min()
            data['resistance_level'] = data['high'].rolling(window=20).max()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating analysis features: {e}")
            return data
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=window).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=data.index)
    
    def identify_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Identify swing highs and lows in the data"""
        try:
            highs = []
            lows = []
            
            for i in range(window, len(data) - window):
                # Check for swing high
                if all(data['high'].iloc[i] > data['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['high'].iloc[i] > data['high'].iloc[i+j] for j in range(1, window+1)):
                    highs.append(i)
                
                # Check for swing low
                if all(data['low'].iloc[i] < data['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['low'].iloc[i] < data['low'].iloc[i+j] for j in range(1, window+1)):
                    lows.append(i)
            
            return highs, lows
            
        except Exception as e:
            self.logger.error(f"Error identifying swing points: {e}")
            return [], []
    
    def calculate_fibonacci_retracements(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            price_range = swing_high - swing_low
            retracement_levels = {}
            
            for ratio in self.config.fibonacci_ratios:
                if ratio <= 1.0:  # Retracement levels
                    level = swing_high - (price_range * ratio)
                    retracement_levels[f'retracement_{ratio:.3f}'] = level
            
            return retracement_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracements: {e}")
            return {}
    
    def calculate_fibonacci_extensions(self, swing_high: float, swing_low: float, 
                                    retracement_point: float) -> Dict[str, float]:
        """Calculate Fibonacci extension levels"""
        try:
            price_range = swing_high - swing_low
            extension_levels = {}
            
            for ratio in self.config.extension_ratios:
                level = retracement_point + (price_range * ratio)
                extension_levels[f'extension_{ratio:.3f}'] = level
            
            return extension_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci extensions: {e}")
            return {}
    
    def calculate_fibonacci_channels(self, data: pd.DataFrame, swing_high_idx: int, 
                                   swing_low_idx: int) -> Dict[str, List[float]]:
        """Calculate Fibonacci channels"""
        try:
            swing_high = data['high'].iloc[swing_high_idx]
            swing_low = data['low'].iloc[swing_low_idx]
            price_range = swing_high - swing_low
            
            channels = {}
            
            # Calculate channel levels for each Fibonacci ratio
            for ratio in self.config.fibonacci_ratios:
                upper_channel = swing_high + (price_range * ratio)
                lower_channel = swing_low - (price_range * ratio)
                
                channels[f'channel_{ratio:.3f}_upper'] = [upper_channel] * len(data)
                channels[f'channel_{ratio:.3f}_lower'] = [lower_channel] * len(data)
            
            return channels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci channels: {e}")
            return {}
    
    def calculate_fibonacci_time_zones(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Calculate Fibonacci time zones"""
        try:
            if not self.config.time_analysis_enabled:
                return []
            
            total_days = (end_date - start_date).days
            time_zones = []
            
            for ratio in self.config.time_zone_ratios:
                time_offset = total_days * ratio
                time_zone_date = start_date + timedelta(days=time_offset)
                time_zones.append(time_zone_date)
            
            return time_zones
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci time zones: {e}")
            return []
    
    def calculate_fibonacci_circles(self, center_price: float, radius: float) -> Dict[str, float]:
        """Calculate Fibonacci circle levels"""
        try:
            circles = {}
            
            for ratio in self.config.fibonacci_ratios:
                circle_radius = radius * ratio
                upper_circle = center_price + circle_radius
                lower_circle = center_price - circle_radius
                
                circles[f'circle_{ratio:.3f}_upper'] = upper_circle
                circles[f'circle_{ratio:.3f}_lower'] = lower_circle
            
            return circles
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci circles: {e}")
            return {}
    
    def calculate_speed_resistance_arcs(self, data: pd.DataFrame, swing_high_idx: int, 
                                      swing_low_idx: int) -> Dict[str, List[float]]:
        """Calculate speed resistance arcs"""
        try:
            if not self.config.speed_resistance_enabled:
                return {}
            
            swing_high = data['high'].iloc[swing_high_idx]
            swing_low = data['low'].iloc[swing_low_idx]
            price_range = swing_high - swing_low
            
            arcs = {}
            
            for angle in self.config.speed_angles:
                arc_levels = []
                
                for i in range(len(data)):
                    # Calculate arc based on time and price relationship
                    time_factor = i / len(data)
                    arc_price = swing_high - (price_range * time_factor * angle / 8)
                    arc_levels.append(arc_price)
                
                arcs[f'arc_{angle}'] = arc_levels
            
            return arcs
            
        except Exception as e:
            self.logger.error(f"Error calculating speed resistance arcs: {e}")
            return {}
    
    def calculate_speed_resistance_fan(self, data: pd.DataFrame, swing_high_idx: int, 
                                     swing_low_idx: int) -> Dict[str, List[float]]:
        """Calculate speed resistance fan"""
        try:
            if not self.config.speed_resistance_enabled:
                return {}
            
            swing_high = data['high'].iloc[swing_high_idx]
            swing_low = data['low'].iloc[swing_low_idx]
            price_range = swing_high - swing_low
            
            fan_lines = {}
            
            for angle in self.config.speed_angles:
                fan_levels = []
                
                for i in range(len(data)):
                    # Calculate fan line based on angle
                    time_factor = i / len(data)
                    fan_price = swing_high - (price_range * time_factor * angle / 8)
                    fan_levels.append(fan_price)
                
                fan_lines[f'fan_{angle}'] = fan_levels
            
            return fan_lines
            
        except Exception as e:
            self.logger.error(f"Error calculating speed resistance fan: {e}")
            return {}
    
    def calculate_gann_square(self, center_price: float, center_time: int) -> np.ndarray:
        """Calculate Gann square"""
        try:
            if not self.config.gann_analysis_enabled:
                return np.array([])
            
            size = self.config.gann_square_size
            gann_square = np.zeros((size, size))
            
            # Fill Gann square with price and time relationships
            for i in range(size):
                for j in range(size):
                    price_offset = (i - size//2) * 0.1 * center_price
                    time_offset = (j - size//2) * 1
                    
                    gann_square[i, j] = center_price + price_offset + (time_offset * 0.001 * center_price)
            
            return gann_square
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann square: {e}")
            return np.array([])
    
    def calculate_gann_fan(self, data: pd.DataFrame, swing_high_idx: int, 
                          swing_low_idx: int) -> Dict[str, List[float]]:
        """Calculate Gann fan lines"""
        try:
            if not self.config.gann_analysis_enabled:
                return {}
            
            swing_high = data['high'].iloc[swing_high_idx]
            swing_low = data['low'].iloc[swing_low_idx]
            price_range = swing_high - swing_low
            
            fan_lines = {}
            
            # Gann angles: 1x1, 2x1, 3x1, 4x1, 6x1, 8x1
            gann_angles = [1, 2, 3, 4, 6, 8]
            
            for angle in gann_angles:
                fan_levels = []
                
                for i in range(len(data)):
                    # Calculate Gann fan line
                    time_factor = i / len(data)
                    fan_price = swing_high - (price_range * time_factor * angle / 8)
                    fan_levels.append(fan_price)
                
                fan_lines[f'gann_fan_{angle}x1'] = fan_levels
            
            return fan_lines
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann fan: {e}")
            return {}
    
    def calculate_gann_box(self, data: pd.DataFrame, swing_high_idx: int, 
                          swing_low_idx: int) -> Dict[str, List[float]]:
        """Calculate Gann box levels"""
        try:
            if not self.config.gann_analysis_enabled:
                return {}
            
            swing_high = data['high'].iloc[swing_high_idx]
            swing_low = data['low'].iloc[swing_low_idx]
            price_range = swing_high - swing_low
            
            box_levels = {}
            
            # Gann box divisions
            box_ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
            
            for ratio in box_ratios:
                box_price = swing_low + (price_range * ratio)
                box_levels[f'gann_box_{ratio:.3f}'] = [box_price] * len(data)
            
            return box_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann box: {e}")
            return {}
    
    def run_comprehensive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive Fibonacci and Gann analysis"""
        try:
            self.logger.info("Starting comprehensive Fibonacci and Gann analysis...")
            
            # Identify swing points
            swing_highs, swing_lows = self.identify_swing_points(data)
            
            if not swing_highs or not swing_lows:
                self.logger.warning("No swing points found for analysis")
                return {}
            
            # Use the most recent significant swing points
            recent_high_idx = swing_highs[-1] if swing_highs else len(data) // 2
            recent_low_idx = swing_lows[-1] if swing_lows else len(data) // 2
            
            swing_high = data['high'].iloc[recent_high_idx]
            swing_low = data['low'].iloc[recent_low_idx]
            
            # Calculate all Fibonacci levels
            retracement_levels = self.calculate_fibonacci_retracements(swing_high, swing_low)
            extension_levels = self.calculate_fibonacci_extensions(swing_high, swing_low, swing_low)
            channels = self.calculate_fibonacci_channels(data, recent_high_idx, recent_low_idx)
            
            # Calculate time analysis
            start_date = data['timestamp'].iloc[0]
            end_date = data['timestamp'].iloc[-1]
            time_zones = self.calculate_fibonacci_time_zones(start_date, end_date)
            
            # Calculate circles
            center_price = (swing_high + swing_low) / 2
            radius = (swing_high - swing_low) / 2
            circles = self.calculate_fibonacci_circles(center_price, radius)
            
            # Calculate speed resistance
            arcs = self.calculate_speed_resistance_arcs(data, recent_high_idx, recent_low_idx)
            fan_lines = self.calculate_speed_resistance_fan(data, recent_high_idx, recent_low_idx)
            
            # Calculate Gann analysis
            gann_square = self.calculate_gann_square(center_price, len(data) // 2)
            gann_fan = self.calculate_gann_fan(data, recent_high_idx, recent_low_idx)
            gann_box = self.calculate_gann_box(data, recent_high_idx, recent_low_idx)
            
            # Compile results
            results = {
                'swing_points': {
                    'high_idx': recent_high_idx,
                    'low_idx': recent_low_idx,
                    'high_price': swing_high,
                    'low_price': swing_low,
                    'high_date': data['timestamp'].iloc[recent_high_idx],
                    'low_date': data['timestamp'].iloc[recent_low_idx]
                },
                'fibonacci_levels': {
                    'retracements': retracement_levels,
                    'extensions': extension_levels,
                    'channels': channels,
                    'circles': circles
                },
                'time_analysis': {
                    'time_zones': time_zones,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'speed_resistance': {
                    'arcs': arcs,
                    'fan_lines': fan_lines
                },
                'gann_analysis': {
                    'square': gann_square.tolist() if gann_square.size > 0 else [],
                    'fan': gann_fan,
                    'box': gann_box
                }
            }
            
            self.fibonacci_results.append(results)
            self.logger.info("Comprehensive analysis completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            raise
    
    def generate_visualizations(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                              output_path: str = None) -> str:
        """Generate comprehensive visualizations"""
        try:
            if not self.config.generate_charts:
                return ""
            
            if output_path is None:
                output_path = os.path.join("./results", f"fibonacci_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Generate main price chart with Fibonacci levels
            self._plot_main_chart(data, analysis_results, output_path)
            
            # Generate Fibonacci retracement chart
            self._plot_retracement_chart(data, analysis_results, output_path)
            
            # Generate time analysis chart
            self._plot_time_analysis_chart(data, analysis_results, output_path)
            
            # Generate Gann analysis chart
            self._plot_gann_analysis_chart(data, analysis_results, output_path)
            
            # Generate speed resistance chart
            self._plot_speed_resistance_chart(data, analysis_results, output_path)
            
            self.logger.info(f"Visualizations saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            return ""
    
    def _plot_main_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any], output_path: str):
        """Plot main price chart with Fibonacci levels"""
        try:
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ))
            
            # Add Fibonacci retracement levels
            retracements = analysis_results['fibonacci_levels']['retracements']
            for level_name, level_price in retracements.items():
                fig.add_hline(
                    y=level_price,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Fib {level_name}",
                    annotation_position="right"
                )
            
            # Add Fibonacci extension levels
            extensions = analysis_results['fibonacci_levels']['extensions']
            for level_name, level_price in extensions.items():
                fig.add_hline(
                    y=level_price,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"Fib {level_name}",
                    annotation_position="right"
                )
            
            # Add swing points
            swing_points = analysis_results['swing_points']
            fig.add_trace(go.Scatter(
                x=[swing_points['high_date']],
                y=[swing_points['high_price']],
                mode='markers',
                marker=dict(size=10, color='red', symbol='triangle-down'),
                name='Swing High'
            ))
            
            fig.add_trace(go.Scatter(
                x=[swing_points['low_date']],
                y=[swing_points['low_price']],
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name='Swing Low'
            ))
            
            fig.update_layout(
                title='Price Chart with Fibonacci Levels',
                xaxis_title='Date',
                yaxis_title='Price',
                height=800
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "main_chart.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting main chart: {e}")
    
    def _plot_retracement_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any], output_path: str):
        """Plot Fibonacci retracement chart"""
        try:
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            # Add retracement levels
            retracements = analysis_results['fibonacci_levels']['retracements']
            swing_points = analysis_results['swing_points']
            
            for level_name, level_price in retracements.items():
                fig.add_hline(
                    y=level_price,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Fib {level_name}",
                    annotation_position="right"
                )
            
            # Add swing points
            fig.add_trace(go.Scatter(
                x=[swing_points['high_date']],
                y=[swing_points['high_price']],
                mode='markers',
                marker=dict(size=10, color='red', symbol='triangle-down'),
                name='Swing High'
            ))
            
            fig.add_trace(go.Scatter(
                x=[swing_points['low_date']],
                y=[swing_points['low_price']],
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name='Swing Low'
            ))
            
            fig.update_layout(
                title='Fibonacci Retracement Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "retracement_chart.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting retracement chart: {e}")
    
    def _plot_time_analysis_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any], output_path: str):
        """Plot time analysis chart"""
        try:
            if not analysis_results['time_analysis']['time_zones']:
                return
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            # Add time zones
            time_zones = analysis_results['time_analysis']['time_zones']
            for i, time_zone in enumerate(time_zones):
                fig.add_vline(
                    x=time_zone,
                    line_dash="dash",
                    line_color="purple",
                    annotation_text=f"Time Zone {i+1}",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title='Fibonacci Time Zones Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "time_analysis_chart.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting time analysis chart: {e}")
    
    def _plot_gann_analysis_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any], output_path: str):
        """Plot Gann analysis chart"""
        try:
            if not analysis_results['gann_analysis']['fan']:
                return
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            # Add Gann fan lines
            gann_fan = analysis_results['gann_analysis']['fan']
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
            
            for i, (fan_name, fan_levels) in enumerate(gann_fan.items()):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=fan_levels,
                    mode='lines',
                    name=fan_name,
                    line=dict(color=color, dash='dash')
                ))
            
            fig.update_layout(
                title='Gann Fan Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "gann_analysis_chart.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting Gann analysis chart: {e}")
    
    def _plot_speed_resistance_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any], output_path: str):
        """Plot speed resistance chart"""
        try:
            if not analysis_results['speed_resistance']['fan_lines']:
                return
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ))
            
            # Add speed resistance fan lines
            fan_lines = analysis_results['speed_resistance']['fan_lines']
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
            
            for i, (fan_name, fan_levels) in enumerate(fan_lines.items()):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=fan_levels,
                    mode='lines',
                    name=fan_name,
                    line=dict(color=color, dash='dot')
                ))
            
            fig.update_layout(
                title='Speed Resistance Fan Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = os.path.join(output_path, "speed_resistance_chart.html")
            fig.write_html(chart_file)
            
        except Exception as e:
            self.logger.error(f"Error plotting speed resistance chart: {e}")
    
    def save_results(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save analysis results"""
        try:
            if output_path is None:
                output_path = os.path.join("./results", f"fibonacci_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Save results to JSON
            results_file = os.path.join(output_path, "fibonacci_analysis_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate visualizations
            if self.config.generate_charts:
                self.generate_visualizations(pd.DataFrame(), results, output_path)
            
            self.logger.info(f"Fibonacci analysis results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise


# Simple strategy class for demonstration
class SimpleMAStrategy(BaseStrategy):
    """Simple moving average strategy for testing"""
    
    def __init__(self, name: str = "SimpleMA", parameters: Dict[str, Any] = None):
        super().__init__(name)
        self.parameters = parameters or {}
        self.short_window = self.parameters.get('short_window', 10)
        self.long_window = self.parameters.get('long_window', 20)
    
    def initialize(self, data: pd.DataFrame):
        """Initialize strategy with data"""
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        signals = pd.Series(0, index=data.index)
        
        if len(data) < self.long_window:
            return signals
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals[short_ma > long_ma] = 1   # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        return signals
