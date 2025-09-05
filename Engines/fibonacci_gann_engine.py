"""
Modern Fibonacci/Gann Analysis Engine

A comprehensive engine for Fibonacci and Gann analysis with advanced features:
- Fibonacci retracements, extensions, channels, and time zones
- Gann squares, fans, boxes, and angle analysis
- Advanced swing point detection
- Interactive visualizations
- Integration with modern engine architecture
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .core_engine import CoreEngine, EngineConfig as CoreEngineConfig


@dataclass
class FibonacciGannConfig:
    """Configuration for Fibonacci/Gann analysis engine"""
    
    # Data settings
    data_path: str = "./Data"
    results_path: str = "./Results"
    
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
    gann_angles: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8])
    
    # Swing point detection
    swing_window: int = 5
    min_swing_distance: float = 0.01  # Minimum price movement for swing
    
    # Speed resistance
    speed_resistance_enabled: bool = True
    speed_angles: List[float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    
    # Visualization
    generate_charts: bool = True
    save_charts: bool = True
    chart_format: str = "html"  # "html", "png", "pdf"
    interactive_charts: bool = True
    
    # Analysis settings
    lookback_periods: int = 100
    confidence_threshold: float = 0.7


@dataclass
class SwingPoint:
    """Swing point data structure"""
    index: int
    price: float
    timestamp: datetime
    type: str  # 'high' or 'low'
    strength: float
    confirmed: bool = False


@dataclass
class FibonacciLevels:
    """Fibonacci analysis results"""
    swing_high: SwingPoint
    swing_low: SwingPoint
    retracement_levels: Dict[str, float]
    extension_levels: Dict[str, float]
    channel_levels: Dict[str, List[float]]
    time_zones: List[datetime]
    circles: Dict[str, float]


@dataclass
class GannAnalysis:
    """Gann analysis results"""
    gann_square: np.ndarray
    gann_angles: List[float]
    gann_fan_levels: Dict[str, List[float]]
    gann_box_levels: Dict[str, List[float]]
    time_price_relationships: Dict[str, Any]


class FibonacciGannEngine(CoreEngine):
    """
    Modern Fibonacci and Gann analysis engine with advanced capabilities
    """
    
    def __init__(self, config: FibonacciGannConfig):
        # Initialize with core engine config
        core_config = CoreEngineConfig(
            data_path=config.data_path,
            results_path=config.results_path,
            initial_cash=100000.0
        )
        super().__init__(core_config)
        
        self.config = config
        self.swing_points: List[SwingPoint] = []
        self.fibonacci_results: List[FibonacciLevels] = []
        self.gann_results: List[GannAnalysis] = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    async def run_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive Fibonacci and Gann analysis"""
        try:
            self.logger.info("Starting Fibonacci/Gann analysis...")
            
            # Detect swing points
            await self._detect_swing_points(data)
            
            if not self.swing_points:
                self.logger.warning("No swing points detected")
                return {}
            
            # Run Fibonacci analysis
            fibonacci_results = await self._run_fibonacci_analysis(data)
            
            # Run Gann analysis
            gann_results = await self._run_gann_analysis(data)
            
            # Compile results
            results = {
                'swing_points': [self._swing_point_to_dict(sp) for sp in self.swing_points],
                'fibonacci_analysis': fibonacci_results,
                'gann_analysis': gann_results,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': data['timestamp'].iloc[0].isoformat(),
                    'end': data['timestamp'].iloc[-1].isoformat(),
                    'total_periods': len(data)
                }
            }
            
            # Generate visualizations
            if self.config.generate_charts:
                await self._generate_visualizations(data, results)
            
            # Save results
            await self._save_results(results)
            
            self.logger.info("Fibonacci/Gann analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci/Gann analysis: {e}")
            raise
    
    async def _detect_swing_points(self, data: pd.DataFrame) -> None:
        """Advanced swing point detection"""
        try:
            self.swing_points = []
            
            # Use multiple methods for swing detection
            highs, lows = self._detect_swing_points_basic(data)
            highs_advanced, lows_advanced = self._detect_swing_points_advanced(data)
            
            # Combine and filter results
            all_highs = list(set(highs + highs_advanced))
            all_lows = list(set(lows + lows_advanced))
            
            # Create swing point objects
            for idx in all_highs:
                if idx < len(data):
                    swing_point = SwingPoint(
                        index=idx,
                        price=data['high'].iloc[idx],
                        timestamp=data['timestamp'].iloc[idx],
                        type='high',
                        strength=self._calculate_swing_strength(data, idx, 'high')
                    )
                    self.swing_points.append(swing_point)
            
            for idx in all_lows:
                if idx < len(data):
                    swing_point = SwingPoint(
                        index=idx,
                        price=data['low'].iloc[idx],
                        timestamp=data['timestamp'].iloc[idx],
                        type='low',
                        strength=self._calculate_swing_strength(data, idx, 'low')
                    )
                    self.swing_points.append(swing_point)
            
            # Sort by timestamp
            self.swing_points.sort(key=lambda x: x.timestamp)
            
            # Confirm swing points
            self._confirm_swing_points(data)
            
            self.logger.info(f"Detected {len(self.swing_points)} swing points")
            
        except Exception as e:
            self.logger.error(f"Error detecting swing points: {e}")
            raise
    
    def _detect_swing_points_basic(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Basic swing point detection using local maxima/minima"""
        highs = []
        lows = []
        window = self.config.swing_window
        
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
    
    def _detect_swing_points_advanced(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Advanced swing point detection using multiple criteria"""
        highs = []
        lows = []
        
        # Calculate additional indicators
        data['atr'] = self._calculate_atr(data)
        data['volatility'] = data['close'].rolling(window=20).std()
        
        window = self.config.swing_window
        
        for i in range(window, len(data) - window):
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            current_atr = data['atr'].iloc[i]
            
            # Advanced swing high detection
            if (current_high > data['high'].iloc[i-window:i].max() and
                current_high > data['high'].iloc[i+1:i+window+1].max() and
                current_high > data['close'].iloc[i] + current_atr * 0.5):
                highs.append(i)
            
            # Advanced swing low detection
            if (current_low < data['low'].iloc[i-window:i].min() and
                current_low < data['low'].iloc[i+1:i+window+1].min() and
                current_low < data['close'].iloc[i] - current_atr * 0.5):
                lows.append(i)
        
        return highs, lows
    
    def _calculate_swing_strength(self, data: pd.DataFrame, idx: int, swing_type: str) -> float:
        """Calculate the strength of a swing point"""
        try:
            if swing_type == 'high':
                price = data['high'].iloc[idx]
                # Calculate how much higher this point is compared to surrounding points
                window = min(10, idx, len(data) - idx - 1)
                surrounding_highs = data['high'].iloc[max(0, idx-window):min(len(data), idx+window+1)]
                strength = (price - surrounding_highs.min()) / surrounding_highs.mean()
            else:
                price = data['low'].iloc[idx]
                # Calculate how much lower this point is compared to surrounding points
                window = min(10, idx, len(data) - idx - 1)
                surrounding_lows = data['low'].iloc[max(0, idx-window):min(len(data), idx+window+1)]
                strength = (surrounding_lows.max() - price) / surrounding_lows.mean()
            
            return max(0, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating swing strength: {e}")
            return 0.0
    
    def _confirm_swing_points(self, data: pd.DataFrame) -> None:
        """Confirm swing points based on additional criteria"""
        for swing_point in self.swing_points:
            # Check if swing point meets minimum distance requirement
            if swing_point.strength >= self.config.min_swing_distance:
                swing_point.confirmed = True
    
    async def _run_fibonacci_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive Fibonacci analysis"""
        try:
            if not self.swing_points:
                return {}
            
            # Get the most significant swing points
            confirmed_swings = [sp for sp in self.swing_points if sp.confirmed]
            if len(confirmed_swings) < 2:
                return {}
            
            # Use the most recent significant swing high and low
            swing_high = max([sp for sp in confirmed_swings if sp.type == 'high'], 
                           key=lambda x: x.strength, default=None)
            swing_low = max([sp for sp in confirmed_swings if sp.type == 'low'], 
                          key=lambda x: x.strength, default=None)
            
            if not swing_high or not swing_low:
                return {}
            
            # Calculate Fibonacci levels
            retracement_levels = self._calculate_fibonacci_retracements(swing_high.price, swing_low.price)
            extension_levels = self._calculate_fibonacci_extensions(swing_high.price, swing_low.price, swing_low.price)
            channel_levels = self._calculate_fibonacci_channels(data, swing_high, swing_low)
            time_zones = self._calculate_fibonacci_time_zones(data)
            circles = self._calculate_fibonacci_circles(swing_high.price, swing_low.price)
            
            fibonacci_results = {
                'swing_high': self._swing_point_to_dict(swing_high),
                'swing_low': self._swing_point_to_dict(swing_low),
                'retracement_levels': retracement_levels,
                'extension_levels': extension_levels,
                'channel_levels': channel_levels,
                'time_zones': [tz.isoformat() for tz in time_zones],
                'circles': circles
            }
            
            return fibonacci_results
            
        except Exception as e:
            self.logger.error(f"Error in Fibonacci analysis: {e}")
            return {}
    
    def _calculate_fibonacci_retracements(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        price_range = swing_high - swing_low
        retracement_levels = {}
        
        for ratio in self.config.fibonacci_ratios:
            if ratio <= 1.0:  # Retracement levels
                level = swing_high - (price_range * ratio)
                retracement_levels[f'fib_{ratio:.3f}'] = level
        
        return retracement_levels
    
    def _calculate_fibonacci_extensions(self, swing_high: float, swing_low: float, 
                                      retracement_point: float) -> Dict[str, float]:
        """Calculate Fibonacci extension levels"""
        price_range = swing_high - swing_low
        extension_levels = {}
        
        for ratio in self.config.extension_ratios:
            level = retracement_point + (price_range * ratio)
            extension_levels[f'ext_{ratio:.3f}'] = level
        
        return extension_levels
    
    def _calculate_fibonacci_channels(self, data: pd.DataFrame, swing_high: SwingPoint, 
                                    swing_low: SwingPoint) -> Dict[str, List[float]]:
        """Calculate Fibonacci channels"""
        price_range = swing_high.price - swing_low.price
        channels = {}
        
        for ratio in self.config.fibonacci_ratios:
            upper_channel = swing_high.price + (price_range * ratio)
            lower_channel = swing_low.price - (price_range * ratio)
            
            channels[f'channel_{ratio:.3f}_upper'] = [upper_channel] * len(data)
            channels[f'channel_{ratio:.3f}_lower'] = [lower_channel] * len(data)
        
        return channels
    
    def _calculate_fibonacci_time_zones(self, data: pd.DataFrame) -> List[datetime]:
        """Calculate Fibonacci time zones"""
        if not self.config.time_analysis_enabled:
            return []
        
        start_date = data['timestamp'].iloc[0]
        end_date = data['timestamp'].iloc[-1]
        total_days = (end_date - start_date).days
        time_zones = []
        
        for ratio in self.config.time_zone_ratios:
            time_offset = total_days * ratio
            time_zone_date = start_date + timedelta(days=time_offset)
            time_zones.append(time_zone_date)
        
        return time_zones
    
    def _calculate_fibonacci_circles(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """Calculate Fibonacci circle levels"""
        center_price = (swing_high + swing_low) / 2
        radius = (swing_high - swing_low) / 2
        circles = {}
        
        for ratio in self.config.fibonacci_ratios:
            circle_radius = radius * ratio
            upper_circle = center_price + circle_radius
            lower_circle = center_price - circle_radius
            
            circles[f'circle_{ratio:.3f}_upper'] = upper_circle
            circles[f'circle_{ratio:.3f}_lower'] = lower_circle
        
        return circles
    
    async def _run_gann_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive Gann analysis"""
        try:
            if not self.config.gann_analysis_enabled or not self.swing_points:
                return {}
            
            # Get center point for Gann analysis
            confirmed_swings = [sp for sp in self.swing_points if sp.confirmed]
            if not confirmed_swings:
                return {}
            
            center_swing = confirmed_swings[len(confirmed_swings)//2]
            center_price = center_swing.price
            center_time = center_swing.index
            
            # Calculate Gann components
            gann_square = self._calculate_gann_square(center_price, center_time)
            gann_fan = self._calculate_gann_fan(data, confirmed_swings)
            gann_box = self._calculate_gann_box(data, confirmed_swings)
            time_price_relationships = self._calculate_time_price_relationships(data)
            
            gann_results = {
                'gann_square': gann_square.tolist() if gann_square.size > 0 else [],
                'gann_fan': gann_fan,
                'gann_box': gann_box,
                'time_price_relationships': time_price_relationships,
                'center_point': {
                    'price': center_price,
                    'time_index': center_time,
                    'timestamp': center_swing.timestamp.isoformat()
                }
            }
            
            return gann_results
            
        except Exception as e:
            self.logger.error(f"Error in Gann analysis: {e}")
            return {}
    
    def _calculate_gann_square(self, center_price: float, center_time: int) -> np.ndarray:
        """Calculate Gann square"""
        size = self.config.gann_square_size
        gann_square = np.zeros((size, size))
        
        # Fill Gann square with price and time relationships
        for i in range(size):
            for j in range(size):
                price_offset = (i - size//2) * 0.1 * center_price
                time_offset = (j - size//2) * 1
                
                gann_square[i, j] = center_price + price_offset + (time_offset * 0.001 * center_price)
        
        return gann_square
    
    def _calculate_gann_fan(self, data: pd.DataFrame, swing_points: List[SwingPoint]) -> Dict[str, List[float]]:
        """Calculate Gann fan lines"""
        if not swing_points:
            return {}
        
        # Use the most significant swing points
        swing_high = max([sp for sp in swing_points if sp.type == 'high'], 
                        key=lambda x: x.strength, default=None)
        swing_low = max([sp for sp in swing_points if sp.type == 'low'], 
                       key=lambda x: x.strength, default=None)
        
        if not swing_high or not swing_low:
            return {}
        
        price_range = swing_high.price - swing_low.price
        fan_lines = {}
        
        for angle in self.config.gann_angles:
            fan_levels = []
            
            for i in range(len(data)):
                # Calculate Gann fan line
                time_factor = i / len(data)
                fan_price = swing_high.price - (price_range * time_factor * angle / 8)
                fan_levels.append(fan_price)
            
            fan_lines[f'gann_fan_{angle}x1'] = fan_levels
        
        return fan_lines
    
    def _calculate_gann_box(self, data: pd.DataFrame, swing_points: List[SwingPoint]) -> Dict[str, List[float]]:
        """Calculate Gann box levels"""
        if not swing_points:
            return {}
        
        # Use the most significant swing points
        swing_high = max([sp for sp in swing_points if sp.type == 'high'], 
                        key=lambda x: x.strength, default=None)
        swing_low = max([sp for sp in swing_points if sp.type == 'low'], 
                       key=lambda x: x.strength, default=None)
        
        if not swing_high or not swing_low:
            return {}
        
        price_range = swing_high.price - swing_low.price
        box_levels = {}
        
        # Gann box divisions
        box_ratios = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        
        for ratio in box_ratios:
            box_price = swing_low.price + (price_range * ratio)
            box_levels[f'gann_box_{ratio:.3f}'] = [box_price] * len(data)
        
        return box_levels
    
    def _calculate_time_price_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-price relationships for Gann analysis"""
        try:
            # Calculate price velocity
            price_changes = data['close'].diff()
            time_changes = pd.Series(range(len(data))).diff()
            
            # Calculate price acceleration
            price_velocity = price_changes / time_changes
            price_acceleration = price_velocity.diff()
            
            # Calculate time cycles
            price_cycles = self._detect_price_cycles(data)
            
            relationships = {
                'price_velocity_mean': float(price_velocity.mean()),
                'price_velocity_std': float(price_velocity.std()),
                'price_acceleration_mean': float(price_acceleration.mean()),
                'price_acceleration_std': float(price_acceleration.std()),
                'detected_cycles': price_cycles
            }
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error calculating time-price relationships: {e}")
            return {}
    
    def _detect_price_cycles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect price cycles using FFT"""
        try:
            # Use close prices for cycle detection
            prices = data['close'].values
            
            # Apply FFT
            fft = np.fft.fft(prices)
            freqs = np.fft.fftfreq(len(prices))
            
            # Find dominant frequencies
            magnitude = np.abs(fft)
            dominant_indices = np.argsort(magnitude)[-5:]  # Top 5 cycles
            
            cycles = []
            for idx in dominant_indices:
                if freqs[idx] > 0:  # Only positive frequencies
                    period = 1 / freqs[idx]
                    cycles.append({
                        'period': float(period),
                        'frequency': float(freqs[idx]),
                        'magnitude': float(magnitude[idx])
                    })
            
            return cycles
            
        except Exception as e:
            self.logger.error(f"Error detecting price cycles: {e}")
            return []
    
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
    
    def _swing_point_to_dict(self, swing_point: SwingPoint) -> Dict[str, Any]:
        """Convert swing point to dictionary"""
        return {
            'index': swing_point.index,
            'price': swing_point.price,
            'timestamp': swing_point.timestamp.isoformat(),
            'type': swing_point.type,
            'strength': swing_point.strength,
            'confirmed': swing_point.confirmed
        }
    
    async def _generate_visualizations(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Generate comprehensive visualizations"""
        try:
            if not self.config.generate_charts:
                return
            
            output_path = Path(self.config.results_path) / "fibonacci_gann_charts"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate main analysis chart
            await self._create_main_chart(data, results, output_path)
            
            # Generate Fibonacci-specific charts
            await self._create_fibonacci_charts(data, results, output_path)
            
            # Generate Gann-specific charts
            await self._create_gann_charts(data, results, output_path)
            
            # Generate swing point analysis chart
            await self._create_swing_analysis_chart(data, results, output_path)
            
            self.logger.info(f"Visualizations saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    async def _create_main_chart(self, data: pd.DataFrame, results: Dict[str, Any], output_path: Path) -> None:
        """Create main analysis chart with all levels"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price with Fibonacci & Gann Levels', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.8, 0.2]
            )
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                showlegend=False
            ), row=1, col=1)
            
            # Add Fibonacci retracement levels
            if 'fibonacci_analysis' in results and 'retracement_levels' in results['fibonacci_analysis']:
                retracements = results['fibonacci_analysis']['retracement_levels']
                for level_name, level_price in retracements.items():
                    fig.add_hline(
                        y=level_price,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Fib {level_name}",
                        annotation_position="right",
                        row=1, col=1
                    )
            
            # Add Gann fan lines
            if 'gann_analysis' in results and 'gann_fan' in results['gann_analysis']:
                gann_fan = results['gann_analysis']['gann_fan']
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                
                for i, (fan_name, fan_levels) in enumerate(gann_fan.items()):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=data['timestamp'],
                        y=fan_levels,
                        mode='lines',
                        name=fan_name,
                        line=dict(color=color, dash='dash'),
                        showlegend=True
                    ), row=1, col=1)
            
            # Add swing points
            if 'swing_points' in results:
                swing_highs = [sp for sp in results['swing_points'] if sp['type'] == 'high']
                swing_lows = [sp for sp in results['swing_points'] if sp['type'] == 'low']
                
                if swing_highs:
                    fig.add_trace(go.Scatter(
                        x=[sp['timestamp'] for sp in swing_highs],
                        y=[sp['price'] for sp in swing_highs],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='triangle-down'),
                        name='Swing Highs',
                        showlegend=True
                    ), row=1, col=1)
                
                if swing_lows:
                    fig.add_trace(go.Scatter(
                        x=[sp['timestamp'] for sp in swing_lows],
                        y=[sp['price'] for sp in swing_lows],
                        mode='markers',
                        marker=dict(size=10, color='green', symbol='triangle-up'),
                        name='Swing Lows',
                        showlegend=True
                    ), row=1, col=1)
            
            # Add volume
            fig.add_trace(go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name='Volume',
                marker_color='lightblue',
                showlegend=False
            ), row=2, col=1)
            
            fig.update_layout(
                title='Comprehensive Fibonacci & Gann Analysis',
                height=800,
                xaxis_rangeslider_visible=False
            )
            
            # Save chart
            chart_file = output_path / "main_analysis_chart.html"
            fig.write_html(str(chart_file))
            
        except Exception as e:
            self.logger.error(f"Error creating main chart: {e}")
    
    async def _create_fibonacci_charts(self, data: pd.DataFrame, results: Dict[str, Any], output_path: Path) -> None:
        """Create Fibonacci-specific charts"""
        try:
            if 'fibonacci_analysis' not in results:
                return
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add Fibonacci levels
            fib_analysis = results['fibonacci_analysis']
            
            # Retracement levels
            if 'retracement_levels' in fib_analysis:
                for level_name, level_price in fib_analysis['retracement_levels'].items():
                    fig.add_hline(
                        y=level_price,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Fib {level_name}",
                        annotation_position="right"
                    )
            
            # Extension levels
            if 'extension_levels' in fib_analysis:
                for level_name, level_price in fib_analysis['extension_levels'].items():
                    fig.add_hline(
                        y=level_price,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Ext {level_name}",
                        annotation_position="right"
                    )
            
            # Add swing points
            if 'swing_high' in fib_analysis:
                swing_high = fib_analysis['swing_high']
                fig.add_trace(go.Scatter(
                    x=[swing_high['timestamp']],
                    y=[swing_high['price']],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='triangle-down'),
                    name='Swing High'
                ))
            
            if 'swing_low' in fib_analysis:
                swing_low = fib_analysis['swing_low']
                fig.add_trace(go.Scatter(
                    x=[swing_low['timestamp']],
                    y=[swing_low['price']],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='triangle-up'),
                    name='Swing Low'
                ))
            
            fig.update_layout(
                title='Fibonacci Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = output_path / "fibonacci_analysis.html"
            fig.write_html(str(chart_file))
            
        except Exception as e:
            self.logger.error(f"Error creating Fibonacci charts: {e}")
    
    async def _create_gann_charts(self, data: pd.DataFrame, results: Dict[str, Any], output_path: Path) -> None:
        """Create Gann-specific charts"""
        try:
            if 'gann_analysis' not in results:
                return
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add Gann fan lines
            gann_analysis = results['gann_analysis']
            if 'gann_fan' in gann_analysis:
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                
                for i, (fan_name, fan_levels) in enumerate(gann_analysis['gann_fan'].items()):
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=data['timestamp'],
                        y=fan_levels,
                        mode='lines',
                        name=fan_name,
                        line=dict(color=color, dash='dash')
                    ))
            
            # Add Gann box levels
            if 'gann_box' in gann_analysis:
                for box_name, box_levels in gann_analysis['gann_box'].items():
                    fig.add_trace(go.Scatter(
                        x=data['timestamp'],
                        y=box_levels,
                        mode='lines',
                        name=box_name,
                        line=dict(color='gray', dash='dot')
                    ))
            
            fig.update_layout(
                title='Gann Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = output_path / "gann_analysis.html"
            fig.write_html(str(chart_file))
            
        except Exception as e:
            self.logger.error(f"Error creating Gann charts: {e}")
    
    async def _create_swing_analysis_chart(self, data: pd.DataFrame, results: Dict[str, Any], output_path: Path) -> None:
        """Create swing point analysis chart"""
        try:
            if 'swing_points' not in results:
                return
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add swing points with strength-based sizing
            swing_points = results['swing_points']
            for swing_point in swing_points:
                size = max(5, min(20, swing_point['strength'] * 100))
                color = 'red' if swing_point['type'] == 'high' else 'green'
                symbol = 'triangle-down' if swing_point['type'] == 'high' else 'triangle-up'
                
                fig.add_trace(go.Scatter(
                    x=[swing_point['timestamp']],
                    y=[swing_point['price']],
                    mode='markers',
                    marker=dict(size=size, color=color, symbol=symbol),
                    name=f"Swing {swing_point['type'].title()}",
                    text=f"Strength: {swing_point['strength']:.3f}",
                    hovertemplate='<b>%{text}</b><br>Price: %{y}<br>Time: %{x}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Swing Point Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                height=600
            )
            
            # Save chart
            chart_file = output_path / "swing_analysis.html"
            fig.write_html(str(chart_file))
            
        except Exception as e:
            self.logger.error(f"Error creating swing analysis chart: {e}")
    
    async def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results"""
        try:
            output_path = Path(self.config.results_path) / "fibonacci_gann_results"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results to JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = output_path / f"fibonacci_gann_analysis_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    async def run(self) -> Dict[str, Any]:
        """Main run method - loads data and runs analysis"""
        try:
            # Load data
            data = await self.load_data()
            
            if data is None or data.empty:
                self.logger.error("No data loaded for analysis")
                return {}
            
            # Run analysis
            results = await self.run_analysis(data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in main run method: {e}")
            raise
