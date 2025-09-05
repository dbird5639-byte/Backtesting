"""
Strategy Factory Pattern
Based on methodologies from AI projects for systematic strategy creation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

class StrategyCategory(Enum):
    """Strategy categories based on AI project methodologies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_MAKING = "market_making"
    REGIME_BASED = "regime_based"
    ENSEMBLE = "ensemble"
    # Bull Run Strategies
    LIQUIDATION_HUNT = "liquidation_hunt"
    MOMENTUM_EXPLOSION = "momentum_explosion"
    FOMO_CAPTURE = "fomo_capture"
    INSTITUTIONAL_FLOW = "institutional_flow"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    SOCIAL_SENTIMENT = "social_sentiment"
    OPTIONS_FLOW = "options_flow"
    MULTI_ASSET_CORRELATION = "multi_asset_correlation"
    NEWS_CATALYST = "news_catalyst"
    TECHNICAL_BREAKOUT_CASCADE = "technical_breakout_cascade"
    QUANTUM_INSPIRED = "quantum_inspired"

class RiskLevel(Enum):
    """Risk levels for strategies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class StrategyConfig:
    """Configuration for strategy creation"""
    name: str
    category: StrategyCategory
    risk_level: RiskLevel
    timeframes: List[str]
    assets: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    expected_performance: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    author: str = "AI Strategy Factory"
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class StrategySignal:
    """Trading signal data structure"""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    strength: float  # Signal strength between -1 and 1
    price: float
    quantity: float
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.signals: List[StrategySignal] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.is_active = False
        
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        pass
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required configuration parameters"""
        return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            "name": self.config.name,
            "category": self.config.category.value,
            "risk_level": self.config.risk_level.value,
            "total_signals": len(self.signals),
            "is_active": self.is_active,
            "performance_metrics": self.performance_metrics
        }

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.required_params = ["lookback_period", "momentum_threshold", "rsi_period", "rsi_overbought", "rsi_oversold"]
    
    def get_required_parameters(self) -> List[str]:
        return self.required_params
    
    def validate_config(self) -> bool:
        for param in self.required_params:
            if param not in self.config.parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        return True
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        if data.empty:
            return data
        
        lookback = self.config.parameters.get("lookback_period", 20)
        rsi_period = self.config.parameters.get("rsi_period", 14)
        
        # Momentum
        data['momentum'] = data['close'].pct_change(lookback)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['sma_short'] = data['close'].rolling(window=10).mean()
        data['sma_long'] = data['close'].rolling(window=30).mean()
        
        # Volume ratio
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
        else:
            data['volume_ratio'] = 1.0
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """Generate momentum signals"""
        signals = []
        
        if not self.validate_config():
            return signals
        
        data_with_indicators = self.calculate_indicators(data)
        threshold = self.config.parameters.get("momentum_threshold", 0.02)
        rsi_overbought = self.config.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.config.parameters.get("rsi_oversold", 30)
        
        for i in range(1, len(data_with_indicators)):
            current = data_with_indicators.iloc[i]
            signal_strength = 0.0
            action = "HOLD"
            reason_parts = []
            
            # Momentum signal
            if current['momentum'] > threshold:
                signal_strength += 0.4
                reason_parts.append(f"Positive momentum: {current['momentum']:.3f}")
            elif current['momentum'] < -threshold:
                signal_strength -= 0.4
                reason_parts.append(f"Negative momentum: {current['momentum']:.3f}")
            
            # RSI signal
            if current['rsi'] < rsi_oversold:
                signal_strength += 0.3
                reason_parts.append(f"RSI oversold: {current['rsi']:.1f}")
            elif current['rsi'] > rsi_overbought:
                signal_strength -= 0.3
                reason_parts.append(f"RSI overbought: {current['rsi']:.1f}")
            
            # Moving average crossover
            if current['sma_short'] > current['sma_long']:
                signal_strength += 0.2
                reason_parts.append("MA crossover bullish")
            else:
                signal_strength -= 0.2
                reason_parts.append("MA crossover bearish")
            
            # Volume confirmation
            if current['volume_ratio'] > 1.2:
                signal_strength *= 1.1
                reason_parts.append(f"High volume: {current['volume_ratio']:.1f}x")
            
            # Determine action
            if signal_strength > 0.5:
                action = "BUY"
            elif signal_strength < -0.5:
                action = "SELL"
            
            if action != "HOLD":
                signal = StrategySignal(
                    timestamp=data_with_indicators.index[i],
                    symbol="BTC",  # Default symbol
                    action=action,
                    strength=abs(signal_strength),
                    price=current['close'],
                    quantity=0,  # Will be calculated by position sizer
                    confidence=min(1.0, abs(signal_strength)),
                    reason="; ".join(reason_parts),
                    metadata={
                        'momentum': current['momentum'],
                        'rsi': current['rsi'],
                        'volume_ratio': current['volume_ratio']
                    }
                )
                signals.append(signal)
        
        return signals

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.required_params = ["lookback_period", "std_dev_threshold", "rsi_period", "rsi_overbought", "rsi_oversold"]
    
    def get_required_parameters(self) -> List[str]:
        return self.required_params
    
    def validate_config(self) -> bool:
        for param in self.required_params:
            if param not in self.config.parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        return True
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        if data.empty:
            return data
        
        lookback = self.config.parameters.get("lookback_period", 50)
        rsi_period = self.config.parameters.get("rsi_period", 14)
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=lookback).mean()
        bb_std = data['close'].rolling(window=lookback).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Z-score
        data['z_score'] = (data['close'] - data['bb_middle']) / bb_std
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Mean reversion probability
        data['reversion_prob'] = 1.0 / (1.0 + np.exp(np.abs(data['z_score']) - 2.0))
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """Generate mean reversion signals"""
        signals = []
        
        if not self.validate_config():
            return signals
        
        data_with_indicators = self.calculate_indicators(data)
        std_dev_threshold = self.config.parameters.get("std_dev_threshold", 2.0)
        rsi_overbought = self.config.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.config.parameters.get("rsi_oversold", 30)
        
        for i in range(1, len(data_with_indicators)):
            current = data_with_indicators.iloc[i]
            signal_strength = 0.0
            action = "HOLD"
            reason_parts = []
            
            # Z-score signal
            if current['z_score'] < -std_dev_threshold:
                signal_strength += 0.4
                reason_parts.append(f"Oversold (z-score: {current['z_score']:.2f})")
            elif current['z_score'] > std_dev_threshold:
                signal_strength -= 0.4
                reason_parts.append(f"Overbought (z-score: {current['z_score']:.2f})")
            
            # Bollinger Bands signal
            if current['close'] <= current['bb_lower']:
                signal_strength += 0.3
                reason_parts.append("Below lower Bollinger Band")
            elif current['close'] >= current['bb_upper']:
                signal_strength -= 0.3
                reason_parts.append("Above upper Bollinger Band")
            
            # RSI signal
            if current['rsi'] < rsi_oversold:
                signal_strength += 0.2
                reason_parts.append(f"RSI oversold: {current['rsi']:.1f}")
            elif current['rsi'] > rsi_overbought:
                signal_strength -= 0.2
                reason_parts.append(f"RSI overbought: {current['rsi']:.1f}")
            
            # Reversion probability boost
            signal_strength *= current['reversion_prob']
            
            # Determine action
            if signal_strength > 0.5:
                action = "BUY"
            elif signal_strength < -0.5:
                action = "SELL"
            
            if action != "HOLD":
                signal = StrategySignal(
                    timestamp=data_with_indicators.index[i],
                    symbol="BTC",  # Default symbol
                    action=action,
                    strength=abs(signal_strength),
                    price=current['close'],
                    quantity=0,  # Will be calculated by position sizer
                    confidence=min(1.0, abs(signal_strength)),
                    reason="; ".join(reason_parts),
                    metadata={
                        'z_score': current['z_score'],
                        'rsi': current['rsi'],
                        'reversion_prob': current['reversion_prob']
                    }
                )
                signals.append(signal)
        
        return signals

class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy"""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.required_params = ["breakout_period", "volume_threshold", "atr_period"]
    
    def get_required_parameters(self) -> List[str]:
        return self.required_params
    
    def validate_config(self) -> bool:
        for param in self.required_params:
            if param not in self.config.parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        return True
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators"""
        if data.empty:
            return data
        
        breakout_period = self.config.parameters.get("breakout_period", 20)
        atr_period = self.config.parameters.get("atr_period", 14)
        
        # Support and resistance levels
        data['resistance'] = data['high'].rolling(window=breakout_period).max()
        data['support'] = data['low'].rolling(window=breakout_period).min()
        
        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Volume indicators
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
        else:
            data['volume_ratio'] = 1.0
        
        # Breakout signals
        data['breakout_up'] = (data['close'] > data['resistance'].shift(1)) & (data['volume_ratio'] > 1.5)
        data['breakout_down'] = (data['close'] < data['support'].shift(1)) & (data['volume_ratio'] > 1.5)
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> List[StrategySignal]:
        """Generate breakout signals"""
        signals = []
        
        if not self.validate_config():
            return signals
        
        data_with_indicators = self.calculate_indicators(data)
        volume_threshold = self.config.parameters.get("volume_threshold", 1.5)
        
        for i in range(1, len(data_with_indicators)):
            current = data_with_indicators.iloc[i]
            previous = data_with_indicators.iloc[i-1]
            signal_strength = 0.0
            action = "HOLD"
            reason_parts = []
            
            # Breakout signals
            if current['breakout_up']:
                signal_strength = 0.8
                action = "BUY"
                reason_parts.append(f"Bullish breakout above {previous['resistance']:.2f}")
            elif current['breakout_down']:
                signal_strength = 0.8
                action = "SELL"
                reason_parts.append(f"Bearish breakout below {previous['support']:.2f}")
            
            # Volume confirmation
            if current['volume_ratio'] > volume_threshold:
                signal_strength *= 1.2
                reason_parts.append(f"High volume: {current['volume_ratio']:.1f}x average")
            
            if action != "HOLD":
                signal = StrategySignal(
                    timestamp=data_with_indicators.index[i],
                    symbol="BTC",  # Default symbol
                    action=action,
                    strength=signal_strength,
                    price=current['close'],
                    quantity=0,  # Will be calculated by position sizer
                    confidence=min(1.0, signal_strength),
                    reason="; ".join(reason_parts),
                    metadata={
                        'resistance': previous['resistance'],
                        'support': previous['support'],
                        'volume_ratio': current['volume_ratio']
                    }
                )
                signals.append(signal)
        
        return signals

class StrategyFactory:
    """Factory for creating trading strategies"""
    
    def __init__(self):
        self.strategy_classes = {
            StrategyCategory.MOMENTUM: MomentumStrategy,
            StrategyCategory.MEAN_REVERSION: MeanReversionStrategy,
            StrategyCategory.BREAKOUT: BreakoutStrategy,
            # Bull Run Strategies
            StrategyCategory.LIQUIDATION_HUNT: None,  # Will be imported dynamically
            StrategyCategory.MOMENTUM_EXPLOSION: None,
            StrategyCategory.FOMO_CAPTURE: None,
            StrategyCategory.INSTITUTIONAL_FLOW: None,
            StrategyCategory.VOLATILITY_BREAKOUT: None,
            StrategyCategory.SOCIAL_SENTIMENT: None,
            StrategyCategory.OPTIONS_FLOW: None,
            StrategyCategory.MULTI_ASSET_CORRELATION: None,
            StrategyCategory.NEWS_CATALYST: None,
            StrategyCategory.TECHNICAL_BREAKOUT_CASCADE: None,
            StrategyCategory.QUANTUM_INSPIRED: None,
        }
        
        # Default configurations
        self.default_configs = {
            StrategyCategory.MOMENTUM: {
                "lookback_period": 20,
                "momentum_threshold": 0.02,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            },
            StrategyCategory.MEAN_REVERSION: {
                "lookback_period": 50,
                "std_dev_threshold": 2.0,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            },
            StrategyCategory.BREAKOUT: {
                "breakout_period": 20,
                "volume_threshold": 1.5,
                "atr_period": 14
            },
            # Bull Run Strategy Configurations
            StrategyCategory.LIQUIDATION_HUNT: {
                "liquidation_lookback": 50,
                "leverage_threshold": 0.8,
                "volume_spike_threshold": 3.0,
                "price_acceleration_threshold": 0.05,
                "rsi_period": 14,
                "atr_multiplier": 3.0
            },
            StrategyCategory.MOMENTUM_EXPLOSION: {
                "momentum_lookback": 30,
                "explosion_threshold": 0.03,
                "volume_explosion_threshold": 2.5,
                "momentum_acceleration_threshold": 0.02,
                "rsi_period": 14,
                "atr_multiplier": 2.0
            },
            StrategyCategory.FOMO_CAPTURE: {
                "fomo_lookback": 20,
                "volume_surge_threshold": 2.0,
                "price_acceleration_threshold": 0.02,
                "retail_flow_threshold": 0.7,
                "rsi_period": 14,
                "atr_multiplier": 2.5
            },
            StrategyCategory.INSTITUTIONAL_FLOW: {
                "institutional_lookback": 50,
                "large_order_threshold": 0.8,
                "volume_profile_threshold": 0.7,
                "smart_money_threshold": 0.6,
                "rsi_period": 14,
                "atr_multiplier": 1.8
            },
            StrategyCategory.VOLATILITY_BREAKOUT: {
                "volatility_lookback": 30,
                "compression_threshold": 0.5,
                "expansion_threshold": 1.5,
                "breakout_threshold": 2.0,
                "rsi_period": 14,
                "atr_multiplier": 2.5
            },
            StrategyCategory.SOCIAL_SENTIMENT: {
                "sentiment_lookback": 20,
                "fear_threshold": 0.2,
                "greed_threshold": 0.8,
                "euphoria_threshold": 0.9,
                "panic_threshold": 0.1,
                "rsi_period": 14,
                "atr_multiplier": 2.0
            },
            StrategyCategory.OPTIONS_FLOW: {
                "options_lookback": 30,
                "unusual_activity_threshold": 0.7,
                "gamma_squeeze_threshold": 0.8,
                "put_call_ratio_threshold": 0.6,
                "rsi_period": 14,
                "atr_multiplier": 2.2
            },
            StrategyCategory.MULTI_ASSET_CORRELATION: {
                "correlation_lookback": 30,
                "correlation_threshold": 0.7,
                "divergence_threshold": 0.3,
                "rotation_threshold": 0.5,
                "rsi_period": 14,
                "atr_multiplier": 2.0
            },
            StrategyCategory.NEWS_CATALYST: {
                "news_lookback": 20,
                "news_impact_threshold": 0.7,
                "sentiment_threshold": 0.6,
                "reaction_threshold": 0.8,
                "rsi_period": 14,
                "atr_multiplier": 2.3
            },
            StrategyCategory.TECHNICAL_BREAKOUT_CASCADE: {
                "breakout_lookback": 30,
                "breakout_threshold": 0.02,
                "cascade_threshold": 0.8,
                "acceleration_threshold": 0.6,
                "rsi_period": 14,
                "atr_multiplier": 2.5
            },
            StrategyCategory.QUANTUM_INSPIRED: {
                "quantum_lookback": 40,
                "superposition_threshold": 0.7,
                "entanglement_threshold": 0.8,
                "tunneling_threshold": 0.9,
                "rsi_period": 14,
                "atr_multiplier": 3.0
            }
        }
    
    def create_strategy(self, config: StrategyConfig) -> Optional[BaseStrategy]:
        """Create a strategy instance"""
        if config.category not in self.strategy_classes:
            self.logger.error(f"Unknown strategy category: {config.category}")
            return None
        
        # Fill in default parameters
        if not config.parameters:
            config.parameters = self.default_configs.get(config.category, {}).copy()
        else:
            # Merge with defaults
            defaults = self.default_configs.get(config.category, {})
            for key, value in defaults.items():
                if key not in config.parameters:
                    config.parameters[key] = value
        
        strategy_class = self.strategy_classes[config.category]
        strategy = strategy_class(config)
        
        # Validate configuration
        if not strategy.validate_config():
            self.logger.error(f"Invalid configuration for {config.category} strategy")
            return None
        
        return strategy
    
    def get_available_strategies(self) -> List[StrategyCategory]:
        """Get list of available strategy categories"""
        return list(self.strategy_classes.keys())
    
    def get_strategy_template(self, category: StrategyCategory) -> Dict[str, Any]:
        """Get strategy configuration template"""
        return {
            "name": f"{category.value.title()} Strategy",
            "category": category.value,
            "risk_level": "medium",
            "timeframes": ["1h", "4h", "1d"],
            "assets": ["BTC", "ETH"],
            "parameters": self.default_configs.get(category, {}),
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss": 0.02,
                "take_profit": 0.04
            },
            "expected_performance": {
                "expected_return": 0.15,
                "expected_volatility": 0.20,
                "expected_sharpe": 0.75
            },
            "description": f"Default {category.value} strategy configuration"
        }
    
    def save_strategy_config(self, config: StrategyConfig, filename: str) -> bool:
        """Save strategy configuration to file"""
        try:
            config_data = {
                "name": config.name,
                "category": config.category.value,
                "risk_level": config.risk_level.value,
                "timeframes": config.timeframes,
                "assets": config.assets,
                "parameters": config.parameters,
                "risk_management": config.risk_management,
                "expected_performance": config.expected_performance,
                "description": config.description,
                "author": config.author,
                "version": config.version,
                "created_at": config.created_at.isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving strategy config: {e}")
            return False
    
    def load_strategy_config(self, filename: str) -> Optional[StrategyConfig]:
        """Load strategy configuration from file"""
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            config = StrategyConfig(
                name=config_data["name"],
                category=StrategyCategory(config_data["category"]),
                risk_level=RiskLevel(config_data["risk_level"]),
                timeframes=config_data["timeframes"],
                assets=config_data["assets"],
                parameters=config_data.get("parameters", {}),
                risk_management=config_data.get("risk_management", {}),
                expected_performance=config_data.get("expected_performance", {}),
                description=config_data.get("description", ""),
                author=config_data.get("author", "Unknown"),
                version=config_data.get("version", "1.0.0"),
                created_at=datetime.fromisoformat(config_data.get("created_at", datetime.now().isoformat()))
            )
            
            return config
        except Exception as e:
            self.logger.error(f"Error loading strategy config: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize factory
    factory = StrategyFactory()
    
    # Create momentum strategy
    momentum_config = StrategyConfig(
        name="AI Momentum Strategy",
        category=StrategyCategory.MOMENTUM,
        risk_level=RiskLevel.MEDIUM,
        timeframes=["1h", "4h"],
        assets=["BTC", "ETH"],
        description="AI-generated momentum strategy"
    )
    
    momentum_strategy = factory.create_strategy(momentum_config)
    
    if momentum_strategy:
        print(f"Created {momentum_strategy.config.name}")
        print(f"Required parameters: {momentum_strategy.get_required_parameters()}")
        
        # Test with sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        sample_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        signals = momentum_strategy.generate_signals(sample_data)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals[:5]:  # Show first 5 signals
            print(f"{signal.timestamp}: {signal.action} - {signal.reason}")
