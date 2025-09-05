#!/usr/bin/env python3
"""
Bull Run Strategies Runner

A comprehensive runner for all 10 bull run strategies designed for extreme price surges and liquidation expectancy.
This script demonstrates how to use each strategy and provides performance analysis.

Strategies included:
1. Liquidation Hunt Strategy
2. Momentum Explosion Strategy  
3. FOMO Capture Strategy
4. Institutional Flow Strategy
5. Volatility Breakout Strategy
6. Social Sentiment Strategy
7. Options Flow Strategy
8. Multi-Asset Correlation Strategy
9. News Catalyst Strategy
10. Technical Breakout Cascade Strategy
11. Quantum Inspired Bull Run Strategy
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_bull_run_data(periods=1000, start_price=50000):
    """Create sample data that simulates bull run conditions"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='1H')
    
    # Create bull run price data with extreme movements
    base_trend = np.linspace(0, 2, periods)  # Overall upward trend
    volatility = np.random.normal(0, 0.02, periods)  # Random volatility
    extreme_moves = np.random.choice([0, 1], periods, p=[0.95, 0.05])  # 5% chance of extreme moves
    extreme_volatility = np.random.normal(0, 0.08, periods)  # High volatility for extreme moves
    
    # Combine trends
    price_changes = base_trend + volatility + (extreme_moves * extreme_volatility)
    prices = start_price * np.exp(np.cumsum(price_changes))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, periods) * (1 + extreme_moves * 3)
    }, index=dates)
    
    # Ensure High >= Low
    data['High'] = np.maximum(data['High'], data['Low'])
    data['High'] = np.maximum(data['High'], data['Close'])
    data['Low'] = np.minimum(data['Low'], data['Close'])
    
    return data

def run_strategy_analysis(strategy_name, strategy_class, data, config):
    """Run a single strategy and return performance metrics"""
    try:
        logger.info(f"Running {strategy_name}...")
        
        # Create strategy instance
        strategy = strategy_class(config)
        
        # Initialize strategy
        strategy.init()
        
        # Run strategy on data
        signals = []
        for i in range(len(data)):
            strategy.next()
            if hasattr(strategy, 'trades_history') and strategy.trades_history:
                signals.extend(strategy.trades_history[-1:])
        
        # Calculate performance metrics
        total_signals = len(signals)
        if total_signals > 0:
            # Simulate P&L based on signal strength
            total_pnl = sum([signal.get('strength', 0) * 0.1 for signal in signals])
            win_rate = len([s for s in signals if s.get('strength', 0) > 0.5]) / total_signals
        else:
            total_pnl = 0
            win_rate = 0
        
        return {
            'strategy_name': strategy_name,
            'total_signals': total_signals,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        logger.error(f"Error running {strategy_name}: {e}")
        return {
            'strategy_name': strategy_name,
            'total_signals': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'status': f'ERROR: {str(e)}'
        }

def main():
    """Main function to run all bull run strategies"""
    logger.info("Starting Bull Run Strategies Analysis")
    
    # Create sample data
    data = create_sample_bull_run_data(periods=500, start_price=50000)
    logger.info(f"Created sample data with {len(data)} periods")
    
    # Strategy configurations
    strategies = {
        'Liquidation Hunt Strategy': {
            'class': 'LiquidationHuntStrategy',
            'config': {
                'name': 'Liquidation Hunt',
                'category': 'liquidation_hunt',
                'risk_level': 'very_high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'liquidation_lookback': 50,
                    'leverage_threshold': 0.8,
                    'volume_spike_threshold': 3.0,
                    'price_acceleration_threshold': 0.05
                },
                'risk_management': {
                    'max_position_size': 0.02,
                    'stop_loss': 0.03,
                    'take_profit': 0.06
                },
                'expected_performance': {
                    'expected_return': 0.25,
                    'expected_volatility': 0.30,
                    'expected_sharpe': 0.83
                },
                'description': 'Targets liquidation events for maximum profit potential'
            }
        },
        'Momentum Explosion Strategy': {
            'class': 'MomentumExplosionStrategy',
            'config': {
                'name': 'Momentum Explosion',
                'category': 'momentum_explosion',
                'risk_level': 'high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'momentum_lookback': 30,
                    'explosion_threshold': 0.03,
                    'volume_explosion_threshold': 2.5,
                    'momentum_acceleration_threshold': 0.02
                },
                'risk_management': {
                    'max_position_size': 0.015,
                    'stop_loss': 0.025,
                    'take_profit': 0.05
                },
                'expected_performance': {
                    'expected_return': 0.20,
                    'expected_volatility': 0.25,
                    'expected_sharpe': 0.80
                },
                'description': 'Captures extreme momentum moves during bull runs'
            }
        },
        'FOMO Capture Strategy': {
            'class': 'FOMOCaptureStrategy',
            'config': {
                'name': 'FOMO Capture',
                'category': 'fomo_capture',
                'risk_level': 'high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'fomo_lookback': 20,
                    'volume_surge_threshold': 2.0,
                    'price_acceleration_threshold': 0.02,
                    'retail_flow_threshold': 0.7
                },
                'risk_management': {
                    'max_position_size': 0.02,
                    'stop_loss': 0.025,
                    'take_profit': 0.05
                },
                'expected_performance': {
                    'expected_return': 0.22,
                    'expected_volatility': 0.28,
                    'expected_sharpe': 0.79
                },
                'description': 'Captures FOMO buying waves during bull runs'
            }
        },
        'Institutional Flow Strategy': {
            'class': 'InstitutionalFlowStrategy',
            'config': {
                'name': 'Institutional Flow',
                'category': 'institutional_flow',
                'risk_level': 'medium',
                'timeframes': ['4h', '1d'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'institutional_lookback': 50,
                    'large_order_threshold': 0.8,
                    'volume_profile_threshold': 0.7,
                    'smart_money_threshold': 0.6
                },
                'risk_management': {
                    'max_position_size': 0.012,
                    'stop_loss': 0.018,
                    'take_profit': 0.036
                },
                'expected_performance': {
                    'expected_return': 0.18,
                    'expected_volatility': 0.22,
                    'expected_sharpe': 0.82
                },
                'description': 'Captures institutional money flows during bull runs'
            }
        },
        'Volatility Breakout Strategy': {
            'class': 'VolatilityBreakoutStrategy',
            'config': {
                'name': 'Volatility Breakout',
                'category': 'volatility_breakout',
                'risk_level': 'high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'volatility_lookback': 30,
                    'compression_threshold': 0.5,
                    'expansion_threshold': 1.5,
                    'breakout_threshold': 2.0
                },
                'risk_management': {
                    'max_position_size': 0.018,
                    'stop_loss': 0.025,
                    'take_profit': 0.05
                },
                'expected_performance': {
                    'expected_return': 0.21,
                    'expected_volatility': 0.26,
                    'expected_sharpe': 0.81
                },
                'description': 'Captures explosive volatility breakouts during bull runs'
            }
        },
        'Social Sentiment Strategy': {
            'class': 'SocialSentimentStrategy',
            'config': {
                'name': 'Social Sentiment',
                'category': 'social_sentiment',
                'risk_level': 'medium',
                'timeframes': ['4h', '1d'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'sentiment_lookback': 20,
                    'fear_threshold': 0.2,
                    'greed_threshold': 0.8,
                    'euphoria_threshold': 0.9,
                    'panic_threshold': 0.1
                },
                'risk_management': {
                    'max_position_size': 0.015,
                    'stop_loss': 0.02,
                    'take_profit': 0.04
                },
                'expected_performance': {
                    'expected_return': 0.16,
                    'expected_volatility': 0.20,
                    'expected_sharpe': 0.80
                },
                'description': 'Captures social sentiment extremes during bull runs'
            }
        },
        'Options Flow Strategy': {
            'class': 'OptionsFlowStrategy',
            'config': {
                'name': 'Options Flow',
                'category': 'options_flow',
                'risk_level': 'high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'options_lookback': 30,
                    'unusual_activity_threshold': 0.7,
                    'gamma_squeeze_threshold': 0.8,
                    'put_call_ratio_threshold': 0.6
                },
                'risk_management': {
                    'max_position_size': 0.014,
                    'stop_loss': 0.022,
                    'take_profit': 0.044
                },
                'expected_performance': {
                    'expected_return': 0.19,
                    'expected_volatility': 0.24,
                    'expected_sharpe': 0.79
                },
                'description': 'Captures options-driven moves during bull runs'
            }
        },
        'Multi-Asset Correlation Strategy': {
            'class': 'MultiAssetCorrelationStrategy',
            'config': {
                'name': 'Multi-Asset Correlation',
                'category': 'multi_asset_correlation',
                'risk_level': 'medium',
                'timeframes': ['4h', '1d'],
                'assets': ['BTC', 'ETH', 'ADA', 'SOL'],
                'parameters': {
                    'correlation_lookback': 30,
                    'correlation_threshold': 0.7,
                    'divergence_threshold': 0.3,
                    'rotation_threshold': 0.5
                },
                'risk_management': {
                    'max_position_size': 0.013,
                    'stop_loss': 0.02,
                    'take_profit': 0.04
                },
                'expected_performance': {
                    'expected_return': 0.17,
                    'expected_volatility': 0.21,
                    'expected_sharpe': 0.81
                },
                'description': 'Captures correlation changes during bull runs'
            }
        },
        'News Catalyst Strategy': {
            'class': 'NewsCatalystStrategy',
            'config': {
                'name': 'News Catalyst',
                'category': 'news_catalyst',
                'risk_level': 'high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'news_lookback': 20,
                    'news_impact_threshold': 0.7,
                    'sentiment_threshold': 0.6,
                    'reaction_threshold': 0.8
                },
                'risk_management': {
                    'max_position_size': 0.016,
                    'stop_loss': 0.023,
                    'take_profit': 0.046
                },
                'expected_performance': {
                    'expected_return': 0.23,
                    'expected_volatility': 0.29,
                    'expected_sharpe': 0.79
                },
                'description': 'Captures news-driven moves during bull runs'
            }
        },
        'Technical Breakout Cascade Strategy': {
            'class': 'TechnicalBreakoutCascadeStrategy',
            'config': {
                'name': 'Technical Breakout Cascade',
                'category': 'technical_breakout_cascade',
                'risk_level': 'high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'breakout_lookback': 30,
                    'breakout_threshold': 0.02,
                    'cascade_threshold': 0.8,
                    'acceleration_threshold': 0.6
                },
                'risk_management': {
                    'max_position_size': 0.017,
                    'stop_loss': 0.025,
                    'take_profit': 0.05
                },
                'expected_performance': {
                    'expected_return': 0.24,
                    'expected_volatility': 0.27,
                    'expected_sharpe': 0.89
                },
                'description': 'Captures technical breakout cascades during bull runs'
            }
        },
        'Quantum Inspired Bull Run Strategy': {
            'class': 'QuantumInspiredBullRunStrategy',
            'config': {
                'name': 'Quantum Inspired',
                'category': 'quantum_inspired',
                'risk_level': 'very_high',
                'timeframes': ['1h', '4h'],
                'assets': ['BTC', 'ETH'],
                'parameters': {
                    'quantum_lookback': 40,
                    'superposition_threshold': 0.7,
                    'entanglement_threshold': 0.8,
                    'tunneling_threshold': 0.9
                },
                'risk_management': {
                    'max_position_size': 0.02,
                    'stop_loss': 0.03,
                    'take_profit': 0.06
                },
                'expected_performance': {
                    'expected_return': 0.26,
                    'expected_volatility': 0.32,
                    'expected_sharpe': 0.81
                },
                'description': 'Uses quantum-inspired algorithms for bull run trading'
            }
        }
    }
    
    # Run all strategies
    results = []
    for strategy_name, strategy_info in strategies.items():
        result = run_strategy_analysis(
            strategy_name, 
            strategy_info['class'], 
            data, 
            strategy_info['config']
        )
        results.append(result)
    
    # Display results
    print("\n" + "="*80)
    print("BULL RUN STRATEGIES PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"{'Strategy Name':<35} {'Signals':<8} {'P&L':<10} {'Win Rate':<10} {'Status':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['strategy_name']:<35} {result['total_signals']:<8} "
              f"{result['total_pnl']:<10.3f} {result['win_rate']:<10.2f} {result['status']:<15}")
    
    # Summary statistics
    total_signals = sum([r['total_signals'] for r in results])
    total_pnl = sum([r['total_pnl'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['win_rate'] > 0])
    successful_strategies = len([r for r in results if r['status'] == 'SUCCESS'])
    
    print("-"*80)
    print(f"SUMMARY:")
    print(f"Total Strategies: {len(results)}")
    print(f"Successful Strategies: {successful_strategies}")
    print(f"Total Signals Generated: {total_signals}")
    print(f"Total P&L: {total_pnl:.3f}")
    print(f"Average Win Rate: {avg_win_rate:.2f}")
    print("="*80)
    
    # Strategy recommendations
    print("\nSTRATEGY RECOMMENDATIONS FOR BULL RUN CONDITIONS:")
    print("-"*60)
    print("1. LIQUIDATION HUNT: Best for extreme volatility and liquidation events")
    print("2. MOMENTUM EXPLOSION: Ideal for capturing explosive price movements")
    print("3. FOMO CAPTURE: Perfect for retail buying wave detection")
    print("4. INSTITUTIONAL FLOW: Excellent for following smart money")
    print("5. VOLATILITY BREAKOUT: Great for volatility compression/expansion")
    print("6. SOCIAL SENTIMENT: Good for contrarian opportunities")
    print("7. OPTIONS FLOW: Advanced strategy for options-driven moves")
    print("8. MULTI-ASSET CORRELATION: Best for sector rotation plays")
    print("9. NEWS CATALYST: Ideal for event-driven trading")
    print("10. TECHNICAL BREAKOUT CASCADE: Perfect for chain reaction breakouts")
    print("11. QUANTUM INSPIRED: Cutting-edge algorithm for maximum profit potential")
    print("-"*60)
    
    logger.info("Bull Run Strategies Analysis Complete")

if __name__ == "__main__":
    main()
