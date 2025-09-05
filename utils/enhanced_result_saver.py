#!/usr/bin/env python3
"""
Enhanced Result Saver for Backtesting

This module provides comprehensive result saving functionality that saves results
in multiple formats (JSON, PNG, CSV, Excel, HTML) for every strategy-data combination.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedResultSaver:
    """
    Enhanced result saver that saves results in multiple formats for every strategy-data combination.
    """
    
    def __init__(self, base_output_dir: str = "results"):
        self.base_output_dir = base_output_dir
        self.ensure_directories()
        
        # Set up matplotlib style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        try:
            sns.set_palette("husl")
        except:
            pass
    
    def ensure_directories(self):
        """Ensure all output directories exist."""
        directories = [
            self.base_output_dir,
            os.path.join(self.base_output_dir, "json"),
            os.path.join(self.base_output_dir, "png"),
            os.path.join(self.base_output_dir, "csv"),
            os.path.join(self.base_output_dir, "excel"),
            os.path.join(self.base_output_dir, "html"),
            os.path.join(self.base_output_dir, "individual"),
            os.path.join(self.base_output_dir, "aggregated"),
            os.path.join(self.base_output_dir, "reports")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_individual_result(self, result: Dict[str, Any], strategy_name: str, 
                             symbol: str, timeframe: str) -> Dict[str, str]:
        """
        Save individual result in multiple formats.
        
        Args:
            result: Backtest result dictionary
            strategy_name: Name of the strategy
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        
        # Create safe filename
        safe_strategy = strategy_name.replace(' ', '_').replace('/', '_')
        safe_symbol = symbol.replace(' ', '_')
        safe_timeframe = timeframe.replace(' ', '_')
        
        base_filename = f"{safe_strategy}_{safe_symbol}_{safe_timeframe}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save as JSON
        json_dir = os.path.join(self.base_output_dir, "json", "individual")
        os.makedirs(json_dir, exist_ok=True)
        json_file = os.path.join(json_dir, f"{base_filename}_{timestamp}.json")
        
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        saved_files['json'] = json_file
        
        # 2. Save as CSV
        csv_dir = os.path.join(self.base_output_dir, "csv", "individual")
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = os.path.join(csv_dir, f"{base_filename}_{timestamp}.csv")
        
        # Convert result to DataFrame
        df_result = self._result_to_dataframe(result, strategy_name, symbol, timeframe)
        df_result.to_csv(csv_file, index=False)
        saved_files['csv'] = csv_file
        
        # 3. Save as Excel
        excel_dir = os.path.join(self.base_output_dir, "excel", "individual")
        os.makedirs(excel_dir, exist_ok=True)
        excel_file = os.path.join(excel_dir, f"{base_filename}_{timestamp}.xlsx")
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df_result.to_excel(writer, sheet_name='Results', index=False)
                
                # Add summary sheet
                summary_data = self._create_summary_data(result)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            saved_files['excel'] = excel_file
        except Exception as e:
            logger.warning(f"Could not save Excel file: {e}")
        
        # 4. Save as HTML
        html_dir = os.path.join(self.base_output_dir, "html", "individual")
        os.makedirs(html_dir, exist_ok=True)
        html_file = os.path.join(html_dir, f"{base_filename}_{timestamp}.html")
        
        try:
            html_content = self._result_to_html(result, strategy_name, symbol, timeframe)
            with open(html_file, 'w') as f:
                f.write(html_content)
            saved_files['html'] = html_file
        except Exception as e:
            logger.warning(f"Could not save HTML file: {e}")
        
        # 5. Generate and save charts
        png_dir = os.path.join(self.base_output_dir, "png", "individual")
        os.makedirs(png_dir, exist_ok=True)
        
        # Equity curve chart
        if 'equity_curve' in result and result['equity_curve']:
            equity_file = os.path.join(png_dir, f"{base_filename}_equity_curve_{timestamp}.png")
            try:
                self._plot_equity_curve(result['equity_curve'], strategy_name, symbol, timeframe, equity_file)
                saved_files['equity_chart'] = equity_file
            except Exception as e:
                logger.warning(f"Could not save equity curve chart: {e}")
        
        # Performance chart
        perf_file = os.path.join(png_dir, f"{base_filename}_performance_{timestamp}.png")
        try:
            self._plot_performance_chart(result, strategy_name, symbol, timeframe, perf_file)
            saved_files['performance_chart'] = perf_file
        except Exception as e:
            logger.warning(f"Could not save performance chart: {e}")
        
        logger.info(f"Saved individual result for {strategy_name} ({symbol}_{timeframe}) in {len(saved_files)} formats")
        return saved_files
    
    def save_aggregated_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Save aggregated results from multiple strategies.
        
        Args:
            all_results: List of result dictionaries
            
        Returns:
            Dictionary with paths to saved files
        """
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save aggregated JSON
        json_dir = os.path.join(self.base_output_dir, "json", "aggregated")
        os.makedirs(json_dir, exist_ok=True)
        json_file = os.path.join(json_dir, f"all_results_{timestamp}.json")
        
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        saved_files['json'] = json_file
        
        # 2. Save aggregated CSV
        csv_dir = os.path.join(self.base_output_dir, "csv", "aggregated")
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = os.path.join(csv_dir, f"all_results_{timestamp}.csv")
        
        df_aggregated = self._aggregate_results_to_dataframe(all_results)
        df_aggregated.to_csv(csv_file, index=False)
        saved_files['csv'] = csv_file
        
        # 3. Save summary report
        summary_file = os.path.join(csv_dir, f"summary_report_{timestamp}.csv")
        summary_df = self._create_summary_report(all_results)
        summary_df.to_csv(summary_file, index=False)
        saved_files['summary'] = summary_file
        
        # 4. Generate comparison charts
        png_dir = os.path.join(self.base_output_dir, "png", "aggregated")
        os.makedirs(png_dir, exist_ok=True)
        
        # Strategy comparison chart
        comp_file = os.path.join(png_dir, f"strategy_comparison_{timestamp}.png")
        try:
            self._plot_strategy_comparison(all_results, comp_file)
            saved_files['comparison_chart'] = comp_file
        except Exception as e:
            logger.warning(f"Could not save comparison chart: {e}")
        
        # Performance heatmap
        heatmap_file = os.path.join(png_dir, f"performance_heatmap_{timestamp}.png")
        try:
            self._plot_performance_heatmap(all_results, heatmap_file)
            saved_files['heatmap'] = heatmap_file
        except Exception as e:
            logger.warning(f"Could not save heatmap: {e}")
        
        logger.info(f"Saved aggregated results in {len(saved_files)} formats")
        return saved_files
    
    def _result_to_dataframe(self, result: Dict[str, Any], strategy_name: str, 
                            symbol: str, timeframe: str) -> pd.DataFrame:
        """Convert result dictionary to DataFrame for CSV/Excel export."""
        # Extract key metrics
        data = {
            'strategy_name': [strategy_name],
            'symbol': [symbol],
            'timeframe': [timeframe],
            'timestamp': [datetime.now().isoformat()]
        }
        
        # Add performance metrics
        metrics_mapping = {
            'total_return': 'total_return',
            'final_cash': 'final_cash',
            'trades_count': 'trades_count',
            'max_drawdown': 'max_drawdown',
            'sharpe_ratio': 'sharpe_ratio',
            'win_rate': 'win_rate'
        }
        
        for key, result_key in metrics_mapping.items():
            if result_key in result:
                data[key] = [result[result_key]]
            else:
                data[key] = [None]
        
        # Add additional metrics if available
        if 'performance_metrics' in result:
            perf_metrics = result['performance_metrics']
            for key, value in perf_metrics.items():
                if key not in data:
                    data[key] = [value]
        
        return pd.DataFrame(data)
    
    def _create_summary_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary data for Excel export."""
        summary = {}
        
        # Basic info
        if 'strategy_name' in result:
            summary['Strategy'] = result['strategy_name']
        if 'symbol' in result:
            summary['Symbol'] = result['symbol']
        if 'timeframe' in result:
            summary['Timeframe'] = result['timeframe']
        
        # Performance metrics
        if 'total_return' in result:
            summary['Total Return'] = f"{result['total_return']:.2%}"
        if 'final_cash' in result:
            summary['Final Cash'] = f"${result['final_cash']:,.2f}"
        if 'trades_count' in result:
            summary['Total Trades'] = result['trades_count']
        if 'max_drawdown' in result:
            summary['Max Drawdown'] = f"{result['max_drawdown']:.2%}"
        if 'sharpe_ratio' in result:
            summary['Sharpe Ratio'] = f"{result['sharpe_ratio']:.3f}"
        if 'win_rate' in result:
            summary['Win Rate'] = f"{result['win_rate']:.2%}"
        
        return summary
    
    def _result_to_html(self, result: Dict[str, Any], strategy_name: str, 
                       symbol: str, timeframe: str) -> str:
        """Convert result to HTML format."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Results - {strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #6c757d; margin-top: 5px; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtest Results</h1>
                <h2>{strategy_name}</h2>
                <p><strong>Symbol:</strong> {symbol} | <strong>Timeframe:</strong> {timeframe}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
        """
        
        # Add key metrics
        metrics_to_show = [
            ('total_return', 'Total Return', 'percentage'),
            ('final_cash', 'Final Cash', 'currency'),
            ('trades_count', 'Total Trades', 'number'),
            ('max_drawdown', 'Max Drawdown', 'percentage'),
            ('sharpe_ratio', 'Sharpe Ratio', 'number'),
            ('win_rate', 'Win Rate', 'percentage')
        ]
        
        for key, label, format_type in metrics_to_show:
            if key in result:
                value = result[key]
                if format_type == 'percentage':
                    formatted_value = f"{value:.2%}"
                    css_class = "positive" if value > 0 else "negative"
                elif format_type == 'currency':
                    formatted_value = f"${value:,.2f}"
                    css_class = "positive"
                else:
                    formatted_value = f"{value:.3f}"
                    css_class = "positive"
                
                html += f"""
                <div class="metric">
                    <div class="metric-value {css_class}">{formatted_value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _aggregate_results_to_dataframe(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of results to aggregated DataFrame."""
        aggregated_data = []
        
        for result in all_results:
            row = {}
            
            # Extract basic info
            if 'strategy_name' in result:
                row['strategy_name'] = result['strategy_name']
            if 'symbol' in result:
                row['symbol'] = result['symbol']
            if 'timeframe' in result:
                row['timeframe'] = result['timeframe']
            
            # Extract performance metrics
            metrics_mapping = {
                'total_return': 'total_return',
                'final_cash': 'final_cash',
                'trades_count': 'trades_count',
                'max_drawdown': 'max_drawdown',
                'sharpe_ratio': 'sharpe_ratio',
                'win_rate': 'win_rate'
            }
            
            for key, result_key in metrics_mapping.items():
                if result_key in result:
                    row[key] = result[result_key]
                else:
                    row[key] = None
            
            # Extract additional metrics if available
            if 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                for key, value in perf_metrics.items():
                    if key not in row:
                        row[key] = value
            
            aggregated_data.append(row)
        
        return pd.DataFrame(aggregated_data)
    
    def _create_summary_report(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary report DataFrame."""
        if not all_results:
            return pd.DataFrame()
        
        # Calculate summary statistics
        summary_data = []
        
        # Strategy summary
        strategies = set()
        symbols = set()
        timeframes = set()
        
        returns = []
        drawdowns = []
        sharpe_ratios = []
        win_rates = []
        
        for result in all_results:
            if 'strategy_name' in result:
                strategies.add(result['strategy_name'])
            if 'symbol' in result:
                symbols.add(result['symbol'])
            if 'timeframe' in result:
                timeframes.add(result['timeframe'])
            
            if 'total_return' in result:
                returns.append(result['total_return'])
            if 'max_drawdown' in result:
                drawdowns.append(result['max_drawdown'])
            if 'sharpe_ratio' in result:
                sharpe_ratios.append(result['sharpe_ratio'])
            if 'win_rate' in result:
                win_rates.append(result['win_rate'])
        
        # Overall statistics
        summary_data.append({
            'metric': 'Total Tests',
            'value': len(all_results),
            'category': 'overall'
        })
        
        summary_data.append({
            'metric': 'Unique Strategies',
            'value': len(strategies),
            'category': 'overall'
        })
        
        summary_data.append({
            'metric': 'Unique Symbols',
            'value': len(symbols),
            'category': 'overall'
        })
        
        summary_data.append({
            'metric': 'Unique Timeframes',
            'value': len(timeframes),
            'category': 'overall'
        })
        
        # Performance statistics
        if returns:
            summary_data.append({
                'metric': 'Average Return',
                'value': f"{np.mean(returns):.2%}",
                'category': 'performance'
            })
            
            summary_data.append({
                'metric': 'Best Return',
                'value': f"{np.max(returns):.2%}",
                'category': 'performance'
            })
            
            summary_data.append({
                'metric': 'Worst Return',
                'value': f"{np.min(returns):.2%}",
                'category': 'performance'
            })
        
        if drawdowns:
            summary_data.append({
                'metric': 'Average Drawdown',
                'value': f"{np.mean(drawdowns):.2%}",
                'category': 'performance'
            })
        
        if sharpe_ratios:
            summary_data.append({
                'metric': 'Average Sharpe Ratio',
                'value': f"{np.mean(sharpe_ratios):.3f}",
                'category': 'performance'
            })
        
        if win_rates:
            summary_data.append({
                'metric': 'Average Win Rate',
                'value': f"{np.mean(win_rates):.2%}",
                'category': 'performance'
            })
        
        return pd.DataFrame(summary_data)
    
    def _plot_equity_curve(self, equity_curve: List[float], strategy_name: str, 
                           symbol: str, timeframe: str, filepath: str):
        """Plot and save equity curve."""
        plt.figure(figsize=(12, 8))
        
        # Convert to pandas Series if it's a list
        if isinstance(equity_curve, list):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve
        
        # Plot equity curve
        plt.plot(equity_series.index, equity_series.values, linewidth=2, label='Equity Curve')
        
        # Add horizontal line for initial capital
        if len(equity_series) > 0:
            initial_capital = equity_series.iloc[0]
            plt.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        
        plt.title(f'Equity Curve - {strategy_name} ({symbol}_{timeframe})', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_chart(self, result: Dict[str, Any], strategy_name: str, 
                               symbol: str, timeframe: str, filepath: str):
        """Plot and save performance metrics chart."""
        plt.figure(figsize=(10, 6))
        
        # Extract metrics for plotting
        metrics = ['Total Return', 'Max Drawdown', 'Sharpe Ratio', 'Win Rate']
        values = []
        labels = []
        
        if 'total_return' in result:
            values.append(result['total_return'] * 100)  # Convert to percentage
            labels.append('Total Return (%)')
        
        if 'max_drawdown' in result:
            values.append(result['max_drawdown'] * 100)  # Convert to percentage
            labels.append('Max Drawdown (%)')
        
        if 'sharpe_ratio' in result:
            values.append(result['sharpe_ratio'])
            labels.append('Sharpe Ratio')
        
        if 'win_rate' in result:
            values.append(result['win_rate'] * 100)  # Convert to percentage
            labels.append('Win Rate (%)')
        
        if values:
            bars = plt.bar(labels, values, color=['#28a745', '#dc3545', '#007bff', '#ffc107'])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.title(f'Performance Metrics - {strategy_name} ({symbol}_{timeframe})', fontsize=14, fontweight='bold')
            plt.ylabel('Value', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_strategy_comparison(self, all_results: List[Dict[str, Any]], filepath: str):
        """Plot and save strategy comparison chart."""
        if not all_results:
            return
        
        plt.figure(figsize=(14, 8))
        
        # Extract strategy names and returns
        strategy_names = []
        returns = []
        
        for result in all_results:
            if 'strategy_name' in result and 'total_return' in result:
                strategy_names.append(result['strategy_name'])
                returns.append(result['total_return'] * 100)  # Convert to percentage
        
        if strategy_names and returns:
            bars = plt.bar(range(len(strategy_names)), returns, color='skyblue')
            
            # Color bars based on performance
            for i, (bar, return_val) in enumerate(zip(bars, returns)):
                if return_val > 0:
                    bar.set_color('#28a745')  # Green for positive
                else:
                    bar.set_color('#dc3545')  # Red for negative
            
            plt.xlabel('Strategy', fontsize=12)
            plt.ylabel('Total Return (%)', fontsize=12)
            plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
            plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_heatmap(self, all_results: List[Dict[str, Any]], filepath: str):
        """Plot and save performance heatmap."""
        if not all_results:
            return
        
        # Create pivot table for heatmap
        heatmap_data = []
        strategies = set()
        symbols = set()
        
        for result in all_results:
            if 'strategy_name' in result and 'symbol' in result and 'total_return' in result:
                strategies.add(result['strategy_name'])
                symbols.add(result['symbol'])
                heatmap_data.append({
                    'strategy': result['strategy_name'],
                    'symbol': result['symbol'],
                    'return': result['total_return']
                })
        
        if heatmap_data:
            df_heatmap = pd.DataFrame(heatmap_data)
            pivot_table = df_heatmap.pivot(index='strategy', columns='symbol', values='return')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Total Return'})
            plt.title('Strategy vs Symbol Performance Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Symbol', fontsize=12)
            plt.ylabel('Strategy', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
