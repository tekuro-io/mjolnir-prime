"""
Multi-Strategy Runner for Trading Algorithm Testing

This module allows running multiple trading strategies simultaneously on the same
tick data to compare their performance side-by-side.
"""

import asyncio
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

from fast_simulation import FastSimulationEngine
from strategy_config import PRESET_CONFIGS, get_preset_config, list_presets


@dataclass
class StrategyResult:
    """Container for individual strategy results"""
    strategy_name: str
    config_name: str
    results: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None


class MultiStrategyRunner:
    """Runner for testing multiple strategies simultaneously"""
    
    def __init__(self, tick_file_path: str, initial_balance: float = 100000, speed_multiplier: int = 1000):
        """
        Initialize multi-strategy runner
        
        Args:
            tick_file_path: Path to JSON Lines file containing tick data
            initial_balance: Starting balance for each strategy
            speed_multiplier: Speed multiplier for simulation
        """
        # Normalize the path to handle Windows path issues
        self.tick_file_path = os.path.normpath(os.path.expanduser(tick_file_path))
        self.initial_balance = initial_balance
        self.speed_multiplier = speed_multiplier
        
        # Verify file exists
        if not os.path.exists(self.tick_file_path):
            raise FileNotFoundError(f"Tick data file not found: {self.tick_file_path}")
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    async def run_all_strategies(self, enable_detailed_logging: bool = False) -> List[StrategyResult]:
        """
        Run all preset strategies on the same data
        
        Returns:
            List of StrategyResult objects with performance data
        """
        strategies = list_presets()
        results = []
        
        print(f"\n🚀 MULTI-STRATEGY TESTING")
        print(f"📁 File: {os.path.basename(self.tick_file_path)}")
        print(f"💰 Initial balance: ${self.initial_balance:,.2f} per strategy")
        print(f"⚡ Speed: {self.speed_multiplier}x")
        print(f"🎯 Testing {len(strategies)} strategies: {', '.join(strategies)}")
        print("="*70)
        
        # Run each strategy
        for i, strategy_name in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] Running {strategy_name}...")
            
            start_time = time.time()
            result = None
            error = None
            
            try:
                # Create simulation engine for this strategy
                simulation = FastSimulationEngine(
                    tick_file_path=self.tick_file_path,
                    speed_multiplier=self.speed_multiplier,
                    strategy_name=strategy_name
                )
                
                # Setup and run
                simulation.setup_trading_engine(initial_balance=self.initial_balance)
                result = await simulation.run_simulation(enable_detailed_logging=enable_detailed_logging)
                
                print(f"✅ {strategy_name} completed")
                
            except Exception as e:
                error = str(e)
                self.logger.error(f"❌ {strategy_name} failed: {e}")
                print(f"❌ {strategy_name} failed: {e}")
            
            processing_time = time.time() - start_time
            
            # Store result
            strategy_result = StrategyResult(
                strategy_name=strategy_name,
                config_name=strategy_name,
                results=result,
                processing_time=processing_time,
                error=error
            )
            results.append(strategy_result)
        
        return results
    
    def display_comparison(self, results: List[StrategyResult]):
        """Display side-by-side comparison of all strategy results"""
        print("\n" + "="*80)
        print("📊 MULTI-STRATEGY COMPARISON RESULTS")
        print("="*80)
        
        # Filter successful results
        successful_results = [r for r in results if r.error is None and r.results is not None]
        failed_results = [r for r in results if r.error is not None]
        
        if failed_results:
            print(f"\n❌ FAILED STRATEGIES ({len(failed_results)}):")
            for result in failed_results:
                print(f"   • {result.strategy_name}: {result.error}")
        
        if not successful_results:
            print("\n❌ No strategies completed successfully!")
            return
        
        print(f"\n✅ SUCCESSFUL STRATEGIES ({len(successful_results)}):")
        
        # Header
        print(f"\n{'Strategy':<12} {'ROI':<8} {'P&L':<12} {'Trades':<8} {'Win Rate':<10} {'Positions':<10}")
        print("-" * 70)
        
        # Sort by ROI descending
        sorted_results = sorted(successful_results, 
                              key=lambda r: self._get_roi(r.results), 
                              reverse=True)
        
        for result in sorted_results:
            performance = result.results.get('performance', {})
            trade_analysis = result.results.get('trade_analysis', {})
            position_closure = result.results.get('position_closure', {})
            
            # Calculate metrics
            roi = self._get_roi(result.results)
            total_pnl = performance.get('total_pnl', 0)
            total_trades = trade_analysis.get('total_trades', 0)
            win_rate = trade_analysis.get('win_rate_percent', 0)
            positions_opened = position_closure.get('positions_opened', 0)
            
            print(f"{result.strategy_name:<12} {roi:>+6.2f}% ${total_pnl:>+8.2f} {total_trades:>6} {win_rate:>8.1f}% {positions_opened:>8}")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED PERFORMANCE BREAKDOWN:")
        print("="*70)
        
        for result in sorted_results:
            self._display_strategy_summary(result)
        
        # Best performers
        self._display_best_performers(sorted_results)
    
    def _get_roi(self, results: Dict[str, Any]) -> float:
        """Calculate ROI percentage from results"""
        if not results:
            return 0.0
        
        performance = results.get('performance', {})
        initial = performance.get('initial_balance', 0)
        final = performance.get('final_balance', 0)
        
        if initial <= 0:
            return 0.0
        
        return ((final - initial) / initial) * 100
    
    def _display_strategy_summary(self, result: StrategyResult):
        """Display detailed summary for a single strategy"""
        config = get_preset_config(result.strategy_name)
        performance = result.results.get('performance', {})
        trade_analysis = result.results.get('trade_analysis', {})
        pattern_analysis = result.results.get('pattern_analysis', {})
        position_closure = result.results.get('position_closure', {})
        
        roi = self._get_roi(result.results)
        total_pnl = performance.get('total_pnl', 0)
        
        print(f"\n🎯 {result.strategy_name.upper()}")
        print(f"   📝 {config.description}")
        print(f"   ⚙️  Settings: {config.buy_confidence_threshold:.0%} confidence, {config.sell_after_candles} candles")
        print(f"   💰 P&L: ${total_pnl:+.2f} ({roi:+.2f}% ROI)")
        print(f"   📊 Trading: {trade_analysis.get('total_trades', 0)} trades, {trade_analysis.get('win_rate_percent', 0):.1f}% win rate")
        print(f"   🎯 Patterns: {pattern_analysis.get('total_patterns_detected', 0)} detected")
        print(f"   🔄 Positions: {position_closure.get('positions_opened', 0)} opened, {position_closure.get('positions_closed', 0)} closed")
        print(f"   ⏱️  Processing: {result.processing_time:.2f}s")
    
    def _display_best_performers(self, sorted_results: List[StrategyResult]):
        """Display best performing strategies by different metrics"""
        if not sorted_results:
            return
        
        print(f"\n🏆 BEST PERFORMERS:")
        print("-" * 40)
        
        # Best ROI
        best_roi = sorted_results[0]
        roi = self._get_roi(best_roi.results)
        print(f"💹 Best ROI: {best_roi.strategy_name} ({roi:+.2f}%)")
        
        # Most trades
        most_trades = max(sorted_results, 
                         key=lambda r: r.results.get('trade_analysis', {}).get('total_trades', 0))
        trades = most_trades.results.get('trade_analysis', {}).get('total_trades', 0)
        print(f"🔄 Most Active: {most_trades.strategy_name} ({trades} trades)")
        
        # Highest win rate
        best_winrate = max(sorted_results,
                          key=lambda r: r.results.get('trade_analysis', {}).get('win_rate_percent', 0))
        winrate = best_winrate.results.get('trade_analysis', {}).get('win_rate_percent', 0)
        print(f"🎯 Best Win Rate: {best_winrate.strategy_name} ({winrate:.1f}%)")
        
        # Most patterns detected
        most_patterns = max(sorted_results,
                           key=lambda r: r.results.get('pattern_analysis', {}).get('total_patterns_detected', 0))
        patterns = most_patterns.results.get('pattern_analysis', {}).get('total_patterns_detected', 0)
        print(f"🔍 Most Patterns: {most_patterns.strategy_name} ({patterns} detected)")
    
    def export_results_csv(self, results: List[StrategyResult], filename: str = None):
        """Export comparison results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_strategy_results_{timestamp}.csv"
        
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                writer.writerow([
                    'Strategy', 'ROI_%', 'Total_PnL', 'Realized_PnL', 'Unrealized_PnL',
                    'Total_Trades', 'Win_Rate_%', 'Winning_Trades', 'Losing_Trades',
                    'Patterns_Detected', 'Positions_Opened', 'Positions_Closed',
                    'Processing_Time_s', 'Error'
                ])
                
                # Data rows
                for result in results:
                    if result.error:
                        writer.writerow([
                            result.strategy_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            result.processing_time, result.error
                        ])
                    else:
                        performance = result.results.get('performance', {})
                        trade_analysis = result.results.get('trade_analysis', {})
                        pattern_analysis = result.results.get('pattern_analysis', {})
                        position_closure = result.results.get('position_closure', {})
                        
                        writer.writerow([
                            result.strategy_name,
                            self._get_roi(result.results),
                            performance.get('total_pnl', 0),
                            performance.get('realized_pnl', 0),
                            performance.get('unrealized_pnl', 0),
                            trade_analysis.get('total_trades', 0),
                            trade_analysis.get('win_rate_percent', 0),
                            trade_analysis.get('winning_trades', 0),
                            trade_analysis.get('losing_trades', 0),
                            pattern_analysis.get('total_patterns_detected', 0),
                            position_closure.get('positions_opened', 0),
                            position_closure.get('positions_closed', 0),
                            result.processing_time,
                            ''
                        ])
            
            print(f"\n📄 Results exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")


async def main():
    """Main CLI interface for multi-strategy testing"""
    import sys
    import argparse
    
    print("Multi-Strategy Trading Simulation Engine")
    print("="*50)
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run all trading strategies on the same tick data for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies that will be tested:
{chr(10).join(f'  {name}: {get_preset_config(name).description}' for name in list_presets())}

Examples:
  python multi_strategy_runner.py data/ticks.jsonl
  python multi_strategy_runner.py data/ticks.jsonl --balance 50000 --speed 100
  python multi_strategy_runner.py data/ticks.jsonl --export results.csv
        """
    )
    
    parser.add_argument('tick_file', help='Path to JSON Lines tick data file')
    parser.add_argument('--balance', '-b', type=float, default=100000,
                       help='Initial balance per strategy (default: $100,000)')
    parser.add_argument('--speed', '-x', type=int, default=1000,
                       help='Speed multiplier (default: 1000x)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable detailed logging')
    parser.add_argument('--export', '-e', type=str, 
                       help='Export results to CSV file (optional filename)')
    
    args = parser.parse_args()
    
    try:
        print(f"File: {args.tick_file}")
        print(f"Balance per strategy: ${args.balance:,.2f}")
        print(f"Speed: {args.speed}x")
        print(f"Strategies to test: {len(list_presets())}")
        
        # Create runner and execute
        runner = MultiStrategyRunner(
            tick_file_path=args.tick_file,
            initial_balance=args.balance,
            speed_multiplier=args.speed
        )
        
        # Run all strategies
        results = await runner.run_all_strategies(enable_detailed_logging=args.verbose)
        
        # Display comparison
        runner.display_comparison(results)
        
        # Export if requested
        if args.export:
            runner.export_results_csv(results, args.export if args.export.endswith('.csv') else f"{args.export}.csv")
        
        print(f"\n🎉 Multi-strategy testing complete!")
        print(f"📊 Tested {len([r for r in results if r.error is None])} successful strategies")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError running multi-strategy test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())