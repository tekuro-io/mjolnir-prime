"""
Basic demonstration of the trading simulator.
"""

from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_simulator.portfolio.portfolio import Portfolio
from trading_simulator.patterns.detector import PatternDetector
from trading_simulator.trading.engine import TradeEngine
from trading_simulator.patterns.strategies import PatternStrategies, StrategyFactory
from trading_simulator.core.models import CandlestickTick
from trading_simulator.core.types import PatternType, OrderSide
from trading_simulator.data.loaders import MockDataGenerator
from trading_simulator.algorithms.base import SimpleBreakoutAlgorithm, AlgorithmManager, BacktestRunner


def create_demo_engine():
    """Factory function to create a demo trading engine"""
    portfolio = Portfolio(initial_balance=100000.0)
    pattern_detector = PatternDetector(tolerance=0.01)
    engine = TradeEngine(portfolio, pattern_detector, ['AAPL', 'GOOGL', 'MSFT'])
    
    return engine


def demo_basic_trading():
    """Demonstrate basic trading functionality"""
    print("=== Basic Trading Demo ===")
    
    engine = create_demo_engine()
    
    # Create strategy factory and add strategies
    strategy_factory = StrategyFactory(engine)
    strategy_factory.create_balanced_setup()
    
    print(f"Active strategies: {strategy_factory.list_strategies()}")
    
    # Generate some test data that will trigger patterns
    ticks = [
        # AAPL flat top breakout pattern
        CandlestickTick('AAPL', datetime.now(), 150.0, 151.0, 149.5, 150.8, 100000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=1), 150.8, 151.2, 150.0, 150.9, 95000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=2), 150.9, 151.1, 150.1, 150.9, 90000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=3), 150.9, 152.5, 150.8, 152.2, 150000),
        
        # GOOGL bullish reversal pattern
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=4), 2800.0, 2820.0, 2795.0, 2815.0, 50000),
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=5), 2815.0, 2818.0, 2790.0, 2795.0, 60000),
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=6), 2795.0, 2800.0, 2780.0, 2785.0, 55000),
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=7), 2785.0, 2825.0, 2783.0, 2820.0, 80000),
    ]
    
    print(f"\nProcessing {len(ticks)} candlestick ticks...")
    
    for i, tick in enumerate(ticks):
        print(f"\n--- Tick {i+1}: {tick.symbol} @ ${tick.close:.2f} ({tick.candle_type.value}) ---")
        engine.process_tick(tick)
    
    # Display results
    summary = engine.get_portfolio_summary()
    print("\n=== Results ===")
    print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl']/summary['initial_balance']*100:.2f}%)")
    print(f"Patterns Detected: {summary['patterns_detected']}")
    print(f"Orders Executed: {summary['executed_orders']}")
    
    if summary['positions']:
        print("\nPositions:")
        for symbol, pos in summary['positions'].items():
            print(f"  {symbol}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")
    
    return engine


def demo_algorithm_framework():
    """Demonstrate the algorithm framework"""
    print("\n=== Algorithm Framework Demo ===")
    
    engine = create_demo_engine()
    algorithm_manager = AlgorithmManager(engine)
    
    # Create and add algorithms
    breakout_algo = SimpleBreakoutAlgorithm(engine, quantity=50)
    algorithm_manager.add_algorithm(breakout_algo)
    
    print(f"Active algorithms: {algorithm_manager.get_active_algorithms()}")
    
    # Generate pattern data
    pattern_data = MockDataGenerator.generate_pattern_data('AAPL', 'flat_top_breakout')
    
    print(f"\nProcessing {len(pattern_data)} pattern candlesticks...")
    for tick in pattern_data:
        engine.process_tick(tick)
    
    # Show algorithm status
    status = algorithm_manager.get_algorithm_status()
    print(f"\nAlgorithm Status:")
    for name, info in status.items():
        print(f"  {name}: {info['trades_made']} trades, Active: {info['is_active']}")
    
    return engine, algorithm_manager


def demo_backtesting():
    """Demonstrate backtesting functionality"""
    print("\n=== Backtesting Demo ===")
    
    engine = create_demo_engine()
    backtest_runner = BacktestRunner(engine)
    
    # Create test algorithms
    breakout_algo = SimpleBreakoutAlgorithm(engine, quantity=100)
    
    # Generate test data
    print("Generating test data...")
    test_data = MockDataGenerator.generate_trending_data(
        'AAPL', start_price=150.0, num_candles=50, trend=0.002, volatility=0.015
    )
    
    # Add some pattern data
    pattern_data = MockDataGenerator.generate_pattern_data('AAPL', 'flat_top_breakout')
    test_data.extend(pattern_data)
    
    # Sort by timestamp
    test_data.sort(key=lambda x: x.timestamp)
    
    print(f"Running backtest with {len(test_data)} candles...")
    results = backtest_runner.run_backtest(breakout_algo, test_data)
    
    print(f"\n=== Backtest Results ===")
    print(f"Algorithm: {results['algorithm_name']}")
    print(f"Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Patterns Detected: {results['patterns_detected']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    return results


def demo_data_loading():
    """Demonstrate data loading capabilities"""
    print("\n=== Data Loading Demo ===")
    
    # Generate and validate mock data
    print("Generating mock trending data...")
    mock_data = MockDataGenerator.generate_trending_data(
        'MSFT', start_price=300.0, num_candles=20, trend=0.001, volatility=0.01
    )
    
    from trading_simulator.data.loaders import DataValidator
    
    # Validate the data
    issues = DataValidator.validate_candlesticks(mock_data)
    
    if issues:
        print(f"Data validation issues found: {len(issues)}")
        for issue in issues[:3]:  # Show first 3 issues
            print(f"  - {issue}")
    else:
        print("✅ All data validation checks passed!")
    
    # Show data statistics
    if mock_data:
        prices = [candle.close for candle in mock_data]
        print(f"\nData Statistics:")
        print(f"  Candles: {len(mock_data)}")
        print(f"  Price Range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"  Total Return: {((prices[-1] - prices[0]) / prices[0] * 100):.2f}%")
        print(f"  Average Volume: {sum(c.volume for c in mock_data) / len(mock_data):,.0f}")
    
    return mock_data


def comprehensive_demo():
    """Run a comprehensive demonstration"""
    print("🚀 Trading Simulator Comprehensive Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        engine1 = demo_basic_trading()
        engine2, algo_manager = demo_algorithm_framework()
        backtest_results = demo_backtesting()
        mock_data = demo_data_loading()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        print("\nThe trading simulator demonstrates:")
        print("  ✓ Pattern detection and strategy execution")
        print("  ✓ Portfolio management and risk tracking")
        print("  ✓ Algorithm framework with backtesting")
        print("  ✓ Data loading and validation")
        print("  ✓ Modular, extensible architecture")
        
        return {
            'basic_engine': engine1,
            'algorithm_engine': engine2,
            'algorithm_manager': algo_manager,
            'backtest_results': backtest_results,
            'mock_data': mock_data
        }
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Check if specific demo is requested
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        
        if demo_name == 'basic':
            demo_basic_trading()
        elif demo_name == 'algorithms':
            demo_algorithm_framework()
        elif demo_name == 'backtest':
            demo_backtesting()
        elif demo_name == 'data':
            demo_data_loading()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Available demos: basic, algorithms, backtest, data")
    else:
        # Run comprehensive demo
        comprehensive_demo()