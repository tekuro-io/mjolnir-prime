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
    
    # Use smaller quantities for the demo
    flat_top_strategy = PatternStrategies.create_flat_top_strategy(
        engine, confidence_threshold=0.7, quantity=50, max_risk_per_trade=0.02
    )
    reversal_strategy = PatternStrategies.create_reversal_strategy(
        engine, confidence_threshold=0.65, quantity=25  # Smaller quantity for expensive stocks
    )
    
    strategy_factory.add_strategy("demo_flat_top", flat_top_strategy, PatternType.FLAT_TOP_BREAKOUT)
    strategy_factory.add_strategy("demo_reversal", reversal_strategy, PatternType.BULLISH_REVERSAL)
    
    print(f"Active strategies: {list(strategy_factory.active_strategies.keys())}")
    
    # Generate some test data that will trigger patterns
    ticks = [
        # AAPL flat top breakout pattern
        CandlestickTick('AAPL', datetime.now(), 150.0, 151.0, 149.5, 150.8, 100000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=1), 150.8, 151.2, 150.0, 150.9, 95000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=2), 150.9, 151.1, 150.1, 150.9, 90000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=3), 150.9, 152.5, 150.8, 152.2, 150000),
        
        # GOOGL bullish reversal pattern (use lower prices)
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=4), 280.0, 282.0, 279.5, 281.5, 50000),
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=5), 281.5, 281.8, 279.0, 279.5, 60000),
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=6), 279.5, 280.0, 278.0, 278.5, 55000),
        CandlestickTick('GOOGL', datetime.now() + timedelta(minutes=7), 278.5, 282.5, 278.3, 282.0, 80000),
    ]
    
    print(f"\nProcessing {len(ticks)} candlestick ticks...")
    
    for i, tick in enumerate(ticks):
        print(f"\n--- Tick {i+1}: {tick.symbol} @ ${tick.close:.2f} ({tick.candle_type.value}) ---")
        engine.process_tick(tick)
    
    # Display results
    summary = engine.get_portfolio_summary()
    print("\n=== Results ===")
    print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"Cash: ${summary['cash']:,.2f}")
    
    if summary['positions']:
        print("Positions:")
        for symbol, pos in summary['positions'].items():
            print(f"  {symbol}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")
            print(f"    Current Price: ${pos['current_price']:.2f}")
            print(f"    Market Value: ${pos['market_value']:,.2f}")
            print(f"    Cost Basis: ${pos['cost_basis']:,.2f}")
            print(f"    Unrealized P&L: ${pos['unrealized_pnl']:,.2f} ({pos['pnl_percent']:.2f}%)")
    else:
        print("No positions held")
    
    print(f"\n--- P&L Summary ---")
    print(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl']/summary['initial_balance']*100:.2f}%)")
    print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    print(f"Realized P&L: ${summary['realized_pnl']:,.2f}")
    print(f"Patterns Detected: {summary['patterns_detected']}")
    print(f"Orders Executed: {summary['executed_orders']}")
    
    return engine


def demo_algorithm_framework():
    """Demonstrate the algorithm framework"""
    print("\n=== Algorithm Framework Demo ===")
    
    engine = create_demo_engine()
    algorithm_manager = AlgorithmManager(engine)
    
    # Create algorithms that match the pattern types we'll generate
    # Use a reversal algorithm since that's what we're generating
    from trading_simulator.algorithms.base import PatternBasedAlgorithm
    
    class CustomReversalAlgorithm(PatternBasedAlgorithm):
        def __init__(self, engine, quantity=30):
            super().__init__(
                name="ReversalTrader",
                target_patterns=[PatternType.BULLISH_REVERSAL],
                confidence_threshold=0.7
            )
            self.engine = engine
            self.quantity = quantity
        
        def on_pattern(self, pattern):
            print(f"🔄 {self.name}: {pattern.pattern_type.value} detected!")
            print(f"   Confidence: {pattern.confidence:.2f}")
            try:
                order_id = self.engine.place_market_order(
                    pattern.symbol, OrderSide.BUY, self.quantity
                )
                print(f"   Order placed: {order_id}")
            except Exception as e:
                print(f"   Error: {e}")
    
    # Add the reversal algorithm
    reversal_algo = CustomReversalAlgorithm(engine, quantity=30)
    algorithm_manager.add_algorithm(reversal_algo)
    
    print(f"Active algorithms: {algorithm_manager.get_active_algorithms()}")
    
    # Generate reversal pattern data (matching what the algorithm expects)
    print("\nGenerating bullish reversal pattern...")
    pattern_data = [
        CandlestickTick('AAPL', datetime.now(), 150.0, 152.0, 149.5, 151.5, 60000),  # Bullish
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=1), 151.5, 151.8, 149.0, 149.5, 65000),  # Bearish
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=2), 149.5, 150.0, 148.0, 148.5, 60000),  # More bearish
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=3), 148.5, 152.5, 148.3, 152.0, 85000),  # Bullish confirmation
    ]
    
    print(f"Processing {len(pattern_data)} pattern candlesticks...")
    for i, tick in enumerate(pattern_data):
        print(f"Tick {i+1}: {tick.symbol} @ ${tick.close:.2f} ({tick.candle_type.value})")
        engine.process_tick(tick)
    
    # Show algorithm status
    status = algorithm_manager.get_algorithm_status()
    print(f"\nAlgorithm Status:")
    for name, info in status.items():
        print(f"  {name}: {info['trades_made']} trades, Active: {info['is_active']}")
    
    # Show results
    summary = engine.get_portfolio_summary()
    print(f"\nResults: Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"Trades Executed: {summary['executed_orders']}")
    
    return engine, algorithm_manager


def demo_backtesting():
    """Demonstrate backtesting functionality"""
    print("\n=== Backtesting Demo ===")
    
    engine = create_demo_engine()
    backtest_runner = BacktestRunner(engine)
    
    # Create algorithm that matches the patterns we'll generate
    from trading_simulator.algorithms.base import PatternBasedAlgorithm
    
    class DemoAlgorithm(PatternBasedAlgorithm):
        def __init__(self, engine, quantity=40):
            super().__init__(
                name="DemoTrader",
                target_patterns=[PatternType.FLAT_TOP_BREAKOUT, PatternType.BULLISH_REVERSAL],
                confidence_threshold=0.6  # Lower threshold for demo
            )
            self.engine = engine
            self.quantity = quantity
        
        def on_pattern(self, pattern):
            print(f"💰 Trading on {pattern.pattern_type.value} (confidence: {pattern.confidence:.2f})")
            try:
                order_id = self.engine.place_market_order(
                    pattern.symbol, OrderSide.BUY, self.quantity
                )
                print(f"   Order executed: {order_id}")
            except Exception as e:
                print(f"   Trade failed: {e}")
    
    demo_algo = DemoAlgorithm(engine, quantity=40)
    
    # Generate simple, clear test data
    print("Generating test data with guaranteed patterns...")
    test_data = []
    
    # Simple flat-top breakout (only 4 candles)
    print("Creating flat-top breakout pattern...")
    test_data = [
        CandlestickTick('AAPL', datetime.now(), 150.0, 150.9, 149.5, 150.8, 50000),
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=1), 150.8, 150.9, 150.0, 150.85, 45000),  # Flat top 1
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=2), 150.85, 150.9, 150.1, 150.87, 40000),  # Flat top 2  
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=3), 150.87, 152.5, 150.8, 152.2, 80000),  # Breakout!
        
        # Simple bullish reversal (4 candles)
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=4), 152.2, 154.0, 151.0, 153.5, 60000),  # Bullish
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=5), 153.5, 153.8, 152.0, 152.5, 65000),  # Bearish
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=6), 152.5, 153.0, 151.0, 151.5, 60000),  # More bearish
        CandlestickTick('AAPL', datetime.now() + timedelta(minutes=7), 151.5, 155.5, 151.3, 155.0, 85000),  # Bullish confirmation
    ]
    
    print(f"Running backtest with {len(test_data)} candles...")
    print("Expected patterns: 1 flat_top_breakout + 1 bullish_reversal = 2 trades")
    
    results = backtest_runner.run_backtest(demo_algo, test_data)
    
    print(f"\n=== Backtest Results ===")
    print(f"Algorithm: {results['algorithm_name']}")
    print(f"Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Patterns Detected: {results['patterns_detected']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    # Show which patterns were detected
    if hasattr(backtest_runner.engine, 'detected_patterns') and backtest_runner.engine.detected_patterns:
        print(f"\nPattern Types Detected:")
        pattern_types = {}
        for pattern in backtest_runner.engine.detected_patterns:
            pattern_type = pattern.pattern_type.value
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = 0
            pattern_types[pattern_type] += 1
        
        for pattern_type, count in pattern_types.items():
            confidence_avg = sum(p.confidence for p in backtest_runner.engine.detected_patterns 
                               if p.pattern_type.value == pattern_type) / count
            print(f"  - {pattern_type}: {count} (avg confidence: {confidence_avg:.2f})")
    
    # Debug: Show final portfolio state
    final_summary = engine.get_portfolio_summary()
    if final_summary['positions']:
        print("\nFinal Positions:")
        for symbol, pos in final_summary['positions'].items():
            print(f"  {symbol}: {pos['quantity']} shares")
    else:
        print("\nNo final positions (this might indicate a bug)")
    
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