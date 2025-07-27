#!/usr/bin/env python3
"""
Test script for enhanced bearish reversal strategy with sell conditions.
"""

import sys
import os
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_simulator import create_demo_setup
from trading_simulator.patterns.strategies import BearishReversalSellStrategy, StrategyFactory
from trading_simulator.data.loaders import MockDataGenerator
from trading_simulator.core.models import PatternMatch
from trading_simulator.core.types import PatternType


def test_bearish_reversal_strategy():
    """Test the bearish reversal strategy with sell conditions"""
    
    print("🧪 Testing Enhanced Bearish Reversal Strategy")
    print("=" * 60)
    
    # Create trading engine
    engine, factory = create_demo_setup()
    
    # Create bearish reversal strategy
    bearish_strategy = BearishReversalSellStrategy(
        engine,
        confidence_threshold=0.65,
        quantity=100,
        stop_loss_pct=0.0005  # 0.05%
    )
    
    print(f"✅ Created bearish reversal strategy:")
    print(f"   Confidence threshold: {bearish_strategy.confidence_threshold:.1%}")
    print(f"   Quantity per trade: {bearish_strategy.quantity}")
    print(f"   Stop loss: {bearish_strategy.stop_loss_pct:.3%}")
    
    # Simulate a bearish reversal pattern
    print(f"\n🎯 Simulating bearish reversal pattern...")
    
    pattern = PatternMatch(
        pattern_type=PatternType.BEARISH_REVERSAL,
        confidence=0.75,
        timestamp=datetime.now(),
        symbol="TEST",
        trigger_price=100.00,
        candles_involved=[],
        metadata={'breakdown_strength': 0.8}
    )
    
    # Test pattern detection
    print(f"\n📉 Triggering bearish reversal detection...")
    bearish_strategy.on_pattern_detected(pattern)
    
    # Check active positions
    positions = bearish_strategy.get_active_positions()
    print(f"\n📊 Active positions: {len(positions)}")
    for symbol, pos in positions.items():
        print(f"   {symbol}: Entry ${pos['entry_price']:.4f}, Stop ${pos['stop_loss_price']:.4f}")
    
    # Test candle monitoring
    print(f"\n📈 Testing candle monitoring...")
    
    # Simulate first candle (green)
    from trading_simulator.core.models import CandlestickTick
    from trading_simulator.core.types import CandleType
    
    candle1 = CandlestickTick(
        symbol="TEST",
        timestamp=datetime.now(),
        open=100.00,
        high=101.50,
        low=99.50,
        close=101.00,  # Green candle
        volume=1000,
        candle_type=CandleType.BULLISH
    )
    
    print(f"   Candle 1: Open ${candle1.open} → Close ${candle1.close} (Green)")
    bearish_strategy.on_candle_completed(candle1)
    
    # Check if position still exists (should not sell on green candle)
    positions = bearish_strategy.get_active_positions()
    print(f"   Position still active: {'Yes' if positions else 'No'}")
    
    # Simulate second candle (red) - should trigger sell
    candle2 = CandlestickTick(
        symbol="TEST",
        timestamp=datetime.now(),
        open=101.00,
        high=101.20,
        low=99.80,
        close=100.20,  # Red candle
        volume=1200,
        candle_type=CandleType.BEARISH
    )
    
    print(f"   Candle 2: Open ${candle2.open} → Close ${candle2.close} (Red)")
    bearish_strategy.on_candle_completed(candle2)
    
    # Check positions after red candle
    positions = bearish_strategy.get_active_positions()
    print(f"   Position after red candle: {'Active' if positions else 'Closed'}")
    
    return bearish_strategy


def test_stop_loss_scenario():
    """Test stop loss trigger scenario"""
    
    print(f"\n" + "=" * 60)
    print("🧪 Testing Stop Loss Scenario")
    print("=" * 60)
    
    # Create new engine and strategy
    engine, factory = create_demo_setup()
    bearish_strategy = BearishReversalSellStrategy(
        engine,
        confidence_threshold=0.65,
        quantity=100,
        stop_loss_pct=0.0005  # 0.05%
    )
    
    # Create pattern at $100
    pattern = PatternMatch(
        pattern_type=PatternType.BEARISH_REVERSAL,
        confidence=0.80,
        timestamp=datetime.now(),
        symbol="STOP_TEST",
        trigger_price=100.00,
        candles_involved=[],
        metadata={}
    )
    
    print(f"📉 Entry at ${pattern.trigger_price:.4f}")
    bearish_strategy.on_pattern_detected(pattern)
    
    positions = bearish_strategy.get_active_positions()
    if positions:
        stop_price = positions["STOP_TEST"]["stop_loss_price"]
        print(f"📍 Stop loss set at ${stop_price:.4f}")
        
        # Test price above stop loss (should not sell)
        print(f"\n💹 Testing price above stop loss...")
        bearish_strategy.on_tick_received("STOP_TEST", 99.98)  # Above stop loss
        
        positions = bearish_strategy.get_active_positions()
        print(f"   Position status: {'Active' if positions else 'Closed'}")
        
        # Test price at stop loss (should sell)
        print(f"\n🚨 Testing stop loss trigger...")
        bearish_strategy.on_tick_received("STOP_TEST", stop_price - 0.001)  # Below stop loss
        
        positions = bearish_strategy.get_active_positions()
        print(f"   Position status: {'Active' if positions else 'Closed (Stop Loss)'}")


def test_two_candle_scenario():
    """Test selling after 2 candles regardless of color"""
    
    print(f"\n" + "=" * 60)
    print("🧪 Testing Two Candle Scenario")
    print("=" * 60)
    
    # Create new engine and strategy
    engine, factory = create_demo_setup()
    bearish_strategy = BearishReversalSellStrategy(
        engine,
        confidence_threshold=0.65,
        quantity=100,
        stop_loss_pct=0.0005
    )
    
    # Create pattern
    pattern = PatternMatch(
        pattern_type=PatternType.BEARISH_REVERSAL,
        confidence=0.70,
        timestamp=datetime.now(),
        symbol="TWO_CANDLE",
        trigger_price=50.00,
        candles_involved=[],
        metadata={}
    )
    
    print(f"📉 Entry at ${pattern.trigger_price:.4f}")
    bearish_strategy.on_pattern_detected(pattern)
    
    from trading_simulator.core.models import CandlestickTick
    from trading_simulator.core.types import CandleType
    
    # First candle (green)
    candle1 = CandlestickTick(
        symbol="TWO_CANDLE",
        timestamp=datetime.now(),
        open=50.00,
        high=50.50,
        low=49.80,
        close=50.30,  # Green
        volume=1000,
        candle_type=CandleType.BULLISH
    )
    
    print(f"📈 Candle 1: ${candle1.open} → ${candle1.close} (Green)")
    bearish_strategy.on_candle_completed(candle1)
    
    positions = bearish_strategy.get_active_positions()
    print(f"   Position after candle 1: {'Active' if positions else 'Closed'}")
    
    # Second candle (also green) - should still sell after 2 candles
    candle2 = CandlestickTick(
        symbol="TWO_CANDLE",
        timestamp=datetime.now(),
        open=50.30,
        high=50.80,
        low=50.10,
        close=50.60,  # Green again
        volume=1200,
        candle_type=CandleType.BULLISH
    )
    
    print(f"📈 Candle 2: ${candle2.open} → ${candle2.close} (Green)")
    bearish_strategy.on_candle_completed(candle2)
    
    positions = bearish_strategy.get_active_positions()
    print(f"   Position after candle 2: {'Active' if positions else 'Closed (2 candles elapsed)'}")


def test_with_strategy_factory():
    """Test integration with StrategyFactory"""
    
    print(f"\n" + "=" * 60)
    print("🧪 Testing Strategy Factory Integration")
    print("=" * 60)
    
    # Create engine and factory
    engine, original_factory = create_demo_setup()
    factory = StrategyFactory(engine)
    
    # Create balanced setup which includes bearish reversal strategy
    print("🏭 Creating balanced setup with bearish reversal strategy...")
    factory.create_balanced_setup()
    
    # Get strategy info
    info = factory.get_strategy_info()
    print(f"✅ Strategy factory created {info['strategy_count']} strategies:")
    for strategy_name in info['active_strategies']:
        print(f"   - {strategy_name}")
    
    # Show portfolio initial state
    summary = engine.get_portfolio_summary()
    print(f"\n💰 Initial Portfolio:")
    print(f"   Cash: ${summary['cash']:,.2f}")
    print(f"   Total Value: ${summary['total_value']:,.2f}")


def main():
    """Main test function"""
    
    print("🚀 Enhanced Bearish Reversal Strategy Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Basic strategy functionality
        strategy1 = test_bearish_reversal_strategy()
        
        # Test 2: Stop loss scenario
        test_stop_loss_scenario()
        
        # Test 3: Two candle scenario
        test_two_candle_scenario()
        
        # Test 4: Strategy factory integration
        test_with_strategy_factory()
        
        print(f"\n" + "=" * 70)
        print("✅ All Tests Complete!")
        print("\n📋 Summary of Bearish Reversal Sell Strategy:")
        print("   ✓ Buys on bearish reversal patterns (anticipating bounce)")
        print("   ✓ Sells after 2 candles OR 1 red candle OR 0.05% stop loss")
        print("   ✓ Tracks positions with P&L calculation")
        print("   ✓ Integrates with Strategy Factory")
        print("   ✓ Monitors real-time ticks for stop losses")
        
        print(f"\n💡 To use in real trading:")
        print("   1. Create StrategyFactory with your engine")
        print("   2. Call factory.create_balanced_setup()")
        print("   3. The bearish reversal strategy will auto-activate")
        print("   4. Monitor logs for buy/sell signals")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")