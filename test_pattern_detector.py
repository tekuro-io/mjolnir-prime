#!/usr/bin/env python3
"""
Simple test script for bull flag pattern detection using test data.
"""

import json
import sys
from datetime import datetime
from trading_simulator.core.models import CandlestickTick
from trading_simulator.patterns.detector import PatternDetector

def load_test_data():
    """Load test data from JSON file"""
    try:
        with open('test_bull_flag_data.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("ERROR: test_bull_flag_data.json not found")
        return None

def tick_to_candle(tick_data, symbol):
    """Convert tick data to CandlestickTick"""
    return CandlestickTick(
        symbol=symbol,
        timestamp=datetime.fromtimestamp(tick_data['timestamp'] / 1000),
        open=tick_data['price'],
        high=tick_data['price'], 
        low=tick_data['price'],
        close=tick_data['price'],
        volume=tick_data['volume']
    )

def create_realistic_bull_flag_candles(symbol):
    """Create realistic bull flag pattern with proper OHLC candles"""
    import random
    from datetime import datetime, timedelta
    
    candles = []
    base_time = datetime(2023, 12, 28, 11, 0)  # Start at 11:00 AM
    
    # Phase 1: Flagpole (Strong upward move) - 8 candles
    current_price = 150.00
    print("Creating flagpole candles (strong upward move)...")
    for i in range(8):
        # Generate bullish candle
        open_price = current_price
        
        # Strong upward move - each candle gains 1.5-2.5%
        gain = random.uniform(0.015, 0.025)
        close_price = open_price * (1 + gain)
        
        # Create realistic OHLC
        low_price = open_price - random.uniform(0, open_price * 0.005)  # Small wick down
        high_price = close_price + random.uniform(0, close_price * 0.003)  # Small wick up
        
        volume = random.randint(300, 500)  # High volume during flagpole
        
        candle = CandlestickTick(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume
        )
        candles.append(candle)
        current_price = close_price
        print(f"   Flagpole candle {i+1}: O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} [{candle.candle_type.value}]")
    
    # Phase 2: Flag (Consolidation/pullback) - 12 candles
    flag_start_price = current_price
    flag_low = flag_start_price * 0.96  # 4% pullback
    print(f"\nCreating flag candles (consolidation from {flag_start_price:.2f} to ~{flag_low:.2f})...")
    
    for i in range(12):
        open_price = current_price
        
        if i < 6:  # First half: pullback
            close_change = random.uniform(-0.008, -0.002)  # 0.2-0.8% decline
        else:  # Second half: consolidation
            close_change = random.uniform(-0.003, 0.003)  # Small moves both ways
            
        close_price = max(flag_low, open_price * (1 + close_change))
        
        # Create OHLC based on candle direction
        if close_price > open_price:  # Bullish candle
            low_price = open_price - random.uniform(0, open_price * 0.003)
            high_price = close_price + random.uniform(0, close_price * 0.002)
        else:  # Bearish candle
            high_price = open_price + random.uniform(0, open_price * 0.002)
            low_price = close_price - random.uniform(0, close_price * 0.003)
        
        volume = random.randint(80, 150)  # Lower volume during flag
        
        candle = CandlestickTick(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=8 + i),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume
        )
        candles.append(candle)
        current_price = close_price
        print(f"   Flag candle {i+1}: O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} [{candle.candle_type.value}]")
    
    # Phase 3: Pre-breakout setup - 5 more consolidation candles
    print(f"\nCreating pre-breakout candles (tight consolidation)...")
    for i in range(5):
        open_price = current_price
        close_change = random.uniform(-0.002, 0.004)  # Slight bias upward
        close_price = open_price * (1 + close_change)
        
        if close_price > open_price:
            low_price = open_price - random.uniform(0, open_price * 0.001)
            high_price = close_price + random.uniform(0, close_price * 0.001)
        else:
            high_price = open_price + random.uniform(0, open_price * 0.001)
            low_price = close_price - random.uniform(0, close_price * 0.001)
        
        volume = random.randint(90, 130)
        
        candle = CandlestickTick(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=20 + i),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume
        )
        candles.append(candle)
        current_price = close_price
        print(f"   Pre-breakout candle {i+1}: O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} [{candle.candle_type.value}]")
    
    return candles

def aggregate_to_minute_candles(ticks, symbol):
    """Use realistic bull flag pattern instead of tick aggregation"""
    return create_realistic_bull_flag_candles(symbol)

def main():
    print("Testing Bull Flag Pattern Detection")
    print("=" * 50)
    
    # Load test data
    data = load_test_data()
    if not data:
        return
    
    print(f"Loaded {len(data['ticks'])} ticks for {data['symbol']}")
    print(f"Expected pattern: {data['description']}")
    print()
    
    # Initialize pattern detector
    detector = PatternDetector(tolerance=0.01, lookback_window=50)
    patterns_found = []
    
    def pattern_callback(pattern):
        patterns_found.append(pattern)
        print(f"*** PATTERN DETECTED! ***")
        print(f"   Symbol: {pattern.symbol}")
        print(f"   Type: {pattern.pattern_type.value}")
        print(f"   Confidence: {pattern.confidence:.1%}")
        print(f"   Trigger Price: ${pattern.trigger_price:.2f}")
        print(f"   Time: {pattern.timestamp.strftime('%H:%M:%S')}")
        if pattern.metadata:
            print(f"   Metadata: {pattern.metadata}")
        print()
    
    # Register callbacks for all pattern types
    from trading_simulator.core.types import PatternType
    for pattern_type in [PatternType.BULL_FLAG, PatternType.BEAR_FLAG, 
                        PatternType.BULLISH_REVERSAL, PatternType.BEARISH_REVERSAL]:
        detector.register_pattern_callback(pattern_type, pattern_callback)
    
    # Convert ticks to 1-minute candles
    candles = aggregate_to_minute_candles(data['ticks'], data['symbol'])
    print(f"Aggregated into {len(candles)} 1-minute candles")
    print()
    
    # Process candles through pattern detector
    print("Processing candles for pattern detection...")
    for i, candle in enumerate(candles):
        print(f"   Candle {i+1:2d}: {candle.timestamp.strftime('%H:%M')} "
              f"O:{candle.open:6.2f} H:{candle.high:6.2f} L:{candle.low:6.2f} "
              f"C:{candle.close:6.2f} V:{candle.volume:4d} [{candle.candle_type.value}]")
        
        # Add candle to detector (simulates live streaming - every candle triggers pattern detection)
        detected = detector.add_candle(candle)
        # Patterns are handled via callbacks automatically
        
        # Debug: check if we have enough candles for bull flag detection
        if i >= 10:  # After enough candles
            print(f"      [DEBUG] Can detect bull flag with {len(detector.candle_history.get(candle.symbol, []))} candles")
            
        # Special debug for the expected pattern area (around candle 50+ for breakout)
        if i >= 45:
            recent_candles = candles[max(0, i-10):i+1]
            price_change = ((recent_candles[-1].close - recent_candles[0].open) / recent_candles[0].open) * 100
            print(f"      [DEBUG] Last 10 candles price change: {price_change:.2f}%")
    
    print()
    print("=" * 50)
    print(f"Analysis Complete - {len(patterns_found)} patterns detected")
    
    if patterns_found:
        print("\nSummary of detected patterns:")
        for i, pattern in enumerate(patterns_found, 1):
            print(f"   {i}. {pattern.pattern_type.value} at ${pattern.trigger_price:.2f} ({pattern.confidence:.1%})")
    else:
        print("No patterns detected in test data")

if __name__ == "__main__":
    main()