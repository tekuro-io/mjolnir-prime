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

def aggregate_to_minute_candles(ticks, symbol):
    """Aggregate tick data into 1-minute candles"""
    candles = []
    current_candle = None
    
    for tick in ticks:
        timestamp = datetime.fromtimestamp(tick['timestamp'] / 1000)
        minute_timestamp = timestamp.replace(second=0, microsecond=0)
        price = tick['price']
        volume = tick['volume']
        
        if current_candle is None or current_candle['timestamp'] != minute_timestamp:
            # Complete previous candle
            if current_candle:
                candles.append(CandlestickTick(
                    symbol=symbol,
                    timestamp=current_candle['timestamp'],
                    open=current_candle['open'],
                    high=current_candle['high'],
                    low=current_candle['low'],
                    close=current_candle['close'],
                    volume=current_candle['volume']
                ))
            
            # Start new candle
            current_candle = {
                'timestamp': minute_timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update current candle
            current_candle['high'] = max(current_candle['high'], price)
            current_candle['low'] = min(current_candle['low'], price)
            current_candle['close'] = price
            current_candle['volume'] += volume
    
    # Complete final candle
    if current_candle:
        candles.append(CandlestickTick(
            symbol=symbol,
            timestamp=current_candle['timestamp'],
            open=current_candle['open'],
            high=current_candle['high'],
            low=current_candle['low'],
            close=current_candle['close'],
            volume=current_candle['volume']
        ))
    
    return candles

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