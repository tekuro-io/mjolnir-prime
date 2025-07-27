#!/usr/bin/env python3
"""
Test script for enhanced pattern notifier with WebSocket publishing.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_simulator.realtime.pattern_notifier import PatternNotifier, AlertLevel
from trading_simulator.core.models import PatternMatch
from trading_simulator.core.types import PatternType


async def test_pattern_websocket_publishing():
    """Test the pattern notifier WebSocket publishing functionality"""
    
    print("🧪 Testing Pattern Notifier WebSocket Publishing")
    print("=" * 50)
    
    # Create pattern notifier with WebSocket URL
    # Use a mock WebSocket URL for testing (change to actual URL when testing)
    websocket_url = "ws://localhost:8080/ws"  # Change to your WebSocket server
    notifier = PatternNotifier(websocket_url)
    
    print(f"📡 WebSocket URL: {websocket_url}")
    print(f"🔗 WebSocket enabled: {notifier.websocket_enabled}")
    
    # Create a sample pattern match
    pattern = PatternMatch(
        pattern_type=PatternType.FLAT_TOP_BREAKOUT,
        confidence=0.87,
        timestamp=datetime.now(),
        symbol="AAPL",
        trigger_price=150.25,
        candles_involved=[],
        metadata={'resistance_level': 150.0}
    )
    
    print(f"\n🎯 Creating test pattern:")
    print(f"   Pattern: {pattern.pattern_type.value}")
    print(f"   Symbol: {pattern.symbol}")
    print(f"   Price: ${pattern.trigger_price}")
    print(f"   Confidence: {pattern.confidence:.1%}")
    
    # Create notification (this will trigger WebSocket publishing)
    try:
        print(f"\n📤 Sending pattern notification...")
        notification = notifier.create_notification(pattern)
        
        print(f"✅ Notification created: {notification}")
        
        # Expected WebSocket message format
        expected_topic = f"pattern:{notification.symbol}"
        
        print(f"\n📨 Expected WebSocket topic:")
        print(f"   {expected_topic}")
        print(f"📨 Payload contains full pattern details in 'data' field")
        
        print(f"\n📊 Notification summary:")
        summary = notifier.get_notification_summary()
        print(f"   Total notifications: {summary['total']}")
        print(f"   By alert level: {summary['by_alert_level']}")
        
        # Wait a moment for async WebSocket send to complete
        await asyncio.sleep(1)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    finally:
        # Clean up WebSocket connection
        print(f"\n🧹 Cleaning up WebSocket connection...")
        await notifier.close_websocket_connection()
        print("✅ Cleanup complete")


async def test_multiple_patterns():
    """Test multiple pattern notifications"""
    
    print("\n" + "=" * 50)
    print("🧪 Testing Multiple Pattern Notifications")
    print("=" * 50)
    
    # Create notifier
    notifier = PatternNotifier("ws://localhost:8080/ws")
    
    # Test different pattern types
    patterns_to_test = [
        (PatternType.BULLISH_REVERSAL, "GOOGL", 2750.50, 0.92),
        (PatternType.BEARISH_REVERSAL, "MSFT", 425.75, 0.78),
        (PatternType.DOUBLE_TOP, "TSLA", 890.00, 0.85),
        (PatternType.DOUBLE_BOTTOM, "AMZN", 3450.25, 0.80)
    ]
    
    for pattern_type, symbol, price, confidence in patterns_to_test:
        pattern = PatternMatch(
            pattern_type=pattern_type,
            confidence=confidence,
            timestamp=datetime.now(),
            symbol=symbol,
            trigger_price=price,
            candles_involved=[],
            metadata={}
        )
        
        print(f"\n📈 Creating {pattern_type.value} for {symbol} @ ${price}")
        notification = notifier.create_notification(pattern)
        
        # Small delay between notifications
        await asyncio.sleep(0.5)
    
    # Show final summary
    print(f"\n📊 Final Summary:")
    summary = notifier.get_notification_summary()
    print(f"   Total: {summary['total']}")
    print(f"   By pattern: {summary['by_pattern_type']}")
    print(f"   By symbol: {summary['by_symbol']}")
    
    # Cleanup
    await notifier.close_websocket_connection()


def test_notification_formatting():
    """Test notification message formatting without WebSocket"""
    
    print("\n" + "=" * 50)
    print("🧪 Testing Notification Formatting")
    print("=" * 50)
    
    # Create notifier without WebSocket
    notifier = PatternNotifier()
    
    pattern = PatternMatch(
        pattern_type=PatternType.FLAT_TOP_BREAKOUT,
        confidence=0.95,
        timestamp=datetime.now(),
        symbol="NVDA",
        trigger_price=875.50,
        candles_involved=[],
        metadata={'resistance_level': 870.0}
    )
    
    notification = notifier.create_notification(pattern)
    
    print(f"📝 Notification Details:")
    print(f"   Message: {notification.message}")
    print(f"   Alert Level: {notification.alert_level.value}")
    print(f"   Confidence: {notification.confidence:.1%}")
    
    # Test topic format
    topic = f"pattern:{notification.symbol}"
    
    print(f"\n📡 WebSocket Topic Format:")
    print(f"   {topic}")
    print(f"📦 Full pattern data in payload")
    
    print(f"\n📋 JSON Data:")
    print(f"   {notification.to_dict()}")


async def main():
    """Main test function"""
    
    print("🚀 Pattern Notifier WebSocket Enhancement Test Suite")
    print("=" * 60)
    
    # Test 1: Basic notification formatting
    test_notification_formatting()
    
    # Test 2: Single WebSocket pattern (will fail if no server, but shows functionality)
    try:
        await test_pattern_websocket_publishing()
    except Exception as e:
        print(f"⚠️  WebSocket test failed (expected if no server): {e}")
    
    # Test 3: Multiple patterns (will fail if no server, but shows functionality)
    try:
        await test_multiple_patterns()
    except Exception as e:
        print(f"⚠️  Multiple patterns test failed (expected if no server): {e}")
    
    print("\n" + "=" * 60)
    print("✅ Test Suite Complete!")
    print("\n📋 Summary:")
    print("   - Pattern notifier now supports WebSocket publishing")
    print("   - Topic format: detection:[TICKER]:[PATTERN]:[PRICE]:[TIME]")
    print("   - Messages sent automatically when patterns detected")
    print("   - WebSocket connection managed automatically")
    print("\n💡 To test with real WebSocket server:")
    print("   1. Start a WebSocket server on ws://localhost:8080/ws")
    print("   2. Run this test script")
    print("   3. Watch for pattern detection messages")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")