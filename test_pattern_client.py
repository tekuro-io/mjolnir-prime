#!/usr/bin/env python3
"""
Test Pattern Client
==================
Simple client to test receiving pattern broadcasts from the live detector.

Usage:
    python test_pattern_client.py
"""

import asyncio
import json
import websockets

async def pattern_client():
    """Connect to pattern broadcast server and listen for patterns"""
    uri = "ws://localhost:8765"
    
    print("🔗 Connecting to pattern broadcast server...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Listening for patterns...")
            print("-" * 50)
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "welcome":
                        print("👋 Welcome message received:")
                        print(f"   Monitoring: {', '.join(data.get('symbols', []))}")
                        print(f"   Patterns: {', '.join(data.get('patterns', []))}")
                        print(f"   Min Confidence: {data.get('min_confidence', 0) * 100:.1f}%")
                        print("-" * 50)
                        
                    elif data.get("type") == "pattern_detected":
                        pattern_data = data.get("data", {})
                        print("🎯 PATTERN DETECTED!")
                        print(f"   📈 Symbol: {pattern_data.get('symbol')}")
                        print(f"   🔍 Pattern: {pattern_data.get('pattern_type')}")
                        print(f"   📊 Confidence: {pattern_data.get('confidence', 0) * 100:.1f}%")
                        print(f"   💰 Price: ${pattern_data.get('trigger_price', 0):.2f}")
                        print(f"   ⏰ Time: {pattern_data.get('timestamp')}")
                        print(f"   📊 OHLCV: O${pattern_data.get('candle_data', {}).get('open', 0):.2f} "
                              f"H${pattern_data.get('candle_data', {}).get('high', 0):.2f} "
                              f"L${pattern_data.get('candle_data', {}).get('low', 0):.2f} "
                              f"C${pattern_data.get('candle_data', {}).get('close', 0):.2f} "
                              f"V{pattern_data.get('candle_data', {}).get('volume', 0):,}")
                        print("-" * 50)
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON received: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Make sure live_pattern_detector.py is running!")
    except KeyboardInterrupt:
        print("\n👋 Disconnected by user")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(pattern_client())