#!/usr/bin/env python3
"""
Live Pattern Detection Tool
===========================
Real-time pattern detection using Polygon.io API data.

Usage:
    python live_pattern_detector.py

API Key Configuration:
    Set your Polygon API key in config/live_config.py
"""


import asyncio
import json
import logging
import os
import sys
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import time
from websockets.server import serve

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_simulator.core.models import CandlestickTick, PatternType
from trading_simulator.patterns.detector import PatternDetector, PatternMatch


@dataclass
class LiveConfig:
    """Configuration for live pattern detection"""
    api_key: str = ""
    symbols: List[str] = None
    pattern_types: List[PatternType] = None
    min_confidence: float = 0.65
    candle_timeframe: str = "1min"  # 1min, 5min, 15min, etc.
    max_lookback_candles: int = 100
    
    # Websocket broadcasting settings
    broadcast_enabled: bool = True
    broadcast_host: str = "localhost"
    broadcast_port: int = 8765
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 'UBER', 'ROKU']
        if self.pattern_types is None:
            self.pattern_types = [PatternType.BULLISH_REVERSAL, PatternType.BEARISH_REVERSAL, 
                                PatternType.FLAT_TOP_BREAKOUT, PatternType.DOUBLE_TOP,
                                PatternType.DOUBLE_BOTTOM, PatternType.BULL_FLAG, 
                                PatternType.BEAR_FLAG]


class LivePatternDetector:
    """Real-time pattern detection using Polygon.io websocket data"""
    
    def __init__(self, config: LiveConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.pattern_detector = PatternDetector()
        
        # Data storage
        self.candle_history: Dict[str, List[CandlestickTick]] = {symbol: [] for symbol in config.symbols}
        self.current_candles: Dict[str, Dict] = {}  # Current building candles
        self.detected_patterns: List[PatternMatch] = []
        
        # Connection state
        self.websocket = None
        self.is_running = False
        
        # Broadcasting state
        self.broadcast_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.broadcast_server = None
        
        # Register pattern callbacks
        for pattern_type in config.pattern_types:
            self.pattern_detector.register_pattern_callback(pattern_type, self._on_pattern_detected)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('live_patterns.log')
            ]
        )
        return logging.getLogger(__name__)
    
    async def start_broadcast_server(self):
        """Start websocket server for broadcasting patterns"""
        if not self.config.broadcast_enabled:
            return
            
        try:
            self.broadcast_server = await serve(
                self._handle_broadcast_client,
                self.config.broadcast_host,
                self.config.broadcast_port
            )
            self.logger.info(f"[BROADCAST] Pattern broadcast server started on ws://{self.config.broadcast_host}:{self.config.broadcast_port}")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to start broadcast server: {e}")
    
    async def _handle_broadcast_client(self, websocket, path):
        """Handle new websocket client connections"""
        self.broadcast_clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.logger.info(f"[CLIENT] New client connected: {client_info}")
        
        try:
            # Send welcome message
            welcome_msg = {
                "type": "welcome",
                "message": "Connected to Live Pattern Detector",
                "symbols": self.config.symbols,
                "patterns": [p.value for p in self.config.pattern_types],
                "min_confidence": self.config.min_confidence,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Keep connection alive
            await websocket.wait_closed()
            
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error with client {client_info}: {e}")
        finally:
            self.broadcast_clients.discard(websocket)
            self.logger.info(f"[CLIENT] Client disconnected: {client_info}")
    
    async def _broadcast_pattern(self, pattern: PatternMatch):
        """Broadcast detected pattern to all connected clients"""
        if not self.config.broadcast_enabled or not self.broadcast_clients:
            return
        
        # Create pattern message in standard format (same as k8s_pattern_detector)
        # Get the trigger candle for candle_data
        trigger_candle = None
        if pattern.candles_involved:
            trigger_candle = pattern.candles_involved[-1]  # Use last candle as trigger
        
        pattern_msg = {
            "type": "pattern_detected",
            "data": {
                "symbol": pattern.symbol,
                "pattern_type": pattern.pattern_type.value,
                "confidence": round(pattern.confidence, 4),
                "trigger_price": round(pattern.trigger_price, 2),
                "timestamp": pattern.timestamp.isoformat(),
                "candle_data": {
                    "open": round(trigger_candle.open if trigger_candle else pattern.trigger_price, 2),
                    "high": round(trigger_candle.high if trigger_candle else pattern.trigger_price, 2),
                    "low": round(trigger_candle.low if trigger_candle else pattern.trigger_price, 2),
                    "close": round(trigger_candle.close if trigger_candle else pattern.trigger_price, 2),
                    "volume": trigger_candle.volume if trigger_candle else 0
                },
                "source": "live_polygon_data"
            }
        }
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.broadcast_clients.copy():
            try:
                await client.send(json.dumps(pattern_msg))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(client)
        
        # Clean up disconnected clients
        self.broadcast_clients -= disconnected_clients
        
        if self.broadcast_clients:
            self.logger.info(f"[BROADCAST] Pattern broadcasted to {len(self.broadcast_clients)} clients")
    
    async def stop_broadcast_server(self):
        """Stop the broadcast server"""
        if self.broadcast_server:
            self.broadcast_server.close()
            await self.broadcast_server.wait_closed()
            self.logger.info("[BROADCAST] Broadcast server stopped")
    
    async def connect(self):
        """Connect to Polygon.io websocket"""
        if not self.config.api_key:
            raise ValueError("API key required. Set POLYGON_API_KEY environment variable or update config.")
        
        # Polygon.io websocket URL
        uri = "wss://socket.polygon.io/stocks"
        
        try:
            self.logger.info("Connecting to Polygon.io websocket...")
            self.websocket = await websockets.connect(uri)
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.config.api_key
            }
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for auth response
            response = await self.websocket.recv()
            auth_data = json.loads(response)
            
            if auth_data.get("status") != "auth_success":
                raise Exception(f"Authentication failed: {auth_data}")
            
            self.logger.info("[SUCCESS] Connected and authenticated!")
            
            # Subscribe to aggregate bars (candles)
            subscribe_message = {
                "action": "subscribe",
                "params": f"AM.{',AM.'.join(self.config.symbols)}"  # AM = aggregate minute bars
            }
            await self.websocket.send(json.dumps(subscribe_message))
            
            self.logger.info(f"[SUBSCRIBE] Subscribed to {len(self.config.symbols)} symbols: {', '.join(self.config.symbols)}")
            self.is_running = True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Connection failed: {e}")
            raise
    
    def _on_pattern_detected(self, pattern: PatternMatch):
        """Handle detected patterns"""
        self.detected_patterns.append(pattern)
        
        confidence_pct = pattern.confidence * 100
        self.logger.info("")
        self.logger.info("[PATTERN] PATTERN DETECTED!")
        self.logger.info(f"   Symbol: {pattern.symbol}")
        self.logger.info(f"   Pattern: {pattern.pattern_type.value}")
        self.logger.info(f"   Confidence: {confidence_pct:.1f}%")
        self.logger.info(f"   Trigger Price: ${pattern.trigger_price:.2f}")
        self.logger.info(f"   Time: {pattern.timestamp.strftime('%H:%M:%S')}")
        self.logger.info("")
        
        # Broadcast pattern to websocket clients
        asyncio.create_task(self._broadcast_pattern(pattern))
        
        # Optional: Play sound or send notification
        self._send_notification(pattern)
    
    def _send_notification(self, pattern: PatternMatch):
        """Send notification (you can customize this)"""
        try:
            # Simple console beep
            print("\a")  # Bell character
        except:
            pass
    
    async def _process_message(self, message: str):
        """Process incoming websocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            for item in data:
                if item.get("ev") == "AM":  # Aggregate Minute bar
                    await self._process_candle_data(item)
                elif item.get("ev") == "status":
                    self.logger.info(f"Status: {item.get('message', 'Unknown')}")
                    
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def _process_candle_data(self, candle_data: Dict):
        """Process incoming candle data and detect patterns"""
        symbol = candle_data.get("sym")
        if symbol not in self.config.symbols:
            return
        
        try:
            # Convert Polygon data to our CandlestickTick format
            candle = CandlestickTick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(candle_data.get("s", 0) / 1000),  # Start timestamp
                open=float(candle_data.get("o", 0)),
                high=float(candle_data.get("h", 0)),
                low=float(candle_data.get("l", 0)),
                close=float(candle_data.get("c", 0)),
                volume=int(candle_data.get("v", 0))
            )
            
            # Add to history
            self.candle_history[symbol].append(candle)
            
            # Keep only recent candles
            if len(self.candle_history[symbol]) > self.config.max_lookback_candles:
                self.candle_history[symbol] = self.candle_history[symbol][-self.config.max_lookback_candles:]
            
            # Log candle data
            self.logger.info(f"[DATA] {symbol}: ${candle.close:.2f} | Vol: {candle.volume:,} | {candle.timestamp.strftime('%H:%M:%S')}")
            
            # Detect patterns if we have enough history
            if len(self.candle_history[symbol]) >= 10:  # Minimum candles for pattern detection
                await self._detect_patterns_for_symbol(symbol)
                
        except Exception as e:
            self.logger.error(f"Error processing candle for {symbol}: {e}")
    
    async def _detect_patterns_for_symbol(self, symbol: str):
        """Run pattern detection for a specific symbol"""
        try:
            candles = self.candle_history[symbol]
            if len(candles) < 10:
                return
            
            # Use the last few candles for pattern detection
            recent_candles = candles[-50:]  # Look at last 50 candles
            
            # Sync our candle history to the pattern detector and run detection
            # Clear the pattern detector's history for this symbol to avoid duplicates
            if symbol in self.pattern_detector.candle_history:
                self.pattern_detector.candle_history[symbol].clear()
            
            # Add all recent candles to pattern detector (it maintains its own history)
            for candle in recent_candles:
                detected_patterns = self.pattern_detector.add_candle(candle)
                # Only process patterns from the latest candle to avoid duplicates
                if candle == recent_candles[-1]:
                    for pattern in detected_patterns:
                        # Check if this is a new pattern and meets confidence threshold
                        if (pattern.confidence >= self.config.min_confidence and 
                            self._is_new_pattern(pattern)):
                            # Pattern callbacks are automatically triggered by add_candle
                            pass
            
        except Exception as e:
            self.logger.error(f"Pattern detection error for {symbol}: {e}")
    
    def _is_new_pattern(self, new_pattern: PatternMatch) -> bool:
        """Check if this pattern was recently detected to avoid duplicates"""
        now = datetime.now()
        recent_threshold = timedelta(minutes=5)  # Don't repeat patterns within 5 minutes
        
        for existing_pattern in self.detected_patterns:
            if (existing_pattern.symbol == new_pattern.symbol and
                existing_pattern.pattern_type == new_pattern.pattern_type and
                abs((now - existing_pattern.timestamp).total_seconds()) < recent_threshold.total_seconds()):
                return False
        
        return True
    
    async def listen(self):
        """Main listening loop"""
        if not self.websocket:
            raise Exception("Not connected. Call connect() first.")
        
        self.logger.info("[LISTEN] Listening for patterns...")
        
        try:
            async for message in self.websocket:
                await self._process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("[WARNING] Connection closed")
            self.is_running = False
        except KeyboardInterrupt:
            self.logger.info("[INFO] Stopped by user")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"[ERROR] Listening error: {e}")
            self.is_running = False
    
    async def disconnect(self):
        """Disconnect from websocket"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_running = False
        self.logger.info("[DISCONNECT] Disconnected")
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        pattern_counts = {}
        for pattern in self.detected_patterns:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            "total_patterns_detected": len(self.detected_patterns),
            "patterns_by_type": pattern_counts,
            "symbols_monitored": len(self.config.symbols),
            "candle_history_lengths": {symbol: len(candles) for symbol, candles in self.candle_history.items()}
        }


def load_config() -> LiveConfig:
    """Load configuration from file or environment"""
    config = LiveConfig()
    
    # Try to load from config file
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
        from live_config import LIVE_CONFIG
        config = LIVE_CONFIG
    except ImportError:
        pass
    
    
    # Override with environment variable if present
    api_key = os.getenv('POLYGON_API_KEY')
    if api_key:
        config.api_key = api_key
    
    return config


async def main():
    """Main entry point"""
    print("[START] LIVE PATTERN DETECTOR")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        
        if not config.api_key:
            print("[ERROR] No API key found!")
            print("Set your Polygon API key in config/live_config.py")
            return
        
        # Create detector
        detector = LivePatternDetector(config)
        
        print(f"[MONITOR] Monitoring {len(config.symbols)} symbols for {len(config.pattern_types)} pattern types")
        print(f"[CONFIG] Minimum confidence: {config.min_confidence * 100:.1f}%")
        print(f"[SYMBOLS] Symbols: {', '.join(config.symbols)}")
        print("")
        
        # Start broadcast server
        await detector.start_broadcast_server()
        
        # Connect and start listening
        await detector.connect()
        await detector.listen()
        
    except KeyboardInterrupt:
        print("\n[EXIT] Goodbye!")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
    finally:
        if 'detector' in locals():
            await detector.disconnect()
            await detector.stop_broadcast_server()


if __name__ == "__main__":
    asyncio.run(main())