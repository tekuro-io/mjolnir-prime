#!/usr/bin/env python3
"""
Kubernetes Pattern Detector
============================
Real-time pattern detection service for K8s deployment.

Features:
- Redis integration for stock list retrieval
- WebSocket client for tick data subscription 
- Pattern detection and publishing
- Health check endpoints
- Environment-based configuration
"""

import asyncio
import json
import logging
import os
import sys
import time
import redis
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_simulator.core.models import CandlestickTick, PatternType
from trading_simulator.patterns.detector import PatternDetector, PatternMatch
from trading_simulator.realtime.pattern_notifier import PatternNotifier


# Pattern Detection Constants
PATTERN_MIN_CONFIDENCE = 0.65
PATTERN_TOLERANCE = 0.01
PATTERN_LOOKBACK_WINDOW = 50
LOG_LEVEL = 'INFO'

@dataclass
class K8sConfig:
    """Configuration for K8s pattern detection service"""
    # Redis Configuration
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    redis_password: Optional[str] = os.getenv('REDIS_PASSWORD')
    stock_list_key: str = os.getenv('STOCK_LIST_KEY', 'stock:latest')
    
    # WebSocket Configuration - derive URL based on environment
    websocket_url: str = os.getenv('WEBSOCKET_URL', 
        'ws://hermes.tekuro.io' if os.getenv('ENVIRONMENT', 'production') == 'production' 
        else 'wss://hermes.tekuro.io')
    
    # Environment Configuration
    environment: str = os.getenv('ENVIRONMENT', 'production')
    
    # Application Configuration
    app_name: str = os.getenv('APP_NAME', 'mjolnir-pattern-detector')
    app_version: str = os.getenv('APP_VERSION', '1.0.0')


class K8sPatternDetector:
    """Kubernetes-ready pattern detection service"""
    
    def __init__(self, config: K8sConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Pattern detection components
        self.pattern_detector = PatternDetector(
            tolerance=PATTERN_TOLERANCE,
            lookback_window=PATTERN_LOOKBACK_WINDOW
        )
        self.pattern_notifier = PatternNotifier(websocket_url=config.websocket_url)
        
        # Redis client
        self.redis_client = None
        
        # WebSocket connection
        self.websocket = None
        
        # Stock management
        self.monitored_stocks: Set[str] = set()
        self.candle_history: Dict[str, List[CandlestickTick]] = {}
        self.current_candles: Dict[str, Dict] = {}  # Building 1-minute candles
        self.last_stock_refresh = 0
        self.stock_refresh_interval = 15  # 15 seconds - check for new stocks very frequently
        
        # Service state
        self.is_running = False
        self.is_healthy = False
        self.is_ready = False
        self.startup_complete = False
        self.last_heartbeat = time.time()
        
        # Statistics
        self.stats = {
            'patterns_detected': 0,
            'messages_processed': 0,
            'errors': 0,
            'uptime_start': time.time(),
            'last_pattern_time': None,
            'stocks_monitored': 0
        }
        
        # Register pattern callbacks
        for pattern_type in [PatternType.BULLISH_REVERSAL, PatternType.BEARISH_REVERSAL, 
                           PatternType.FLAT_TOP_BREAKOUT, PatternType.DOUBLE_TOP, 
                           PatternType.DOUBLE_BOTTOM, PatternType.BULL_FLAG, 
                           PatternType.BEAR_FLAG]:
            self.pattern_detector.register_pattern_callback(pattern_type, self._on_pattern_detected)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(self.config.app_name)
        logger.info(f"Logging configured at {LOG_LEVEL} level")
        return logger
    
    async def initialize(self):
        """Initialize all service components"""
        try:
            self.logger.info(f"Initializing {self.config.app_name} v{self.config.app_version}")
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Load initial stock list
            await self._refresh_stock_list()
            
            if not self.monitored_stocks:
                self.logger.warning("No stocks found in Redis - will continue checking periodically")
            
            # Initialize WebSocket connection
            await self._initialize_websocket()
            
            
            self.is_healthy = True
            self.is_ready = True
            self.startup_complete = True
            
            self.logger.info("Service initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            self.is_healthy = False
            self.is_ready = False
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            redis_kwargs = {
                'host': self.config.redis_host,
                'port': self.config.redis_port,
                'decode_responses': True,
                'socket_connect_timeout': 5,
                'socket_timeout': 5
            }
            
            if self.config.redis_password:
                redis_kwargs['password'] = self.config.redis_password
            
            self.redis_client = redis.Redis(**redis_kwargs)
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            
            self.logger.info(f"Redis connection established: {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    async def _refresh_stock_list(self):
        """Refresh monitored stock list from Redis"""
        try:
            current_time = time.time()
            if current_time - self.last_stock_refresh < self.stock_refresh_interval:
                return
            
            # Get all keys matching the pattern scanner:latest:*
            pattern = f"{self.config.stock_list_key}:*"
            keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, pattern
            )
            
            # Extract stock symbols from keys
            new_stocks = set()
            for key in keys:
                # Extract ticker from key like "scanner:latest:AAPL"
                parts = key.split(':')
                if len(parts) >= 3:
                    ticker = parts[2]
                    new_stocks.add(ticker)
            
            # Update monitored stocks
            added_stocks = new_stocks - self.monitored_stocks
            removed_stocks = self.monitored_stocks - new_stocks
            
            if added_stocks:
                self.logger.info(f"✅ Added stocks to monitoring: {sorted(added_stocks)}")
                for stock in added_stocks:
                    # Initialize tracking for new stock
                    self.candle_history[stock] = []
                    self.current_candles[stock] = {}
                    
            if removed_stocks:
                self.logger.info(f"❌ Removed stocks from monitoring: {sorted(removed_stocks)}")
                # Clean up history for removed stocks
                for stock in removed_stocks:
                    self.candle_history.pop(stock, None)
                    self.current_candles.pop(stock, None)
            
            self.monitored_stocks = new_stocks
            self.stats['stocks_monitored'] = len(self.monitored_stocks)
            self.last_stock_refresh = current_time
            
            self.logger.info(f"Stock list refreshed: {len(self.monitored_stocks)} stocks monitored")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh stock list: {e}")
            self.stats['errors'] += 1
    
    async def _initialize_websocket(self):
        """Initialize WebSocket connection"""
        try:
            self.logger.info(f"Connecting to WebSocket: {self.config.websocket_url}")
            
            # Connect to WebSocket server
            self.websocket = await websockets.connect(
                self.config.websocket_url,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.logger.info("WebSocket connection established")
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            raise
    
    
    def _on_pattern_detected(self, pattern: PatternMatch):
        """Handle detected patterns with detailed logging"""
        try:
            self.stats['patterns_detected'] += 1
            self.stats['last_pattern_time'] = datetime.now().isoformat()
            
            confidence_pct = pattern.confidence * 100
            
            # Enhanced logging with detailed pattern information
            self.logger.info("")
            self.logger.info("🎯 PATTERN DETECTED!")
            self.logger.info(f"   📈 Symbol: {pattern.symbol}")
            self.logger.info(f"   🔍 Pattern: {pattern.pattern_type.value}")
            self.logger.info(f"   💰 Trigger Price: ${pattern.trigger_price:.2f}")
            self.logger.info(f"   📊 Confidence: {confidence_pct:.1f}%")
            self.logger.info(f"   ⏰ Time: {pattern.timestamp.strftime('%H:%M:%S')}")
            
            # Log candle information if available
            if pattern.candles_involved:
                candle_count = len(pattern.candles_involved)
                self.logger.info(f"   📊 Candles Involved: {candle_count}")
                
                # Log the trigger candle details
                trigger_candle = pattern.candles_involved[-1]
                self.logger.info(f"   📊 Trigger Candle: O={trigger_candle.open:.2f} H={trigger_candle.high:.2f} L={trigger_candle.low:.2f} C={trigger_candle.close:.2f} V={trigger_candle.volume}")
            
            # Log metadata if available
            if pattern.metadata:
                self.logger.info(f"   📋 Metadata: {pattern.metadata}")
            
            self.logger.info("")
            
            # Create and send notification
            notification = self.pattern_notifier.create_notification(pattern)
            
            # Send pattern message to WebSocket (async)
            asyncio.create_task(self._send_pattern_message(pattern))
            
        except Exception as e:
            self.logger.error(f"Error handling detected pattern: {e}")
            self.stats['errors'] += 1
    
    async def _send_pattern_message(self, pattern: PatternMatch):
        """Send pattern detection message to WebSocket in live_pattern_detector format"""
        try:
            if not self.websocket or self.websocket.closed:
                self.logger.warning("WebSocket not connected, cannot send pattern message")
                return
            
            # Get the trigger candle for candle_data
            trigger_candle = None
            if pattern.candles_involved:
                trigger_candle = pattern.candles_involved[-1]  # Use last candle as trigger
            elif pattern.symbol in self.candle_history and self.candle_history[pattern.symbol]:
                trigger_candle = self.candle_history[pattern.symbol][-1]  # Use most recent candle
            
            # Create message in live_pattern_detector format
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
                    "source": "k8s_pattern_detector"
                }
            }
            
            await self.websocket.send(json.dumps(pattern_msg))
            self.logger.debug(f"Pattern message sent: {pattern.symbol} - {pattern.pattern_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to send pattern message: {e}")
            self.stats['errors'] += 1
    
    async def _initialize_connection(self):
        """Initialize WebSocket connection - no explicit subscription needed"""
        self.logger.info(f"WebSocket connected to {self.config.websocket_url}")
        self.logger.info(f"Monitoring {len(self.monitored_stocks)} stocks: {sorted(self.monitored_stocks)}")
        
        # No explicit subscription needed - just start listening for messages
        # The WebSocket server automatically sends data for all stocks
    
    async def _process_tick_message(self, message: str):
        """Process incoming tick data message and aggregate into 1-minute candles"""
        try:
            data = json.loads(message)
            self.stats['messages_processed'] += 1
            
            # Extract tick information
            topic = data.get('topic', '')
            tick_data = data.get('data', {})
            
            # Parse topic to get ticker (format: "stock:TICKER")
            if not topic.startswith('stock:'):
                return
            
            ticker = topic.split(':', 1)[1]
            if ticker not in self.monitored_stocks:
                return
            
            # Extract tick data from the payload format
            timestamp = tick_data.get('timestamp', 0)
            price = float(tick_data.get('price', 0))
            prev_price = float(tick_data.get('prev_price', price))
            volume = int(tick_data.get('volume', 0))
            
            if price <= 0:
                return
            
            # Convert to datetime
            dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
            
            # Aggregate into 1-minute candles
            await self._aggregate_tick_to_candle(ticker, dt, price, prev_price, volume)
            
            self.last_heartbeat = time.time()
            
        except Exception as e:
            self.logger.error(f"Error processing tick message: {e}")
            self.stats['errors'] += 1
    
    async def _aggregate_tick_to_candle(self, ticker: str, timestamp: datetime, price: float, prev_price: float, volume: int):
        """Aggregate tick data into 1-minute candles"""
        try:
            # Round timestamp to 1-minute boundary
            minute_timestamp = timestamp.replace(second=0, microsecond=0)
            candle_key = f"{ticker}_{minute_timestamp.isoformat()}"
            
            # Initialize current candle if it doesn't exist
            if ticker not in self.current_candles:
                self.current_candles[ticker] = {}
            
            if candle_key not in self.current_candles[ticker]:
                # Start new 1-minute candle
                self.current_candles[ticker][candle_key] = {
                    'symbol': ticker,
                    'timestamp': minute_timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'tick_count': 1,
                    'last_update': timestamp
                }
            else:
                # Update existing candle
                candle = self.current_candles[ticker][candle_key]
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price  # Latest price becomes close
                candle['volume'] += volume
                candle['tick_count'] += 1
                candle['last_update'] = timestamp
            
            # Check if we should complete any candles (candles older than current minute)
            await self._complete_old_candles(ticker, minute_timestamp)
            
        except Exception as e:
            self.logger.error(f"Error aggregating tick to candle for {ticker}: {e}")
            self.stats['errors'] += 1
    
    async def _complete_old_candles(self, ticker: str, current_minute: datetime):
        """Complete and process candles that are older than the current minute"""
        if ticker not in self.current_candles:
            return
        
        completed_candles = []
        candles_to_remove = []
        
        for candle_key, candle_data in self.current_candles[ticker].items():
            candle_timestamp = candle_data['timestamp']
            
            # If candle is from previous minute(s), complete it
            if candle_timestamp < current_minute:
                # Create CandlestickTick object
                completed_candle = CandlestickTick(
                    symbol=candle_data['symbol'],
                    timestamp=candle_data['timestamp'],
                    open=candle_data['open'],
                    high=candle_data['high'],
                    low=candle_data['low'],
                    close=candle_data['close'],
                    volume=candle_data['volume']
                )
                
                completed_candles.append(completed_candle)
                candles_to_remove.append(candle_key)
        
        # Remove completed candles from current_candles
        for key in candles_to_remove:
            del self.current_candles[ticker][key]
        
        # Process completed candles
        for candle in completed_candles:
            await self._process_completed_candle(candle)
    
    async def _process_completed_candle(self, candle: CandlestickTick):
        """Process a completed 1-minute candle for pattern detection"""
        try:
            ticker = candle.symbol
            
            # Add to history
            if ticker not in self.candle_history:
                self.candle_history[ticker] = []
            
            self.candle_history[ticker].append(candle)
            
            # Keep only recent history
            if len(self.candle_history[ticker]) > PATTERN_LOOKBACK_WINDOW:
                self.candle_history[ticker] = self.candle_history[ticker][-PATTERN_LOOKBACK_WINDOW:]
            
            # Log completed candle
            self.logger.debug(
                f"Completed 1min candle for {ticker}: "
                f"O:{candle.open:.2f} H:{candle.high:.2f} L:{candle.low:.2f} C:{candle.close:.2f} V:{candle.volume}"
            )
            
            # Run pattern detection if we have enough history
            if len(self.candle_history[ticker]) >= 10:
                patterns = self.pattern_detector.add_candle(candle)
                # Patterns are automatically processed via callbacks
            
        except Exception as e:
            self.logger.error(f"Error processing completed candle for {candle.symbol}: {e}")
            self.stats['errors'] += 1
    
    async def run(self):
        """Main service loop"""
        self.is_running = True
        self.logger.info("Pattern detection service started")
        
        try:
            # Initialize connection (no subscription needed)
            await self._initialize_connection()
            
            # Periodic tasks
            stock_refresh_task = asyncio.create_task(self._periodic_stock_refresh())
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            # Main message processing loop
            async for message in self.websocket:
                await self._process_tick_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
        except KeyboardInterrupt:
            self.logger.info("Service stopped by user")
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            self.is_healthy = False
        finally:
            self.is_running = False
            await self._cleanup()
    
    async def _periodic_stock_refresh(self):
        """Periodically refresh stock list"""
        while self.is_running:
            try:
                await asyncio.sleep(self.stock_refresh_interval)
                old_count = len(self.monitored_stocks)
                await self._refresh_stock_list()
                new_count = len(self.monitored_stocks)
                
                if old_count != new_count:
                    self.logger.info(f"Stock count changed: {old_count} → {new_count}")
                    
            except Exception as e:
                self.logger.error(f"Error in periodic stock refresh: {e}")
                self.stats['errors'] += 1
    
    async def _heartbeat_monitor(self):
        """Monitor service health"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check if we've received messages recently
                time_since_heartbeat = time.time() - self.last_heartbeat
                if time_since_heartbeat > 300:  # 5 minutes without messages
                    self.logger.warning(f"No messages received for {time_since_heartbeat:.1f} seconds")
                    self.is_healthy = False
                else:
                    self.is_healthy = True
                    
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.close)
        
        await self.pattern_notifier.close_websocket_connection()
        
        self.logger.info("Cleanup completed")


async def main():
    """Main entry point"""
    # Setup signal handlers for graceful shutdown
    config = K8sConfig()
    detector = K8sPatternDetector(config)
    
    def signal_handler(signum, frame):
        detector.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        detector.is_running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize service
        await detector.initialize()
        
        # Run main service loop
        await detector.run()
        
    except Exception as e:
        detector.logger.error(f"Service error: {e}")
    finally:
        await detector._cleanup()


if __name__ == "__main__":
    asyncio.run(main())