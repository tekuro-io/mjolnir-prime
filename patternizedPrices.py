import asyncio
import websockets
import time
import json
import random
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque
from datetime import datetime
from enum import Enum

# --- Configuration ---
# The URL of your Python WebSocket server (where the server container is exposed)
WS_SERVER_URL = "wss://hermes.tekuro.io/"

# Pattern notification classes (simplified from trading_simulator)
class PatternType(Enum):
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    FLAT_TOP_BREAKOUT = "flat_top_breakout"

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PatternNotification:
    pattern_type: str
    symbol: str
    timestamp: datetime
    confidence: float
    trigger_price: float
    alert_level: str
    message: str
    metadata: Dict = None
    
    def to_websocket_message(self) -> Dict:
        # Determine if pattern is bullish or bearish
        is_bullish = "bullish" in self.pattern_type.lower()
        is_bearish = "bearish" in self.pattern_type.lower()
        
        # Create display name from pattern type
        display_name = self.pattern_type.replace('_', ' ').title()
        
        return {
            "topic": "pattern_detection",
            "data": {
                "ticker": self.symbol,
                "pattern": self.pattern_type,
                "pattern_display_name": display_name,
                "price": self.trigger_price,
                "timestamp": self.timestamp.isoformat(),
                "confidence": self.confidence,
                "alert_level": self.alert_level,
                "message": self.message,
                "is_bullish": is_bullish,
                "is_bearish": is_bearish,
                "direction": "bullish" if is_bullish else ("bearish" if is_bearish else "neutral"),
                "metadata": self.metadata or {}
            }
        }

@dataclass
class SupportResistancePattern:
    support_level: float
    resistance_level: float
    bounce_strength: float = 0.8
    active: bool = False
    activation_time: Optional[float] = None
    duration: float = 60.0

@dataclass
class Candle:
    timestamp: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    is_bullish: bool = None
    
    def __post_init__(self):
        if self.is_bullish is None:
            self.is_bullish = self.close_price > self.open_price

@dataclass
class ActivePattern:
    pattern_type: str
    stage: int
    start_time: float
    target_price: Optional[float] = None
    confidence: float = 0.7
    duration: float = 30.0

@dataclass
class CandleBuilder:
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    start_time: float
    tick_count: int = 0
    
    def add_tick(self, price: float):
        self.close_price = price
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.tick_count += 1
    
    def to_candle(self) -> Candle:
        return Candle(
            timestamp=self.start_time,
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            close_price=self.close_price
        )

class PatternManager:
    def __init__(self, history_size: int = 50, buffer_size: int = 10):
        self.patterns: Dict[str, SupportResistancePattern] = {}
        self.pattern_counters: Dict[str, int] = {}
        self.candle_history: Dict[str, deque] = {}
        self.active_patterns: Dict[str, List[ActivePattern]] = {}
        self.last_pattern_time: Dict[str, float] = {}
        self.current_candles: Dict[str, CandleBuilder] = {}
        self.history_size = history_size
        self.tolerance = 0.01
        self.max_inactivity_time = 8.0  # 8 minutes (2 candles)
        
        # Buffering for synchronization
        self.buffer_size = buffer_size
        self.completed_candles_buffer: Dict[str, deque] = {}  # Recent completed candles
        self.last_sync_time: Dict[str, float] = {}  # Track last sync for each ticker
        
        # Connection status tracking
        self.last_connection_log = 0.0
        
        # Pattern notification tracking
        self.websocket_url = None
        self.pattern_notifications: List[PatternNotification] = []
    
    def add_support_resistance(self, ticker: str, support: float, resistance: float, duration: float = 60.0):
        self.patterns[ticker] = SupportResistancePattern(
            support_level=support,
            resistance_level=resistance,
            duration=duration,
            activation_time=time.time()
        )
        self.pattern_counters[ticker] = 0
    
    def should_activate_pattern(self, ticker: str, current_price: float) -> bool:
        # Disable support/resistance patterns - only use reversal patterns
        return False
    
    def apply_support_resistance(self, ticker: str, current_price: float, price_change: float) -> float:
        if ticker not in self.patterns:
            return price_change
        
        pattern = self.patterns[ticker]
        new_price = current_price + price_change
        
        if new_price <= pattern.support_level:
            bounce_strength = pattern.bounce_strength + random.uniform(0, 0.3)
            return (pattern.support_level - current_price) + (bounce_strength * abs(price_change))
        
        elif new_price >= pattern.resistance_level:
            bounce_strength = pattern.bounce_strength + random.uniform(0, 0.3)
            return (pattern.resistance_level - current_price) - (bounce_strength * abs(price_change))
        
        return price_change
    
    def add_price_tick(self, ticker: str, price: float) -> bool:
        """Add a price tick and return True if a new 1-minute candle was completed"""
        current_time = time.time()
        minute_timestamp = self._get_candle_start_time(current_time)
        
        if ticker not in self.candle_history:
            self.candle_history[ticker] = deque(maxlen=self.history_size)
            self.active_patterns[ticker] = []
        
        # Check if we need to start a new candle
        if ticker not in self.current_candles:
            self.current_candles[ticker] = CandleBuilder(price, price, price, price, minute_timestamp)
            return False
        
        current_candle = self.current_candles[ticker]
        
        # If we've moved to a new minute boundary, complete the current candle and start a new one
        if minute_timestamp > current_candle.start_time:
            completed_candle = current_candle.to_candle()
            self.candle_history[ticker].append(completed_candle)
            
            # Close the completed candle at the exact minute boundary
            completed_candle.timestamp = current_candle.start_time
            
            # Add to buffer for synchronization
            self._add_to_buffer(ticker, completed_candle)
            
            # Start new candle at the new minute boundary
            self.current_candles[ticker] = CandleBuilder(price, price, price, price, minute_timestamp)
            return True
        else:
            # Add tick to current candle
            current_candle.add_tick(price)
            return False
    
    def _get_candle_start_time(self, current_time: float) -> float:
        """Get the start time of the current 1-minute candle boundary"""
        return int(current_time // 60) * 60
    
    def _format_candle_time(self, timestamp: float) -> str:
        """Format timestamp for display"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    
    def _add_to_buffer(self, ticker: str, candle: Candle):
        """Add completed candle to synchronization buffer"""
        if ticker not in self.completed_candles_buffer:
            self.completed_candles_buffer[ticker] = deque(maxlen=self.buffer_size)
        
        self.completed_candles_buffer[ticker].append(candle)
        self.last_sync_time[ticker] = time.time()
    
    def get_buffered_candles(self, ticker: str, since_time: float = None) -> List[Candle]:
        """Get buffered candles for a ticker, optionally since a specific time"""
        if ticker not in self.completed_candles_buffer:
            return []
        
        candles = list(self.completed_candles_buffer[ticker])
        
        if since_time is not None:
            candles = [c for c in candles if c.timestamp >= since_time]
        
        return candles
    
    def get_current_candle_state(self, ticker: str) -> Optional[Dict]:
        """Get current incomplete candle state for synchronization"""
        if ticker not in self.current_candles:
            return None
        
        builder = self.current_candles[ticker]
        current_time = time.time()
        
        return {
            'ticker': ticker,
            'start_time': builder.start_time,
            'open_price': builder.open_price,
            'high_price': builder.high_price,
            'low_price': builder.low_price,
            'close_price': builder.close_price,
            'tick_count': builder.tick_count,
            'age_seconds': current_time - builder.start_time,
            'next_boundary': self._get_candle_start_time(current_time + 60)
        }
    
    def sync_candle_state(self, ticker: str, candle_state: Dict):
        """Sync to an existing candle state (for mid-candle connections)"""
        self.current_candles[ticker] = CandleBuilder(
            open_price=candle_state['open_price'],
            high_price=candle_state['high_price'],
            low_price=candle_state['low_price'],
            close_price=candle_state['close_price'],
            start_time=candle_state['start_time'],
            tick_count=candle_state['tick_count']
        )
    
    def detect_flat_top_breakout(self, ticker: str) -> Optional[ActivePattern]:
        history = self.candle_history.get(ticker, [])
        if len(history) < 3:
            return None
        
        candle1, candle2, candle3 = history[-3], history[-2], history[-1]
        
        # Check if first two candles have similar highs
        high_diff = abs(candle1.high_price - candle2.high_price) / candle1.high_price
        if high_diff > self.tolerance:
            return None
        
        # Check if third candle breaks above resistance
        resistance_level = max(candle1.high_price, candle2.high_price)
        if candle3.close_price > resistance_level * (1 + self.tolerance):
            target_price = resistance_level * 1.03  # 3% above resistance
            return ActivePattern("flat_top_breakout", 1, time.time(), target_price, 0.8, 240.0)  # 4 minutes
        
        return None
    
    def detect_bullish_reversal(self, ticker: str) -> Optional[ActivePattern]:
        history = self.candle_history.get(ticker, [])
        if len(history) < 4:
            return None
        
        c1, c2, c3, c4 = history[-4], history[-3], history[-2], history[-1]
        
        # Pattern: Bullish -> Bearish -> Bearish -> Bullish confirmation
        if (c1.is_bullish and 
            not c2.is_bullish and c2.close_price < c1.close_price and
            not c3.is_bullish and c3.close_price < c2.close_price and
            c4.is_bullish and c4.close_price > c2.open_price):
            
            target_price = c1.high_price * 1.02  # 2% above initial high
            return ActivePattern("bullish_reversal", 1, time.time(), target_price, 0.75, 300.0)  # 5 minutes
        
        return None
    
    def detect_bearish_reversal(self, ticker: str) -> Optional[ActivePattern]:
        history = self.candle_history.get(ticker, [])
        if len(history) < 4:
            return None
        
        c1, c2, c3, c4 = history[-4], history[-3], history[-2], history[-1]
        
        # Pattern: Bearish -> Bullish -> Bullish -> Bearish confirmation
        if (not c1.is_bullish and 
            c2.is_bullish and c2.close_price > c1.close_price and
            c3.is_bullish and c3.close_price > c2.close_price and
            not c4.is_bullish and c4.close_price < c2.open_price):
            
            target_price = c1.low_price * 0.98  # 2% below initial low
            return ActivePattern("bearish_reversal", 1, time.time(), target_price, 0.75, 300.0)  # 5 minutes
        
        return None
    
    def apply_pattern_movement(self, ticker: str, current_price: float, base_change: float) -> float:
        if ticker not in self.active_patterns:
            return base_change
        
        current_time = time.time()
        active_list = self.active_patterns[ticker]
        
        # Remove expired patterns and log when they stop
        expired_patterns = [p for p in active_list if (current_time - p.start_time) >= p.duration]
        for pattern in expired_patterns:
            print(f"⏹️  PATTERN STOPPED: {pattern.pattern_type} for {ticker} (expired after {pattern.duration/60:.1f}min)")
        
        active_list[:] = [p for p in active_list if (current_time - p.start_time) < p.duration]
        
        if not active_list:
            return base_change
        
        # Apply the strongest pattern
        strongest_pattern = max(active_list, key=lambda p: p.confidence)
        
        if strongest_pattern.target_price:
            direction = 1 if strongest_pattern.target_price > current_price else -1
            strength = strongest_pattern.confidence * 0.3  # Max 30% of price change
            pattern_influence = direction * abs(base_change) * strength
            
            return base_change + pattern_influence
        
        return base_change
    
    def detect_and_activate_patterns(self, ticker: str):
        patterns_to_check = [
            self.detect_bullish_reversal,
            # self.detect_bearish_reversal  # Disabled for testing
        ]
        
        for detect_func in patterns_to_check:
            pattern = detect_func(ticker)
            if pattern:
                if ticker not in self.active_patterns:
                    self.active_patterns[ticker] = []
                self.active_patterns[ticker].append(pattern)
                print(f"🚀 PATTERN STARTED: {pattern.pattern_type} for {ticker} - confidence: {pattern.confidence:.0%}, target: ${pattern.target_price:.2f}")
                
                # Create and send pattern notification
                self._send_pattern_notification(ticker, pattern, "natural")
                
                self.last_pattern_time[ticker] = time.time()
                break
    
    def needs_forced_pattern_activation(self, ticker: str) -> bool:
        current_time = time.time()
        
        # Only check for active reversal patterns
        if ticker in self.active_patterns and self.active_patterns[ticker]:
            return False
        
        # Check time since last pattern
        if ticker not in self.last_pattern_time:
            self.last_pattern_time[ticker] = current_time
            return False
        
        time_since_last_pattern = current_time - self.last_pattern_time[ticker]
        return time_since_last_pattern > self.max_inactivity_time
    
    def force_new_reversal_pattern(self, ticker: str, current_price: float):
        # Force inject a reversal pattern - only bullish for testing
        pattern_type = "bullish_reversal"  # Always bullish for testing
        
        if pattern_type == "bullish_reversal":
            target_price = current_price * 1.02  # 2% above
        else:
            target_price = current_price * 0.98  # 2% below
            
        forced_pattern = ActivePattern(pattern_type, 1, time.time(), target_price, 0.8, 300.0)  # 5 minutes
        
        if ticker not in self.active_patterns:
            self.active_patterns[ticker] = []
        self.active_patterns[ticker].append(forced_pattern)
        self.last_pattern_time[ticker] = time.time()
        
        print(f"🔄 FORCED PATTERN STARTED: {pattern_type} for {ticker} - target: ${target_price:.2f} (auto-generated after inactivity)")
        
        # Create and send pattern notification for forced pattern
        self._send_pattern_notification(ticker, forced_pattern, "forced")
    
    def set_websocket_url(self, url: str):
        """Set WebSocket URL for pattern notifications"""
        self.websocket_url = url
    
    def _send_pattern_notification(self, ticker: str, pattern: ActivePattern, detection_type: str):
        """Create and queue pattern notification for WebSocket sending"""
        # Determine alert level based on confidence
        if pattern.confidence >= 0.85:
            alert_level = "high"
        elif pattern.confidence >= 0.7:
            alert_level = "medium"
        else:
            alert_level = "low"
        
        # Create message
        confidence_pct = int(pattern.confidence * 100)
        message = f"{pattern.pattern_type.replace('_', ' ').title()} detected for {ticker} at ${pattern.target_price:.2f} ({confidence_pct}% confidence)"
        
        if detection_type == "forced":
            message += " [Auto-generated]"
        
        if alert_level == "high":
            message += " ⚡ High confidence signal"
        
        # Create notification
        notification = PatternNotification(
            pattern_type=pattern.pattern_type,
            symbol=ticker,
            timestamp=datetime.now(),
            confidence=pattern.confidence,
            trigger_price=pattern.target_price,
            alert_level=alert_level,
            message=message,
            metadata={
                "detection_type": detection_type,
                "duration_minutes": pattern.duration / 60,
                "stage": pattern.stage
            }
        )
        
        # Store notification
        self.pattern_notifications.append(notification)
        
        # Send via WebSocket if URL is set
        if self.websocket_url:
            asyncio.create_task(self._send_notification_to_websocket(notification))
    
    async def _send_notification_to_websocket(self, notification: PatternNotification):
        """Send pattern notification to WebSocket"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                message = notification.to_websocket_message()
                await websocket.send(json.dumps(message))
                print(f"📡 Sent pattern notification for {notification.symbol} via WebSocket")
        except Exception as e:
            print(f"❌ Failed to send pattern notification: {e}")

def generate_stock_data(ticker: str, current_price: float, pattern_manager: PatternManager):
    """
    Generates a single simulated stock data point with pattern injection.
    It simulates a small random price fluctuation, modified by active patterns.
    """
    timestamp = int(time.time() * 1000) # Milliseconds since epoch
    # Simulate price fluctuation
    base_price_change = random.uniform(-0.02, 0.02)  # Smaller changes for better pattern visibility
    
    # Add price tick and check if a new candle was completed
    candle_completed = pattern_manager.add_price_tick(ticker, current_price)
    
    # Only detect patterns when a new 1-minute candle is completed
    if candle_completed:
        pattern_manager.detect_and_activate_patterns(ticker)
        
        # Check for forced pattern activation
        if pattern_manager.needs_forced_pattern_activation(ticker):
            pattern_manager.force_new_reversal_pattern(ticker, current_price)
    
    # Skip support/resistance - only use reversal patterns
    price_change = base_price_change
    
    # Then apply multi-candle pattern movements
    price_change = pattern_manager.apply_pattern_movement(ticker, current_price, price_change)
    
    new_price = current_price + price_change
    
    # Ensure price doesn't go negative
    if new_price < 0:
        new_price = 0.01
    
    # Candle bullish/bearish determination is handled in CandleBuilder.to_candle()
        
    return {
        "ticker": ticker.upper(),
        "timestamp": timestamp,
        "price": round(new_price, 2) # Round price to 2 decimal places
    }, new_price

async def publish_stock_data_via_websocket(tickers: list[str], interval_seconds: float = 1.0):
    """
    Publishes simulated stock data directly to the WebSocket server for a list of tickers.
    The data is wrapped in a JSON object that includes the 'topic' and the 'data' payload.
    """
    print(f"\n--- Starting WebSocket producer for tickers: {', '.join(tickers)} ---")
    print(f"Attempting to connect to WebSocket server at: {WS_SERVER_URL}")
    print("Press Ctrl+C to stop.")

    # Initialize pattern manager and base prices
    pattern_manager = PatternManager()
    pattern_manager.set_websocket_url(WS_SERVER_URL)  # Enable pattern notifications
    current_prices = {ticker.upper(): random.uniform(1.60, 1.62) for ticker in tickers}
    start_time = time.time()
    
    # Only using reversal patterns - no support/resistance setup needed
    print("Pattern detection enabled: Bullish Reversal Only (Bearish Disabled for Testing)")
    while True:
        try:
            # Establish WebSocket connection. The 'async with' ensures proper connection/disconnection.
            async with websockets.connect(WS_SERVER_URL) as websocket:
                print("WebSocket connection established.")
                message_count = 0
                while True:
                    for ticker in tickers:
                        # Get the current price for this specific ticker
                        current_price = current_prices[ticker.upper()]
                        
                        # Generate the actual stock data payload and update the price
                        data_payload, new_price = generate_stock_data(ticker, current_price, pattern_manager)
                        current_prices[ticker.upper()] = new_price # Update the stored price

                        # The topic for this producer will be based on the stock ticker
                        topic = f"stock:{ticker.upper()}"
                        
                        # Get current candle state for synchronization
                        candle_state = pattern_manager.get_current_candle_state(ticker.upper())
                        
                        # Construct the full message expected by the server:
                        # {"topic": "stock:TICKER", "data": { "ticker": "...", "price": ..., "timestamp": ... }}
                        message_to_send = {
                            "topic": topic,
                            "data": data_payload,
                            "sync": {
                                "candle_state": candle_state,
                                "server_time": time.time(),
                                "next_candle_boundary": pattern_manager._get_candle_start_time(time.time() + 60)
                            }
                        }
                        # Convert the Python dictionary to a JSON string
                        message_json = json.dumps(message_to_send)

                        # Send the JSON string over the WebSocket
                        await websocket.send(message_json)
                        message_count += 1

                        # Log WebSocket connection status once per minute
                        current_time = time.time()
                        if current_time - pattern_manager.last_connection_log >= 60.0:
                            pattern_manager.last_connection_log = current_time
                            print(f"✓ WebSocket connected - streaming {len(tickers)} tickers (uptime: {(current_time - start_time)/60:.1f}min)")
                        
                        # Only log individual messages in debug mode (commented out for production)
                        # print(f"Sent via WebSocket to topic '{topic}': {message_json} (Messages sent: {message_count})")
                    
                    # Wait for the specified interval before sending the next batch of messages
                    await asyncio.sleep(interval_seconds)

        # --- Error Handling and Reconnection Logic ---
        # Handle graceful connection closure by the server or client
        except websockets.exceptions.ConnectionClosedOK:
            print("WebSocket connection closed cleanly. Attempting to reconnect...")
        # Handle connection closure with an error
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection closed with error: {e}. Attempting to reconnect...")
        # Handle cases where the server is not reachable
        except ConnectionRefusedError:
            print(f"Connection refused. Is the WebSocket server running at {WS_SERVER_URL}? Retrying in 3 seconds...")
        # Catch any other unexpected exceptions
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in 3 seconds...")

        # Wait before attempting to reconnect to prevent rapid-fire connection attempts
        await asyncio.sleep(3)

if __name__ == "__main__":
    # Default tickers if no command-line arguments are provided
    #default_tickers = ["AGH","ALZN","BRAG","CBAN","AGH","EEX","ERNA","FEAM","FTEL","GCTK","GSRT","IPOD","LIDR","NAMM","ODYS","OSRH","PN","POLE","RPID","SOWG","SRTS","STFS","SUGP","WLDS","ZCMD"]
    default_tickers = ["LOOP", "FINW", "PHIO", "LPAA", "JEM", "MCVT", "CMCT", "LYEL", "SPPL", "LIDR", "SMX", "OCTO"]
    # Get tickers from command-line arguments, or use default
    # Expects arguments like: python your_script.py TICKER1 TICKER2 TICKER3
    tickers_input = sys.argv[1:] if len(sys.argv) > 1 else default_tickers
    
    if len(sys.argv) <= 1:
        print(f"No tickers provided. Using default tickers: {', '.join(default_tickers)}")

    try:
        # Run the asynchronous function
        asyncio.run(publish_stock_data_via_websocket(tickers_input))
    except KeyboardInterrupt:
        # Handle Ctrl+C to stop the script gracefully
        print("\nProducer stopped by user.")
    except Exception as e:
        print(f"An error occurred during asyncio run: {e}")
