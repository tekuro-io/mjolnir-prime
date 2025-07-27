"""
Example of a synchronized WebSocket listener that can connect mid-candle
and sync to the current candle state.
"""

import asyncio
import websockets
import json
import time
from datetime import datetime
from typing import Dict, Optional

class SynchronizedListener:
    def __init__(self, websocket_url: str = "wss://hermes.tekuro.io/"):
        self.websocket_url = websocket_url
        self.synced_candles: Dict[str, Dict] = {}  # Track current candle state per ticker
        self.completed_candles: Dict[str, list] = {}  # Store completed candles
        self.last_sync_time: Dict[str, float] = {}
        
    def format_time(self, timestamp: float) -> str:
        """Format timestamp for display"""
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    
    def handle_sync_data(self, ticker: str, sync_data: Dict):
        """Handle synchronization data from the producer"""
        candle_state = sync_data.get('candle_state')
        server_time = sync_data.get('server_time', time.time())
        next_boundary = sync_data.get('next_candle_boundary')
        
        if candle_state:
            # Check if this is a new candle or continuation
            current_candle = self.synced_candles.get(ticker)
            
            if not current_candle or current_candle['start_time'] != candle_state['start_time']:
                # New candle detected - save previous if exists
                if current_candle and ticker in self.completed_candles:
                    print(f"📊 Candle completed for {ticker}: O={current_candle['open_price']:.2f} C={current_candle['close_price']:.2f}")
                    self.completed_candles[ticker].append(current_candle.copy())
                
                # Initialize new candle tracking
                if ticker not in self.completed_candles:
                    self.completed_candles[ticker] = []
                
                print(f"🔄 Syncing to candle for {ticker} at {self.format_time(candle_state['start_time'])} (age: {candle_state['age_seconds']:.1f}s)")
            
            # Update current candle state
            self.synced_candles[ticker] = candle_state.copy()
            self.last_sync_time[ticker] = server_time
    
    def handle_price_tick(self, ticker: str, price: float, timestamp: int):
        """Handle incoming price tick"""
        current_candle = self.synced_candles.get(ticker)
        
        if current_candle:
            # Update current candle with new tick
            current_candle['close_price'] = price
            current_candle['high_price'] = max(current_candle['high_price'], price)
            current_candle['low_price'] = min(current_candle['low_price'], price)
            current_candle['tick_count'] += 1
            
            age = time.time() - current_candle['start_time']
            print(f"📈 {ticker} tick: ${price:.2f} | Candle: O={current_candle['open_price']:.2f} H={current_candle['high_price']:.2f} L={current_candle['low_price']:.2f} C={current_candle['close_price']:.2f} | Age: {age:.1f}s")
        else:
            print(f"⚠️  Received tick for {ticker} but no candle state - waiting for sync...")
    
    def get_sync_status(self) -> Dict:
        """Get current synchronization status"""
        current_time = time.time()
        status = {}
        
        for ticker, candle in self.synced_candles.items():
            last_sync = self.last_sync_time.get(ticker, 0)
            age_since_sync = current_time - last_sync
            candle_age = current_time - candle['start_time']
            
            status[ticker] = {
                'candle_start': self.format_time(candle['start_time']),
                'candle_age_seconds': candle_age,
                'last_sync_seconds_ago': age_since_sync,
                'tick_count': candle['tick_count'],
                'is_synced': age_since_sync < 5.0  # Consider synced if updated within 5 seconds
            }
        
        return status
    
    async def listen(self):
        """Listen to WebSocket stream with synchronization"""
        print(f"🚀 Starting synchronized listener for {self.websocket_url}")
        print("This listener can connect mid-candle and sync to current state\n")
        
        while True:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    print("✅ WebSocket connection established")
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            # Extract components
                            topic = data.get('topic', '')
                            price_data = data.get('data', {})
                            sync_data = data.get('sync', {})
                            
                            if topic.startswith('stock:'):
                                ticker = topic.split(':')[1]
                                
                                # Handle synchronization data first
                                if sync_data:
                                    self.handle_sync_data(ticker, sync_data)
                                
                                # Handle price tick
                                if price_data:
                                    price = price_data.get('price')
                                    timestamp = price_data.get('timestamp')
                                    
                                    if price is not None:
                                        self.handle_price_tick(ticker, price, timestamp)
                            
                        except json.JSONDecodeError:
                            print(f"❌ Failed to parse message: {message}")
                        except Exception as e:
                            print(f"❌ Error processing message: {e}")
            
            except websockets.exceptions.ConnectionClosedOK:
                print("🔌 Connection closed cleanly. Reconnecting...")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"🔌 Connection closed with error: {e}. Reconnecting...")
            except ConnectionRefusedError:
                print(f"❌ Connection refused. Is the server running? Retrying in 3 seconds...")
            except Exception as e:
                print(f"❌ Unexpected error: {e}. Retrying in 3 seconds...")
            
            await asyncio.sleep(3)

async def main():
    """Main function to run the synchronized listener"""
    listener = SynchronizedListener()
    
    # Start listening
    listen_task = asyncio.create_task(listener.listen())
    
    # Periodic status reporting
    async def status_reporter():
        while True:
            await asyncio.sleep(10)  # Report every 10 seconds
            status = listener.get_sync_status()
            if status:
                print("\n📊 === SYNC STATUS ===")
                for ticker, info in status.items():
                    sync_indicator = "🟢" if info['is_synced'] else "🔴"
                    print(f"{sync_indicator} {ticker}: Candle {info['candle_start']} | Age: {info['candle_age_seconds']:.1f}s | Ticks: {info['tick_count']} | Last sync: {info['last_sync_seconds_ago']:.1f}s ago")
                print("=" * 30 + "\n")
    
    status_task = asyncio.create_task(status_reporter())
    
    # Run both tasks
    await asyncio.gather(listen_task, status_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Listener stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")