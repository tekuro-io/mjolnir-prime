"""
Real-time trading engine with WebSocket integration and pattern notifications.
"""

import asyncio
import logging
from typing import List, Dict, Callable, Optional
from datetime import datetime

from ..trading.engine import TradeEngine
from ..data.websocket_client import WebSocketClient, TickData
from ..data.candle_aggregator import CandleAggregator, RealTimeDataManager
from ..config.websocket_config import TradingWebSocketConfig
from ..core.models import CandlestickTick, PatternMatch
from .pattern_notifier import PatternNotifier


class RealTimeTradingEngine:
    """Enhanced trading engine for real-time WebSocket data processing"""
    
    def __init__(self, trade_engine: TradeEngine, websocket_config: TradingWebSocketConfig):
        self.trade_engine = trade_engine
        self.websocket_config = websocket_config
        
        # Real-time components
        self.websocket_client = None
        self.candle_aggregator = None
        self.data_manager = None
        self.pattern_notifier = PatternNotifier(websocket_config.url)
        
        # State
        self.is_running = False
        self.start_time = None
        
        # Callbacks
        self.pattern_callbacks: List[Callable[[PatternMatch], None]] = []
        self.candle_callbacks: List[Callable[[CandlestickTick], None]] = []
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        
        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'candles_completed': 0,
            'patterns_detected': 0,
            'uptime_seconds': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def register_pattern_callback(self, callback: Callable[[PatternMatch], None]):
        """Register callback for pattern detection events"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.pattern_callbacks.append(callback)
        
    def register_candle_callback(self, callback: Callable[[CandlestickTick], None]):
        """Register callback for completed candles"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.candle_callbacks.append(callback)
        
    def register_tick_callback(self, callback: Callable[[TickData], None]):
        """Register callback for raw tick data"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.tick_callbacks.append(callback)
    
    def _setup_components(self):
        """Initialize WebSocket and data processing components"""
        # Create WebSocket client
        ws_config = self.websocket_config.to_websocket_config()
        self.websocket_client = WebSocketClient(ws_config)
        
        # Create candle aggregator
        self.candle_aggregator = CandleAggregator(
            interval_minutes=self.websocket_config.candle_interval_minutes
        )
        
        # Create data manager to wire everything together
        self.data_manager = RealTimeDataManager(
            websocket_client=self.websocket_client,
            candle_aggregator=self.candle_aggregator,
            trade_engine=self.trade_engine
        )
        
        # Setup pattern detection callbacks if enabled
        if self.websocket_config.enable_pattern_detection:
            self._setup_pattern_detection()
        
        # Setup additional callbacks
        self._setup_callbacks()
    
    def _setup_pattern_detection(self):
        """Setup pattern detection with notifications"""
        # Override trade engine's pattern detector to add notifications
        original_trigger_callbacks = self.trade_engine.pattern_detector._trigger_callbacks
        
        def enhanced_trigger_callbacks(pattern: PatternMatch):
            # Call original callbacks first
            original_trigger_callbacks(pattern)
            
            # Update stats
            self.stats['patterns_detected'] += 1
            
            # Create notification
            notification = self.pattern_notifier.create_notification(pattern)
            self.logger.info(f"Pattern detected: {notification}")
            
            # Call registered pattern callbacks
            for callback in self.pattern_callbacks:
                try:
                    callback(pattern)
                except Exception as e:
                    self.logger.error(f"Error in pattern callback: {e}")
        
        # Replace the method
        self.trade_engine.pattern_detector._trigger_callbacks = enhanced_trigger_callbacks
    
    def _setup_callbacks(self):
        """Setup additional callbacks for monitoring"""
        # Monitor ticks
        def on_tick(tick: TickData):
            self.stats['ticks_processed'] += 1
            
            # Call registered tick callbacks
            for callback in self.tick_callbacks:
                try:
                    callback(tick)
                except Exception as e:
                    self.logger.error(f"Error in tick callback: {e}")
        
        self.websocket_client.register_tick_callback(on_tick)
        
        # Monitor completed candles
        def on_candle(candle: CandlestickTick):
            self.stats['candles_completed'] += 1
            
            # Call registered candle callbacks  
            for callback in self.candle_callbacks:
                try:
                    callback(candle)
                except Exception as e:
                    self.logger.error(f"Error in candle callback: {e}")
        
        self.candle_aggregator.register_candle_callback(on_candle)
    
    async def start(self):
        """Start the real-time trading engine"""
        if self.is_running:
            self.logger.warning("Engine is already running")
            return
            
        self.logger.info("Starting real-time trading engine")
        self.start_time = datetime.now()
        
        # Setup components
        self._setup_components()
        
        # Start data processing
        self.is_running = True
        
        try:
            await self.data_manager.start()
        except Exception as e:
            self.logger.error(f"Error starting data manager: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the real-time trading engine"""
        if not self.is_running:
            self.logger.warning("Engine is not running")
            return
            
        self.logger.info("Stopping real-time trading engine")
        
        # Stop data processing
        if self.data_manager:
            self.data_manager.stop()
        
        # Close pattern notifier WebSocket connection
        if self.pattern_notifier:
            asyncio.create_task(self.pattern_notifier.close_websocket_connection())
        
        # Update uptime
        if self.start_time:
            uptime = datetime.now() - self.start_time
            self.stats['uptime_seconds'] = uptime.total_seconds()
        
        self.is_running = False
        self.logger.info("Real-time trading engine stopped")
    
    def get_status(self) -> Dict:
        """Get current engine status"""
        base_status = self.trade_engine.get_portfolio_summary()
        
        # Add real-time specific status
        realtime_status = {
            'realtime_engine': {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'websocket_connected': self.websocket_client.is_connected if self.websocket_client else False,
                'symbols': self.websocket_config.symbols,
                'candle_interval_minutes': self.websocket_config.candle_interval_minutes,
                'pattern_detection_enabled': self.websocket_config.enable_pattern_detection
            },
            'statistics': self.stats.copy(),
            'data_manager_status': self.data_manager.get_status() if self.data_manager else None
        }
        
        # Merge with base status
        base_status.update(realtime_status)
        return base_status
    
    def get_live_pattern_summary(self) -> Dict:
        """Get summary of recently detected patterns"""
        recent_patterns = self.trade_engine.get_portfolio_summary().get('recent_patterns', [])
        
        return {
            'total_patterns_detected': self.stats['patterns_detected'],
            'recent_patterns': recent_patterns,
            'pattern_types_detected': list(set(p.get('type') for p in recent_patterns)),
            'symbols_with_patterns': list(set(p.get('symbol') for p in recent_patterns))
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'ticks_processed': 0,
            'candles_completed': 0,
            'patterns_detected': 0,
            'uptime_seconds': 0
        }
        self.start_time = datetime.now() if self.is_running else None
    
    def force_complete_candles(self):
        """Force completion of any active candles"""
        if self.candle_aggregator:
            self.candle_aggregator.force_complete_all_candles()
    
    # Convenience methods to access underlying engine functionality
    def place_market_order(self, symbol: str, side, quantity: int) -> str:
        """Place market order through underlying engine"""
        return self.trade_engine.place_market_order(symbol, side, quantity)
    
    def place_limit_order(self, symbol: str, side, quantity: int, price: float) -> str:
        """Place limit order through underlying engine"""
        return self.trade_engine.place_limit_order(symbol, side, quantity, price)
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from underlying engine"""
        return self.trade_engine.get_portfolio_summary()
    
    def get_trading_statistics(self) -> Dict:
        """Get trading statistics from underlying engine"""
        return self.trade_engine.get_trading_statistics()