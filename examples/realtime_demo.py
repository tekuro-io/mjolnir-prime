"""
Real-time WebSocket trading demonstration.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_simulator.portfolio.portfolio import Portfolio
from trading_simulator.patterns.detector import PatternDetector
from trading_simulator.trading.engine import TradeEngine
from trading_simulator.patterns.strategies import StrategyFactory
from trading_simulator.config.websocket_config import TradingWebSocketConfig, LOCAL_DEV_CONFIG
from trading_simulator.realtime.trading_engine import RealTimeTradingEngine
from trading_simulator.realtime.pattern_notifier import PatternNotification
from trading_simulator.data.websocket_client import TickData
from trading_simulator.core.models import CandlestickTick, PatternMatch


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_realtime_engine(websocket_url: str = "ws://localhost:8080/ws", 
                          symbols: list = None) -> RealTimeTradingEngine:
    """Create a real-time trading engine with WebSocket integration"""
    if symbols is None:
        symbols = ["AGH", "AAPL", "GOOGL"]
    
    # Create base trading components
    portfolio = Portfolio(initial_balance=100000.0)
    pattern_detector = PatternDetector(tolerance=0.01)
    trade_engine = TradeEngine(portfolio, pattern_detector, symbols)
    
    # Add strategies
    strategy_factory = StrategyFactory(trade_engine)
    strategy_factory.create_balanced_setup()
    
    # Create WebSocket configuration
    ws_config = TradingWebSocketConfig(
        url=websocket_url,
        symbols=symbols,
        enable_pattern_detection=True,
        candle_interval_minutes=1
    )
    
    # Create real-time engine
    rt_engine = RealTimeTradingEngine(trade_engine, ws_config)
    
    return rt_engine


def setup_callbacks(rt_engine: RealTimeTradingEngine):
    """Setup callbacks for monitoring real-time events"""
    
    def on_pattern_detected(pattern: PatternMatch):
        """Handle pattern detection"""
        logger.info(f"🎯 PATTERN DETECTED: {pattern.pattern_type.value} for {pattern.symbol}")
        logger.info(f"   Confidence: {pattern.confidence:.1%}")
        logger.info(f"   Trigger Price: ${pattern.trigger_price:.4f}")
        logger.info(f"   Timestamp: {pattern.timestamp}")
        
        # Example: Place order based on pattern
        if pattern.confidence > 0.8:
            from trading_simulator.core.types import OrderSide
            try:
                order_id = rt_engine.place_market_order(
                    symbol=pattern.symbol,
                    side=OrderSide.BUY,
                    quantity=100
                )
                logger.info(f"   ➡️  Placed BUY order {order_id} for {pattern.symbol}")
            except Exception as e:
                logger.error(f"   ❌ Failed to place order: {e}")
    
    def on_candle_completed(candle: CandlestickTick):
        """Handle completed candles"""
        candle_type_emoji = "🟢" if candle.candle_type.value == "bullish" else "🔴"
        logger.info(f"{candle_type_emoji} CANDLE: {candle.symbol} @ {candle.timestamp.strftime('%H:%M:%S')}")
        logger.info(f"   OHLC: ${candle.open:.4f} / ${candle.high:.4f} / ${candle.low:.4f} / ${candle.close:.4f}")
        logger.info(f"   Volume: {candle.volume:,}")
    
    def on_tick_received(tick: TickData):
        """Handle raw ticks (be careful with volume here)"""
        # Only log every 10th tick to avoid spam
        if hasattr(on_tick_received, 'counter'):
            on_tick_received.counter += 1
        else:
            on_tick_received.counter = 1
            
        if on_tick_received.counter % 10 == 0:
            logger.debug(f"📊 TICK: {tick.ticker} @ ${tick.price:.4f} ({tick.timestamp.strftime('%H:%M:%S')})")
    
    def on_notification(notification: PatternNotification):
        """Handle pattern notifications"""
        logger.info(f"🚨 ALERT: {notification}")
    
    # Register callbacks
    rt_engine.register_pattern_callback(on_pattern_detected)
    rt_engine.register_candle_callback(on_candle_completed)
    rt_engine.register_tick_callback(on_tick_received)
    rt_engine.pattern_notifier.register_notification_callback(on_notification)


async def demo_realtime_trading(websocket_url: str, symbols: list, duration_minutes: int = 5):
    """Run real-time trading demo"""
    logger.info("🚀 Starting Real-Time Trading Demo")
    logger.info(f"WebSocket URL: {websocket_url}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Duration: {duration_minutes} minutes")
    logger.info("-" * 50)
    
    # Create real-time engine
    rt_engine = create_realtime_engine(websocket_url, symbols)
    
    # Setup monitoring callbacks
    setup_callbacks(rt_engine)
    
    try:
        # Start the engine
        logger.info("🔌 Connecting to WebSocket...")
        
        # Start engine in background
        engine_task = asyncio.create_task(rt_engine.start())
        
        # Wait for connection
        await asyncio.sleep(2)
        
        if rt_engine.is_running:
            logger.info("✅ Connected and receiving data!")
            logger.info("📈 Monitoring for patterns...")
            
            # Monitor for specified duration
            for minute in range(duration_minutes):
                await asyncio.sleep(60)  # Wait 1 minute
                
                # Print status update
                status = rt_engine.get_status()
                stats = status['statistics']
                
                logger.info(f"📊 Minute {minute + 1}/{duration_minutes} Status:")
                logger.info(f"   Ticks processed: {stats['ticks_processed']}")
                logger.info(f"   Candles completed: {stats['candles_completed']}")
                logger.info(f"   Patterns detected: {stats['patterns_detected']}")
                logger.info(f"   Portfolio value: ${status['portfolio_value']:,.2f}")
                
                # Show recent patterns
                pattern_summary = rt_engine.get_live_pattern_summary()
                if pattern_summary['recent_patterns']:
                    logger.info(f"   Recent patterns: {len(pattern_summary['recent_patterns'])}")
        else:
            logger.error("❌ Failed to connect to WebSocket")
            return
            
    except KeyboardInterrupt:
        logger.info("⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        rt_engine.stop()
        
        # Final status
        final_status = rt_engine.get_status()
        final_stats = final_status['statistics']
        
        logger.info("📋 Final Results:")
        logger.info(f"   Total runtime: {final_stats['uptime_seconds']:.1f} seconds")
        logger.info(f"   Total ticks: {final_stats['ticks_processed']}")
        logger.info(f"   Total candles: {final_stats['candles_completed']}")
        logger.info(f"   Total patterns: {final_stats['patterns_detected']}")
        logger.info(f"   Final portfolio: ${final_status['portfolio_value']:,.2f}")
        logger.info(f"   Total P&L: ${final_status['total_pnl']:,.2f}")
        
        if final_status['positions']:
            logger.info("   Final positions:")
            for symbol, pos in final_status['positions'].items():
                logger.info(f"     {symbol}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")


async def demo_websocket_config():
    """Demonstrate WebSocket configuration management"""
    logger.info("🔧 WebSocket Configuration Demo")
    
    from trading_simulator.config.websocket_config import WebSocketConfigManager, create_custom_config
    
    # Create config manager
    config_manager = WebSocketConfigManager("demo_configs")
    
    # Create sample configurations
    config_manager.create_sample_configs()
    
    # Create custom config
    custom_config = create_custom_config(
        url="ws://example.com/stream",
        symbols=["CUSTOM1", "CUSTOM2"],
        enable_pattern_detection=True,
        candle_interval_minutes=1
    )
    
    config_manager.save_config("custom_demo", custom_config)
    
    # List configurations
    configs = config_manager.list_configs()
    logger.info(f"Available configurations: {configs}")
    
    # Load and display a config
    try:
        local_config = config_manager.load_config("local_dev")
        logger.info(f"Local dev config: {local_config.url} with symbols {local_config.symbols}")
    except Exception as e:
        logger.error(f"Error loading config: {e}")


def demo_pattern_notifications():
    """Demonstrate pattern notification system"""
    logger.info("🔔 Pattern Notification Demo")
    
    from trading_simulator.realtime.pattern_notifier import PatternNotifier, AlertLevel
    from trading_simulator.core.types import PatternType
    from trading_simulator.core.models import PatternMatch
    
    # Create notifier
    notifier = PatternNotifier()
    
    # Create sample pattern
    pattern = PatternMatch(
        pattern_type=PatternType.FLAT_TOP_BREAKOUT,
        confidence=0.85,
        timestamp=datetime.now(),
        symbol="AAPL",
        trigger_price=150.50,
        candles_involved=[],
        metadata={'resistance_level': 150.0}
    )
    
    # Create notification
    notification = notifier.create_notification(pattern)
    logger.info(f"Created notification: {notification}")
    
    # Show summary
    summary = notifier.get_notification_summary()
    logger.info(f"Notification summary: {summary}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        
        if demo_type == "config":
            asyncio.run(demo_websocket_config())
        elif demo_type == "notifications":
            demo_pattern_notifications()
        elif demo_type == "realtime":
            # Real-time demo with optional parameters
            url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8080/ws"
            symbols = sys.argv[3].split(",") if len(sys.argv) > 3 else ["AGH", "AAPL"]
            duration = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            
            asyncio.run(demo_realtime_trading(url, symbols, duration))
        else:
            print(f"Unknown demo type: {demo_type}")
            print("Available demos: config, notifications, realtime")
    else:
        # Run all demos
        print("🎬 Running all real-time demos...")
        
        # Config demo
        asyncio.run(demo_websocket_config())
        print()
        
        # Notifications demo
        demo_pattern_notifications()
        print()
        
        # Note about real-time demo
        print("⚠️  To run the real-time demo, you need a WebSocket server running.")
        print("Usage: python realtime_demo.py realtime [url] [symbols] [duration_minutes]")
        print("Example: python realtime_demo.py realtime ws://localhost:8080/ws AGH,AAPL 3")