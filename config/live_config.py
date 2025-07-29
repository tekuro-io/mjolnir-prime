"""
Live Pattern Detection Configuration
====================================
Configuration file for live pattern detection tool.

To use:
1. Add your Polygon API key below
2. Customize symbols and settings as needed
"""

from trading_simulator.core.models import PatternType
from live_pattern_detector import LiveConfig

# INSERT YOUR POLYGON API KEY HERE
POLYGON_API_KEY = ""

# Live detection configuration
LIVE_CONFIG = LiveConfig(
    api_key=POLYGON_API_KEY,
    
    # Symbols to monitor (you can customize this list)
    symbols=[
        'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'META', 
        'UBER', 'ROKU', 'PLTR', 'AVGO', 'DNA', 'SPY', 'QQQ'
    ],
    
    # Pattern types to detect
    pattern_types=[
        PatternType.BULLISH_REVERSAL,
        PatternType.BEARISH_REVERSAL,
        # PatternType.FLAT_TOP_BREAKOUT,  # Uncomment to enable
        # PatternType.CUP_AND_HANDLE,    # Uncomment to enable
    ],
    
    # Detection settings
    min_confidence=0.65,          # Minimum confidence threshold (65%)
    candle_timeframe="1min",      # Timeframe for candles
    max_lookback_candles=100,     # How many candles to keep in memory
    
    # Websocket broadcasting settings
    broadcast_enabled=True,       # Enable pattern broadcasting
    broadcast_host="localhost",   # Host for websocket server
    broadcast_port=8765,          # Port for websocket server
)

# Alternative: You can also set via environment variable
# export POLYGON_API_KEY="your_key_here"
# Then the tool will automatically pick it up