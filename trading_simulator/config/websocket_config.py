"""
WebSocket configuration and connection management.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import json

from ..data.websocket_client import WebSocketConfig


@dataclass
class TradingWebSocketConfig:
    """Enhanced WebSocket configuration for trading"""
    # Connection settings
    url: str
    symbols: List[str]
    
    # Connection behavior
    reconnect_interval: int = 5
    max_reconnects: int = 10
    ping_interval: int = 30
    ping_timeout: int = 10
    
    # Authentication (if needed)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Subscription settings
    auto_subscribe: bool = True
    subscription_format: str = "stock:{symbol}"
    
    # Data processing
    enable_pattern_detection: bool = True
    candle_interval_minutes: int = 1
    
    def to_websocket_config(self) -> WebSocketConfig:
        """Convert to basic WebSocketConfig"""
        return WebSocketConfig(
            url=self.url,
            symbols=self.symbols,
            reconnect_interval=self.reconnect_interval,
            max_reconnects=self.max_reconnects,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TradingWebSocketConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to load WebSocket config from {config_path}: {e}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        # Convert to dict, excluding None values
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_env(cls) -> 'TradingWebSocketConfig':
        """Create configuration from environment variables"""
        url = os.getenv('WEBSOCKET_URL')
        if not url:
            raise ValueError("WEBSOCKET_URL environment variable is required")
            
        symbols_str = os.getenv('WEBSOCKET_SYMBOLS', 'AAPL,GOOGL,MSFT')
        symbols = [s.strip() for s in symbols_str.split(',')]
        
        return cls(
            url=url,
            symbols=symbols,
            reconnect_interval=int(os.getenv('WEBSOCKET_RECONNECT_INTERVAL', '5')),
            max_reconnects=int(os.getenv('WEBSOCKET_MAX_RECONNECTS', '10')),
            ping_interval=int(os.getenv('WEBSOCKET_PING_INTERVAL', '30')),
            ping_timeout=int(os.getenv('WEBSOCKET_PING_TIMEOUT', '10')),
            api_key=os.getenv('WEBSOCKET_API_KEY'),
            api_secret=os.getenv('WEBSOCKET_API_SECRET'),
            enable_pattern_detection=os.getenv('ENABLE_PATTERN_DETECTION', 'true').lower() == 'true',
            candle_interval_minutes=int(os.getenv('CANDLE_INTERVAL_MINUTES', '1'))
        )


# Predefined configurations for common WebSocket providers
ALPACA_PAPER_CONFIG = TradingWebSocketConfig(
    url="wss://paper-api.alpaca.markets/stream",
    symbols=["AAPL", "GOOGL", "MSFT"],
    subscription_format="trades.{symbol}"
)

POLYGON_CONFIG = TradingWebSocketConfig(
    url="wss://socket.polygon.io/stocks",
    symbols=["AAPL", "GOOGL", "MSFT"],
    subscription_format="T.{symbol}"
)

# Local development/testing config
LOCAL_DEV_CONFIG = TradingWebSocketConfig(
    url="ws://localhost:8080/ws",
    symbols=["AGH", "AAPL", "GOOGL"],
    reconnect_interval=2,
    max_reconnects=5
)


class WebSocketConfigManager:
    """Manage WebSocket configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, name: str, config: TradingWebSocketConfig):
        """Save a named configuration"""
        config_path = os.path.join(self.config_dir, f"{name}.json")
        config.save_to_file(config_path)
    
    def load_config(self, name: str) -> TradingWebSocketConfig:
        """Load a named configuration"""
        config_path = os.path.join(self.config_dir, f"{name}.json")
        return TradingWebSocketConfig.from_file(config_path)
    
    def list_configs(self) -> List[str]:
        """List available configuration names"""
        configs = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                configs.append(filename[:-5])  # Remove .json extension
        return configs
    
    def delete_config(self, name: str):
        """Delete a named configuration"""
        config_path = os.path.join(self.config_dir, f"{name}.json")
        if os.path.exists(config_path):
            os.remove(config_path)
    
    def create_sample_configs(self):
        """Create sample configuration files"""
        self.save_config("alpaca_paper", ALPACA_PAPER_CONFIG)
        self.save_config("polygon", POLYGON_CONFIG)
        self.save_config("local_dev", LOCAL_DEV_CONFIG)


def create_custom_config(url: str, symbols: List[str], **kwargs) -> TradingWebSocketConfig:
    """Create a custom WebSocket configuration"""
    return TradingWebSocketConfig(
        url=url,
        symbols=symbols,
        **kwargs
    )