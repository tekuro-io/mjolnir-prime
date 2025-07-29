"""
Configurable Strategy System for Trading Simulations

This module provides a flexible framework for defining and testing trading strategies
with configurable buy/sell conditions, stop losses, and extensible criteria.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import time
import logging
from datetime import datetime

from trading_simulator.core.types import PatternType, OrderSide
from trading_simulator.core.models import PatternMatch, CandlestickTick


class SellTrigger(Enum):
    """Types of sell triggers available"""
    CANDLES_ELAPSED = "candles_elapsed"
    RED_CANDLE = "red_candle"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"


@dataclass
class StrategyConfig:
    """
    Configuration for a trading strategy.
    Designed to be easily extensible for complex conditions.
    """
    # Strategy identification
    name: str = "Default Strategy"
    description: str = "Basic configurable strategy"
    
    # Buy conditions
    buy_pattern_type: PatternType = PatternType.BULLISH_REVERSAL
    buy_confidence_threshold: float = 0.65
    # Future extensibility: buy_volume_min, buy_rsi_range, buy_custom_conditions
    
    # Sell conditions
    sell_after_candles: int = 3
    sell_on_red_candle: bool = True
    stop_loss_percent: float = 0.0  # 0.0 = disabled
    take_profit_percent: float = 0.0  # 0.0 = disabled, future feature
    # Future extensibility: trailing_stop, sell_volume_conditions, sell_custom_conditions
    
    # Position management
    quantity: int = 100
    max_risk_per_trade: float = 0.02  # 2% of portfolio value (or allocation % if use_percentage_allocation=True)
    max_positions: int = 5  # Maximum concurrent positions
    use_percentage_allocation: bool = False  # If True, use max_risk_per_trade as portfolio allocation %
    
    # Risk management
    max_daily_loss: float = 0.05  # 5% daily loss limit (future feature)
    cooldown_after_loss: int = 0  # Minutes to wait after loss (future feature)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create config from dictionary"""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if self.buy_confidence_threshold < 0 or self.buy_confidence_threshold > 1:
            errors.append("buy_confidence_threshold must be between 0 and 1")
        
        if self.sell_after_candles < 1:
            errors.append("sell_after_candles must be at least 1")
            
        if self.stop_loss_percent < 0:
            errors.append("stop_loss_percent cannot be negative")
            
        if self.quantity <= 0:
            errors.append("quantity must be positive")
            
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 1:
            errors.append("max_risk_per_trade must be between 0 and 1")
            
        return errors


class ConfigurableStrategy:
    """
    A trading strategy that can be configured dynamically.
    Supports buy/sell conditions, stop losses, and position management.
    """
    
    def __init__(self, config: StrategyConfig, engine):
        self.config = config
        self.engine = engine
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        
        # Track active positions
        self.active_positions: Dict[str, Dict] = {}
        self.total_positions = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid strategy config: {'; '.join(errors)}")
            
        self.logger.info(f"🔧 Initialized strategy: {config.name}")
        self.logger.info(f"   Buy: {config.buy_pattern_type.value} @ {config.buy_confidence_threshold:.0%} confidence")
        self.logger.info(f"   Sell: {config.sell_after_candles} candles, Red candle: {config.sell_on_red_candle}")
        if config.stop_loss_percent > 0:
            self.logger.info(f"   Stop Loss: {config.stop_loss_percent:.1%}")
    
    def on_pattern_detected(self, pattern: PatternMatch):
        """Handle pattern detection for buy signals"""
        self.logger.info(f"🎯 PATTERN RECEIVED: {pattern.pattern_type.value} @ {pattern.confidence:.2f} for {pattern.symbol}")
        
        # Check if this pattern matches our buy criteria
        if (pattern.pattern_type == self.config.buy_pattern_type and 
            pattern.confidence >= self.config.buy_confidence_threshold):
            
            self.logger.info(f"✅ PATTERN CRITERIA MET: {pattern.pattern_type.value} @ {pattern.confidence:.2f} >= {self.config.buy_confidence_threshold:.2f}")
            
            # Risk management checks
            if not self._can_open_position(pattern.symbol):
                self.logger.info(f"❌ POSITION BLOCKED: Cannot open position for {pattern.symbol}")
                return
                
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(pattern.trigger_price)
            self.logger.info(f"📊 POSITION SIZE: {position_size} shares for {pattern.symbol}")
            
            if position_size > 0:
                self._execute_buy(pattern, position_size)
            else:
                self.logger.info(f"❌ ZERO POSITION SIZE: Cannot buy {pattern.symbol} with calculated size {position_size}")
        else:
            # Log why pattern was rejected
            if pattern.pattern_type != self.config.buy_pattern_type:
                self.logger.info(f"❌ WRONG PATTERN TYPE: Got {pattern.pattern_type.value}, need {self.config.buy_pattern_type.value}")
            elif pattern.confidence < self.config.buy_confidence_threshold:
                self.logger.info(f"❌ LOW CONFIDENCE: {pattern.confidence:.2f} < {self.config.buy_confidence_threshold:.2f} threshold")
    
    def on_candle_completed(self, candle: CandlestickTick):
        """Monitor positions for sell conditions"""
        if candle.symbol not in self.active_positions:
            return
            
        position = self.active_positions[candle.symbol]
        position['candles_seen'] += 1
        
        # Check if it's a red candle
        is_red_candle = candle.close < candle.open
        if is_red_candle:
            position['red_candles_seen'] += 1
        
        # Update current price for stop loss monitoring
        position['current_price'] = candle.close
        
        self.logger.info(f"📊 {candle.symbol}: Candle {position['candles_seen']}, "
                        f"Red: {position['red_candles_seen']}, Price: ${candle.close:.2f}")
        
        # Check sell conditions
        sell_reason = self._check_sell_conditions(position, candle)
        if sell_reason:
            self.logger.info(f"🔥 SELL TRIGGERED for {candle.symbol}: {sell_reason}")
            self._execute_sell(candle.symbol, sell_reason, candle.close)
        else:
            self.logger.info(f"⏳ {candle.symbol}: No sell conditions met yet (need {self.config.sell_after_candles} candles or red candle={self.config.sell_on_red_candle})")
    
    def on_tick_received(self, symbol: str, price: float):
        """Monitor for stop loss on every price tick"""
        if symbol in self.active_positions and self.config.stop_loss_percent > 0:
            position = self.active_positions[symbol]
            stop_loss_price = position['entry_price'] * (1 - self.config.stop_loss_percent)
            
            if price <= stop_loss_price:
                self._execute_sell(symbol, f"stop loss @ {self.config.stop_loss_percent:.1%}", price)
    
    def _can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position"""
        # Check if we already have a position in this symbol
        if symbol in self.active_positions:
            self.logger.debug(f"❌ Already have position in {symbol}")
            return False
            
        # Check maximum positions limit
        if len(self.active_positions) >= self.config.max_positions:
            self.logger.debug(f"❌ Maximum positions reached ({self.config.max_positions})")
            return False
            
        # Future: Check daily loss limits, cooldown periods, etc.
        
        return True
    
    def _calculate_position_size(self, entry_price: float) -> int:
        """Calculate position size based on portfolio percentage allocation"""
        # Get current portfolio value
        portfolio_value = self.engine.portfolio.get_portfolio_value(self.engine.current_prices)
        
        # Use portfolio percentage allocation (max_risk_per_trade acts as allocation percentage)
        position_value = portfolio_value * self.config.max_risk_per_trade
        
        # Calculate quantity based on allocation
        calculated_quantity = int(position_value / entry_price)
        
        # If quantity is configured (not using percentage mode), use smaller value
        if hasattr(self.config, 'use_percentage_allocation') and self.config.use_percentage_allocation:
            return calculated_quantity
        else:
            return min(self.config.quantity, calculated_quantity)
    
    def _execute_buy(self, pattern: PatternMatch, quantity: int):
        """Execute a buy order"""
        try:
            order_id = self.engine.place_market_order(pattern.symbol, OrderSide.BUY, quantity)
            
            # Track the position
            self.active_positions[pattern.symbol] = {
                'order_id': order_id,
                'entry_price': pattern.trigger_price,
                'entry_time': pattern.timestamp,
                'quantity': quantity,
                'candles_seen': 0,
                'red_candles_seen': 0,
                'current_price': pattern.trigger_price,
                'stop_loss_price': pattern.trigger_price * (1 - self.config.stop_loss_percent) if self.config.stop_loss_percent > 0 else 0,
                'pattern_timestamp': pattern.timestamp  # Store for pattern event linking
            }
            
            self.total_positions += 1
            
            # Notify simulation engine about the executed trade
            if hasattr(self.engine, 'simulation_engine') and self.engine.simulation_engine:
                self.engine.simulation_engine._update_pattern_action(pattern, 'buy', quantity, order_id)
            
            self.logger.info("")
            self.logger.info("🎯 BUY SIGNAL EXECUTED:")
            self.logger.info(f"   📈 {pattern.symbol} @ ${pattern.trigger_price:.2f} | {quantity} shares")
            self.logger.info(f"   🎲 Confidence: {pattern.confidence:.0%} | Order: {order_id}")
            if self.config.stop_loss_percent > 0:
                self.logger.info(f"   🛡️ Stop Loss: ${self.active_positions[pattern.symbol]['stop_loss_price']:.2f}")
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute buy for {pattern.symbol}: {e}")
    
    def _check_sell_conditions(self, position: Dict, candle: CandlestickTick) -> Optional[str]:
        """Check if any sell conditions are met"""
        # Condition 1: Candles elapsed
        if position['candles_seen'] >= self.config.sell_after_candles:
            return f"{self.config.sell_after_candles} candles elapsed"
        
        # Condition 2: Red candle (if enabled)
        if (self.config.sell_on_red_candle and 
            position['red_candles_seen'] >= 1):
            return "red candle detected"
        
        # Condition 3: Stop loss (checked on ticks, but also here for safety)
        if (self.config.stop_loss_percent > 0 and 
            candle.close <= position['stop_loss_price']):
            return f"stop loss @ {self.config.stop_loss_percent:.1%}"
        
        # Future conditions: take profit, trailing stop, custom indicators
        
        return None
    
    def _execute_sell(self, symbol: str, reason: str, current_price: float):
        """Execute a sell order"""
        if symbol not in self.active_positions:
            return
            
        position = self.active_positions[symbol]
        
        try:
            order_id = self.engine.place_market_order(symbol, OrderSide.SELL, position['quantity'])
            
            # Calculate P&L
            entry_price = position['entry_price']
            pnl = (current_price - entry_price) * position['quantity']
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            self.daily_pnl += pnl
            
            self.logger.info("")
            self.logger.info("💰 SELL EXECUTED:")
            self.logger.info(f"   📉 {symbol} position CLOSED | Reason: {reason}")
            self.logger.info(f"   🔄 Entry: ${entry_price:.2f} → Exit: ${current_price:.2f} | {position['quantity']} shares")
            self.logger.info(f"   💸 P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Order: {order_id}")
            self.logger.info("")
            
            # Remove position from tracking
            del self.active_positions[symbol]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute sell for {symbol}: {e}")
    
    def get_active_positions(self) -> Dict:
        """Get current active positions"""
        return self.active_positions.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        return {
            'config': self.config.to_dict(),
            'active_positions': len(self.active_positions),
            'total_positions_opened': self.total_positions,
            'daily_pnl': self.daily_pnl,
            'max_positions_limit': self.config.max_positions
        }


# Predefined strategy configurations for quick testing
PRESET_CONFIGS = {
    'conservative': StrategyConfig(
        name="Conservative",
        description="High confidence, long hold, stop loss protection",
        buy_confidence_threshold=0.8,
        sell_after_candles=6,  # Was 5, now 6 to actually hold for 5 completed candles after buy
        sell_on_red_candle=False,
        stop_loss_percent=0.02,  # 2% stop loss
        quantity=50
    ),
    
    'aggressive': StrategyConfig(
        name="Aggressive", 
        description="Lower confidence, quick exit, high volume",
        buy_confidence_threshold=0.6,
        sell_after_candles=3,  # Was 2, now 3 to actually wait for 2 completed candles after buy
        sell_on_red_candle=True,
        stop_loss_percent=0.01,  # 1% stop loss
        quantity=200
    ),
    
    'scalper': StrategyConfig(
        name="Scalper",
        description="Fast in and out, minimal risk",
        buy_confidence_threshold=0.7,
        sell_after_candles=2,  # Was 1, now 2 to actually wait for 1 completed candle after buy
        sell_on_red_candle=True,
        stop_loss_percent=0.005,  # 0.5% stop loss
        quantity=100
    ),
    
    'hodler': StrategyConfig(
        name="Hodler",
        description="Long term holds, no red candle exits",
        buy_confidence_threshold=0.75,
        sell_after_candles=11,  # Was 10, now 11 to actually hold for 10 completed candles after buy
        sell_on_red_candle=False,
        stop_loss_percent=0.05,  # 5% stop loss
        quantity=100
    ),
    
    'breakout_hunter': StrategyConfig(
        name="Breakout Hunter",
        description="Targets flat top breakouts with tight stops",
        buy_pattern_type=PatternType.FLAT_TOP_BREAKOUT,
        buy_confidence_threshold=0.65,
        sell_after_candles=4,  # Hold for 3 candles after breakout
        sell_on_red_candle=True,
        stop_loss_percent=0.015,  # 1.5% stop loss
        quantity=150
    ),
    
    'descending_triangle_trader': StrategyConfig(
        name="Descending Triangle Trader",
        description="Trades descending triangle breakouts with tight risk management",
        buy_pattern_type=PatternType.DESCENDING_TRIANGLE,
        buy_confidence_threshold=0.70,
        sell_after_candles=4,  # Quick exit after breakout confirmation
        sell_on_red_candle=True,
        stop_loss_percent=0.02,  # 2% stop loss
        quantity=110
    ),
    
    'double_bottom_trader': StrategyConfig(
        name="Double Bottom Trader",
        description="Specialized in double bottom reversal patterns",
        buy_pattern_type=PatternType.DOUBLE_BOTTOM,
        buy_confidence_threshold=0.75,
        sell_after_candles=7,  # Hold for 6 candles to capture reversal
        sell_on_red_candle=False,  # Let double bottoms play out
        stop_loss_percent=0.03,  # 3% stop loss
        quantity=120
    ),
    
    'triangle_player': StrategyConfig(
        name="Triangle Player",
        description="Trades ascending triangle breakouts aggressively",
        buy_pattern_type=PatternType.ASCENDING_TRIANGLE,
        buy_confidence_threshold=0.60,
        sell_after_candles=5,  # Hold for 4 candles after triangle breakout
        sell_on_red_candle=True,
        stop_loss_percent=0.02,  # 2% stop loss
        quantity=100
    )
}


def create_strategy_from_config(config: StrategyConfig, engine) -> ConfigurableStrategy:
    """Factory function to create a strategy from configuration"""
    return ConfigurableStrategy(config, engine)


def get_preset_config(preset_name: str) -> StrategyConfig:
    """Get a predefined strategy configuration"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset_name]


def list_presets() -> List[str]:
    """List available preset strategy names"""
    return list(PRESET_CONFIGS.keys())