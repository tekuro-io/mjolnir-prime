"""
Pre-built pattern-based trading strategies.
"""

from typing import Callable, Dict, Any
from ..core.models import PatternMatch
from ..core.types import OrderSide, PatternType


class PatternStrategies:
    """Pre-built pattern-based trading strategies"""
    
    @staticmethod
    def create_flat_top_strategy(engine, confidence_threshold: float = 0.7, 
                               quantity: int = 100, max_risk_per_trade: float = 0.02) -> Callable:
        """
        Create a flat top breakout strategy
        
        Args:
            engine: Trading engine instance
            confidence_threshold: Minimum confidence to trigger trade
            quantity: Number of shares to trade
            max_risk_per_trade: Maximum risk as percentage of portfolio
        """
        def strategy(pattern: PatternMatch):
            if pattern.confidence < confidence_threshold:
                return
            
            # Risk management - don't risk more than max_risk_per_trade of portfolio
            portfolio_value = engine.portfolio.get_portfolio_value(engine.current_prices)
            max_position_value = portfolio_value * max_risk_per_trade
            current_price = engine.current_prices.get(pattern.symbol, pattern.trigger_price)
            
            # Adjust quantity based on risk
            adjusted_quantity = min(quantity, int(max_position_value / current_price))
            
            if adjusted_quantity > 0:
                print(f"🚀 Flat top breakout: {pattern.symbol}")
                print(f"   Confidence: {pattern.confidence:.2f}")
                print(f"   Trigger price: ${pattern.trigger_price:.2f}")
                print(f"   Resistance level: ${pattern.metadata.get('resistance_level', 'N/A'):.2f}")
                print(f"   Trading {adjusted_quantity} shares")
                
                order_id = engine.place_market_order(pattern.symbol, OrderSide.BUY, adjusted_quantity)
                print(f"   Order placed: {order_id}")
        
        return strategy
    
    @staticmethod
    def create_reversal_strategy(engine, confidence_threshold: float = 0.6,
                               quantity: int = 50, use_recovery_strength: bool = True) -> Callable:
        """
        Create a bullish reversal strategy
        
        Args:
            engine: Trading engine instance
            confidence_threshold: Minimum confidence to trigger trade
            quantity: Number of shares to trade
            use_recovery_strength: Whether to adjust quantity based on recovery strength
        """
        def strategy(pattern: PatternMatch):
            if pattern.confidence < confidence_threshold:
                return
            
            recovery_strength = pattern.metadata.get('recovery_strength', 0.5)
            
            # Adjust quantity based on recovery strength if enabled
            if use_recovery_strength:
                adjusted_quantity = int(quantity * max(0.5, recovery_strength))
            else:
                adjusted_quantity = quantity
            
            if adjusted_quantity > 0:
                print(f"📈 Bullish reversal: {pattern.symbol}")
                print(f"   Confidence: {pattern.confidence:.2f}")
                print(f"   Recovery strength: {recovery_strength:.2f}")
                print(f"   Trading {adjusted_quantity} shares")
                
                order_id = engine.place_market_order(pattern.symbol, OrderSide.BUY, adjusted_quantity)
                print(f"   Order placed: {order_id}")
        
        return strategy
    
    @staticmethod
    def create_mean_reversion_strategy(engine, confidence_threshold: float = 0.65,
                                     quantity: int = 75) -> Callable:
        """
        Create a mean reversion strategy for bearish patterns
        """
        def strategy(pattern: PatternMatch):
            if (pattern.pattern_type == PatternType.BEARISH_REVERSAL and 
                pattern.confidence >= confidence_threshold):
                
                print(f"📉 Mean reversion opportunity: {pattern.symbol}")
                print(f"   Confidence: {pattern.confidence:.2f}")
                print(f"   Shorting {quantity} shares")
                
                # In a real system, you'd implement short selling
                # For now, we'll just log the opportunity
                print(f"   Short sell signal logged (not implemented)")
        
        return strategy
    
    @staticmethod
    def create_momentum_strategy(engine, lookback_periods: int = 5,
                               momentum_threshold: float = 0.02) -> Callable:
        """
        Create a momentum-based strategy that combines with patterns
        """
        def algorithm_callback(tick, new_patterns):
            # Calculate simple momentum
            if len(engine.pattern_detector.candle_history.get(tick.symbol, [])) >= lookback_periods:
                candles = list(engine.pattern_detector.candle_history[tick.symbol])
                recent_candles = candles[-lookback_periods:]
                
                price_change = (recent_candles[-1].close - recent_candles[0].close) / recent_candles[0].close
                
                # Strong momentum + pattern confirmation
                for pattern in new_patterns:
                    if (price_change > momentum_threshold and 
                        pattern.pattern_type in [PatternType.FLAT_TOP_BREAKOUT, PatternType.BULLISH_REVERSAL]):
                        
                        print(f"🚀 Momentum + Pattern confirmation: {tick.symbol}")
                        print(f"   Momentum: {price_change:.2%}")
                        print(f"   Pattern: {pattern.pattern_type.value}")
                        
                        order_id = engine.place_market_order(tick.symbol, OrderSide.BUY, 100)
                        print(f"   Momentum order: {order_id}")
        
        return algorithm_callback


class StrategyFactory:
    """Factory class for creating and managing strategies"""
    
    def __init__(self, engine):
        self.engine = engine
        self.active_strategies: Dict[str, Callable] = {}
    
    def add_strategy(self, name: str, strategy_func: Callable, pattern_type: PatternType = None):
        """Add a strategy to the factory"""
        self.active_strategies[name] = strategy_func
        
        if pattern_type:
            self.engine.register_pattern_strategy(pattern_type, strategy_func)
        else:
            # Assume it's an algorithm callback
            self.engine.register_algorithm(strategy_func)
    
    def create_conservative_setup(self):
        """Create a conservative trading setup"""
        # High confidence flat top breakouts
        flat_top = PatternStrategies.create_flat_top_strategy(
            self.engine, 
            confidence_threshold=0.8, 
            quantity=50,
            max_risk_per_trade=0.01
        )
        self.add_strategy("conservative_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        # High confidence reversals
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.75,
            quantity=25
        )
        self.add_strategy("conservative_reversal", reversal, PatternType.BULLISH_REVERSAL)
    
    def create_aggressive_setup(self):
        """Create an aggressive trading setup"""
        # Lower confidence thresholds, higher quantities
        flat_top = PatternStrategies.create_flat_top_strategy(
            self.engine,
            confidence_threshold=0.6,
            quantity=200,
            max_risk_per_trade=0.05
        )
        self.add_strategy("aggressive_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.5,
            quantity=100
        )
        self.add_strategy("aggressive_reversal", reversal, PatternType.BULLISH_REVERSAL)
        
        # Add momentum strategy
        momentum = PatternStrategies.create_momentum_strategy(self.engine)
        self.add_strategy("momentum", momentum)
    
    def create_balanced_setup(self):
        """Create a balanced trading setup"""
        flat_top = PatternStrategies.create_flat_top_strategy(
            self.engine,
            confidence_threshold=0.7,
            quantity=100,
            max_risk_per_trade=0.025
        )
        self.add_strategy("balanced_flat_top", flat_top, PatternType.FLAT_TOP_BREAKOUT)
        
        reversal = PatternStrategies.create_reversal_strategy(
            self.engine,
            confidence_threshold=0.65,
            quantity=75
        )
        self.add_strategy("balanced_reversal", reversal, PatternType.BULLISH_REVERSAL)
        
        # Add mean reversion for balance
        