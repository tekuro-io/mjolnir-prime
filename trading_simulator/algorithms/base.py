"""
Base classes and interfaces for trading algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.models import CandlestickTick, PatternMatch, Trade
from ..core.types import PatternType


class TradingAlgorithm(ABC):
    """Base class for trading algorithms"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.is_active = True
        self.trades_made = 0
        self.last_trade_time: Optional[datetime] = None
        
    @abstractmethod
    def on_tick(self, tick: CandlestickTick, patterns: List[PatternMatch]) -> None:
        """Called when new tick is processed"""
        pass
    
    @abstractmethod
    def on_pattern(self, pattern: PatternMatch) -> None:
        """Called when new pattern is detected"""
        pass
    
    def on_trade_executed(self, trade: Trade) -> None:
        """Called when a trade is executed (optional override)"""
        self.trades_made += 1
        self.last_trade_time = trade.timestamp
    
    def get_status(self) -> Dict[str, Any]:
        """Get algorithm status"""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'trades_made': self.trades_made,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'parameters': self.parameters
        }
    
    def activate(self):
        """Activate the algorithm"""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate the algorithm"""
        self.is_active = False


class PatternBasedAlgorithm(TradingAlgorithm):
    """Base class for pattern-based algorithms"""
    
    def __init__(self, name: str, target_patterns: List[PatternType], 
                 confidence_threshold: float = 0.6, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.target_patterns = target_patterns
        self.confidence_threshold = confidence_threshold
        self.pattern_history: List[PatternMatch] = []
    
    def on_tick(self, tick: CandlestickTick, patterns: List[PatternMatch]) -> None:
        """Process tick and filter relevant patterns"""
        if not self.is_active:
            return
            
        relevant_patterns = [
            p for p in patterns 
            if p.pattern_type in self.target_patterns and p.confidence >= self.confidence_threshold
        ]
        
        for pattern in relevant_patterns:
            self.pattern_history.append(pattern)
            self.on_pattern(pattern)
    
    @abstractmethod
    def on_pattern(self, pattern: PatternMatch) -> None:
        """Handle relevant pattern detection"""
        pass


class MomentumAlgorithm(TradingAlgorithm):
    """Base class for momentum-based algorithms"""
    
    def __init__(self, name: str, lookback_periods: int = 10, 
                 momentum_threshold: float = 0.02, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
        self.price_history: Dict[str, List[float]] = {}
    
    def calculate_momentum(self, symbol: str, current_price: float) -> float:
        """Calculate momentum for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(current_price)
        
        # Keep only the required lookback periods
        if len(self.price_history[symbol]) > self.lookback_periods:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_periods:]
        
        if len(self.price_history[symbol]) < 2:
            return 0.0
        
        # Simple momentum calculation
        start_price = self.price_history[symbol][0]
        return (current_price - start_price) / start_price


class MeanReversionAlgorithm(TradingAlgorithm):
    """Base class for mean reversion algorithms"""
    
    def __init__(self, name: str, lookback_periods: int = 20, 
                 deviation_threshold: float = 2.0, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.lookback_periods = lookback_periods
        self.deviation_threshold = deviation_threshold
        self.price_history: Dict[str, List[float]] = {}
    
    def calculate_mean_deviation(self, symbol: str, current_price: float) -> float:
        """Calculate how far current price deviates from mean"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(current_price)
        
        if len(self.price_history[symbol]) > self.lookback_periods:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_periods:]
        
        if len(self.price_history[symbol]) < 3:
            return 0.0
        
        prices = self.price_history[symbol]
        mean_price = sum(prices) / len(prices)
        std_dev = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (current_price - mean_price) / std_dev


class SimpleBreakoutAlgorithm(PatternBasedAlgorithm):
    """Simple implementation of a breakout algorithm"""
    
    def __init__(self, engine, quantity: int = 100):
        super().__init__(
            name="SimpleBreakout",
            target_patterns=[PatternType.FLAT_TOP_BREAKOUT],
            confidence_threshold=0.7,
            parameters={'quantity': quantity}
        )
        self.engine = engine
        self.quantity = quantity
    
    def on_pattern(self, pattern: PatternMatch) -> None:
        """Execute breakout strategy"""
        print(f"🚀 {self.name}: Breakout detected for {pattern.symbol}")
        print(f"   Confidence: {pattern.confidence:.2%}")
        
        try:
            from ..core.types import OrderSide
            order_id = self.engine.place_market_order(
                pattern.symbol, OrderSide.BUY, self.quantity
            )
            print(f"   Order placed: {order_id}")
        except Exception as e:
            print(f"   Error placing order: {e}")


class SimpleMomentumAlgorithm(MomentumAlgorithm):
    """Simple momentum algorithm implementation"""
    
    def __init__(self, engine, quantity: int = 50):
        super().__init__(
            name="SimpleMomentum",
            lookback_periods=5,
            momentum_threshold=0.01,
            parameters={'quantity': quantity}
        )
        self.engine = engine
        self.quantity = quantity
    
    def on_tick(self, tick: CandlestickTick, patterns: List[PatternMatch]) -> None:
        """Check momentum on each tick"""
        if not self.is_active:
            return
        
        momentum = self.calculate_momentum(tick.symbol, tick.close)
        
        if momentum > self.momentum_threshold:
            print(f"📈 {self.name}: Strong momentum for {tick.symbol}: {momentum:.2%}")
            
            try:
                from ..core.types import OrderSide
                order_id = self.engine.place_market_order(
                    tick.symbol, OrderSide.BUY, self.quantity
                )
                print(f"   Momentum order: {order_id}")
            except Exception as e:
                print(f"   Error placing order: {e}")
    
    def on_pattern(self, pattern: PatternMatch) -> None:
        """Not used in this momentum algorithm"""
        pass


class AlgorithmManager:
    """Manages multiple trading algorithms"""
    
    def __init__(self, engine):
        self.engine = engine
        self.algorithms: Dict[str, TradingAlgorithm] = {}
        
    def add_algorithm(self, algorithm: TradingAlgorithm):
        """Add an algorithm to the manager"""
        self.algorithms[algorithm.name] = algorithm
        
        # Register the algorithm with the engine
        def combined_callback(tick, patterns):
            algorithm.on_tick(tick, patterns)
            
        self.engine.register_algorithm(combined_callback)
    
    def remove_algorithm(self, name: str):
        """Remove an algorithm"""
        if name in self.algorithms:
            self.algorithms[name].deactivate()
            del self.algorithms[name]
    
    def activate_algorithm(self, name: str):
        """Activate a specific algorithm"""
        if name in self.algorithms:
            self.algorithms[name].activate()
    
    def deactivate_algorithm(self, name: str):
        """Deactivate a specific algorithm"""
        if name in self.algorithms:
            self.algorithms[name].deactivate()
    
    def get_algorithm_status(self) -> Dict[str, Dict]:
        """Get status of all algorithms"""
        return {name: algo.get_status() for name, algo in self.algorithms.items()}
    
    def get_active_algorithms(self) -> List[str]:
        """Get list of active algorithm names"""
        return [name for name, algo in self.algorithms.items() if algo.is_active]
    
    def activate_all(self):
        """Activate all algorithms"""
        for algo in self.algorithms.values():
            algo.activate()
    
    def deactivate_all(self):
        """Deactivate all algorithms"""
        for algo in self.algorithms.values():
            algo.deactivate()


class BacktestRunner:
    """Run backtests on algorithms"""
    
    def __init__(self, engine):
        self.engine = engine
        self.results = {}
    
    def run_backtest(self, algorithm: TradingAlgorithm, 
                    candlesticks: List[CandlestickTick], 
                    initial_balance: float = 100000) -> Dict[str, Any]:
        """Run a backtest on historical data"""
        # Reset engine state
        self.engine.reset()
        self.engine.portfolio.cash = initial_balance
        self.engine.portfolio.initial_balance = initial_balance
        
        # Clear any existing algorithm callbacks to prevent duplicates
        self.engine.algorithm_callbacks.clear()
        
        # Add algorithm (this registers it with the engine)
        algorithm.activate()
        
        # Register the algorithm callback properly
        def algorithm_callback(tick, patterns):
            algorithm.on_tick(tick, patterns)
        
        self.engine.register_algorithm(algorithm_callback)
        
        start_time = datetime.now()
        
        # Process each candlestick
        for candle in candlesticks:
            try:
                self.engine.process_tick(candle)
            except Exception as e:
                print(f"Error processing candle: {e}")
                continue
        
        end_time = datetime.now()
        
        # Calculate results
        final_portfolio_value = self.engine.portfolio.get_portfolio_value(self.engine.current_prices)
        total_return = (final_portfolio_value - initial_balance) / initial_balance
        
        results = {
            'algorithm_name': algorithm.name,
            'initial_balance': initial_balance,
            'final_balance': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': len(self.engine.executed_orders),
            'patterns_detected': len(self.engine.detected_patterns),
            'execution_time': (end_time - start_time).total_seconds(),
            'candles_processed': len(candlesticks),
            'final_positions': {
                symbol: {'quantity': pos.quantity, 'avg_cost': pos.avg_cost}
                for symbol, pos in self.engine.portfolio.positions.items()
                if pos.quantity > 0
            }
        }
        
        self.results[algorithm.name] = results
        return results
    
    def compare_algorithms(self, algorithms: List[TradingAlgorithm], 
                          candlesticks: List[CandlestickTick]) -> Dict[str, Any]:
        """Compare multiple algorithms on the same data"""
        comparison_results = {}
        
        for algorithm in algorithms:
            print(f"Running backtest for {algorithm.name}...")
            results = self.run_backtest(algorithm, candlesticks)
            comparison_results[algorithm.name] = results
        
        # Sort by total return
        sorted_algos = sorted(
            comparison_results.items(), 
            key=lambda x: x[1]['total_return'], 
            reverse=True
        )
        
        return {
            'individual_results': comparison_results,
            'ranked_by_return': [(name, data['total_return']) for name, data in sorted_algos],
            'best_algorithm': sorted_algos[0][0] if sorted_algos else None
        }