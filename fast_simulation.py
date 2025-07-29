"""
Fast Simulation Mode for Trading Algorithm Testing

This module reads tick data from JSON Lines files and processes them through
the trading algorithm at high speed for backtesting purposes.
"""

import asyncio
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from trading_simulator.realtime.trading_engine import RealTimeTradingEngine
from trading_simulator.trading.engine import TradeEngine
from trading_simulator.portfolio.portfolio import Portfolio
from trading_simulator.patterns.detector import PatternDetector
from trading_simulator.config.websocket_config import TradingWebSocketConfig
from trading_simulator.patterns.strategies import StrategyFactory
from trading_simulator.core.models import CandlestickTick, PatternMatch
from trading_simulator.core.types import PatternType, OrderSide
from trading_simulator.data.websocket_client import TickData
from strategy_config import StrategyConfig, ConfigurableStrategy, get_preset_config, list_presets, create_strategy_from_config


@dataclass
class PatternEvent:
    """Records a pattern detection event"""
    timestamp: datetime
    symbol: str
    pattern_type: PatternType
    confidence: float
    trigger_price: float
    action_taken: str  # "buy", "sell", "none"
    quantity: int = 0
    order_id: str = ""


@dataclass
class TradeEvent:
    """Records a trade execution event"""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    price: float
    order_id: str
    reason: str  # "pattern_detected", "position_close", etc.


class FastSimulationEngine:
    """Fast simulation engine that processes tick data from files"""
    
    def __init__(self, tick_file_path: str, speed_multiplier: int = 1, strategy_name: str = "conservative"):
        """
        Initialize fast simulation engine
        
        Args:
            tick_file_path: Path to JSON Lines file containing tick data
            speed_multiplier: How many times faster than real-time (1 = real-time)
            strategy_name: Name of preset strategy to use ('conservative', 'aggressive', 'scalper', 'hodler')
        """
        self.tick_file_path = tick_file_path
        self.speed_multiplier = speed_multiplier
        self.strategy_name = strategy_name
        
        # Verify file exists
        if not os.path.exists(tick_file_path):
            raise FileNotFoundError(f"Tick data file not found: {tick_file_path}")
        
        # Trading components
        self.trade_engine: Optional[TradeEngine] = None
        self.configurable_strategy: Optional[ConfigurableStrategy] = None
        
        # Event tracking
        self.pattern_events: List[PatternEvent] = []
        self.trade_events: List[TradeEvent] = []
        
        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'candles_generated': 0,
            'patterns_detected': 0,
            'trades_executed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'simulation_start_time': None,
            'simulation_end_time': None,
            'data_start_time': None,
            'data_end_time': None,
            'processing_time_seconds': 0,
            'symbols_processed': set()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def setup_trading_engine(self, initial_balance: float = 100000):
        """Setup trading engine and configurable strategies for simulation"""
        self.logger.info(f"Setting up trading engine with '{self.strategy_name}' strategy...")
        
        # Get the strategy configuration
        try:
            strategy_config = get_preset_config(self.strategy_name)
            self.logger.info(f"Using strategy: {strategy_config.name} - {strategy_config.description}")
        except ValueError as e:
            self.logger.error(f"Invalid strategy name: {e}")
            available = list_presets()
            self.logger.error(f"Available strategies: {', '.join(available)}")
            raise
        
        # We'll determine symbols from the file data
        initial_symbols = ["TEMP"]  # Placeholder, will be updated from file
        
        # Create portfolio and pattern detector
        portfolio = Portfolio(initial_balance=initial_balance)
        pattern_detector = PatternDetector(tolerance=0.01, lookback_window=50)
        
        # Create trading engine
        self.trade_engine = TradeEngine(
            portfolio=portfolio,
            pattern_detector=pattern_detector,
            symbols=initial_symbols
        )
        
        # Create configurable strategy
        self.configurable_strategy = create_strategy_from_config(strategy_config, self.trade_engine)
        
        # Link simulation engine to trade engine for pattern action tracking
        self.trade_engine.simulation_engine = self
        
        # Add pattern detection event tracking
        def pattern_event_tracker(pattern: PatternMatch):
            self.logger.info(f"PATTERN DETECTED: {pattern.pattern_type.value} for {pattern.symbol} - confidence: {pattern.confidence:.2f}")
            
            # Record pattern event
            pattern_event = PatternEvent(
                timestamp=pattern.timestamp,
                symbol=pattern.symbol,
                pattern_type=pattern.pattern_type,
                confidence=pattern.confidence,
                trigger_price=pattern.trigger_price,
                action_taken="pending"  # Will be updated when action is taken
            )
            self.pattern_events.append(pattern_event)
            self.stats['patterns_detected'] += 1
            
        # Register pattern callback - first the tracker, then the strategy
        def combined_pattern_callback(pattern: PatternMatch):
            pattern_event_tracker(pattern)  # Track the event
            
            # DEBUG: Log strategy execution attempt
            self.logger.info(f"🔍 STRATEGY CHECK: {pattern.pattern_type.value} confidence {pattern.confidence:.2f} vs threshold {strategy_config.buy_confidence_threshold:.2f}")
            
            self.configurable_strategy.on_pattern_detected(pattern)  # Execute strategy
            
        # Register pattern callback for the strategy's pattern type
        self.trade_engine.pattern_detector.register_pattern_callback(
            strategy_config.buy_pattern_type, 
            combined_pattern_callback
        )
        
        self.logger.info(f"Trading engine initialized with ${initial_balance:,.2f} balance")
        self.logger.info(f"Strategy config: Buy on {strategy_config.buy_pattern_type.value} @ {strategy_config.buy_confidence_threshold:.0%} confidence")
        self.logger.info(f"Sell conditions: {strategy_config.sell_after_candles} candles, Red candle: {strategy_config.sell_on_red_candle}")
        if strategy_config.stop_loss_percent > 0:
            self.logger.info(f"Stop loss: {strategy_config.stop_loss_percent:.1%}")
        else:
            self.logger.info("Stop loss: Disabled")
    
    def read_tick_data(self) -> List[Dict]:
        """Read and parse tick data from JSON Lines file"""
        self.logger.info(f"Reading tick data from: {self.tick_file_path}")
        
        tick_data = []
        symbols = set()
        
        try:
            with open(self.tick_file_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Validate expected format
                        if 'topic' in data and 'data' in data:
                            tick_info = data['data']
                            if all(key in tick_info for key in ['ticker', 'timestamp', 'price']):
                                tick_data.append(tick_info)
                                symbols.add(tick_info['ticker'])
                            else:
                                self.logger.warning(f"Line {line_num}: Missing required fields in tick data")
                        else:
                            self.logger.warning(f"Line {line_num}: Invalid JSON structure")
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Line {line_num}: JSON decode error: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error reading tick file: {e}")
            raise
        
        if not tick_data:
            raise ValueError("No valid tick data found in file")
        
        # Sort by timestamp
        tick_data.sort(key=lambda x: x['timestamp'])
        
        # Update symbols in trading engine
        if self.trade_engine:
            self.trade_engine.symbols = list(symbols)
        
        self.stats['symbols_processed'] = symbols
        self.logger.info(f"Loaded {len(tick_data)} ticks for {len(symbols)} symbols: {', '.join(sorted(symbols))}")
        
        return tick_data
    
    def _aggregate_to_candles(self, tick_data: List[Dict], candle_duration_ms: int = 60000) -> List[Dict]:
        """
        Aggregate tick data into OHLC candles
        
        Args:
            tick_data: List of tick data points
            candle_duration_ms: Duration of each candle in milliseconds
            
        Returns:
            List of OHLC candle data
        """
        if not tick_data:
            return []
        
        candles = []
        symbol_candles = {}  # Track current candle for each symbol
        
        for tick in tick_data:
            symbol = tick['ticker']
            timestamp = tick['timestamp']
            price = tick['price']
            
            # Determine candle boundary (minute-based)
            candle_start = (timestamp // candle_duration_ms) * candle_duration_ms
            
            # Initialize symbol tracking if needed
            if symbol not in symbol_candles:
                symbol_candles[symbol] = {}
            
            # Check if we need to complete previous candle and start new one
            if candle_start not in symbol_candles[symbol]:
                # Complete previous candle if exists
                for prev_start, prev_candle in symbol_candles[symbol].items():
                    if prev_start < candle_start:
                        candles.append(prev_candle)
                
                # Clear old candles and start new one
                symbol_candles[symbol] = {
                    candle_start: {
                        'symbol': symbol,
                        'timestamp': candle_start / 1000,  # Convert to seconds
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': 100,  # Mock volume
                        'tick_count': 1
                    }
                }
            else:
                # Update existing candle
                candle = symbol_candles[symbol][candle_start]
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
                candle['volume'] += 100
                candle['tick_count'] += 1
        
        # Add remaining candles
        for symbol_candle_dict in symbol_candles.values():
            for candle in symbol_candle_dict.values():
                candles.append(candle)
        
        # Sort by timestamp
        candles.sort(key=lambda x: x['timestamp'])
        
        return candles
    
    async def run_simulation(self, enable_detailed_logging: bool = False) -> Dict[str, Any]:
        """
        Run the fast simulation with file-based tick data
        
        Args:
            enable_detailed_logging: Enable detailed logging for debugging
            
        Returns:
            Simulation results and statistics
        """
        if not self.trade_engine:
            self.setup_trading_engine()
        
        # Read tick data from file
        tick_data = self.read_tick_data()
        
        # Record data timeframe
        self.stats['data_start_time'] = datetime.fromtimestamp(tick_data[0]['timestamp'] / 1000)
        self.stats['data_end_time'] = datetime.fromtimestamp(tick_data[-1]['timestamp'] / 1000)
        
        self.logger.info(f"Processing {len(tick_data)} ticks from {self.stats['data_start_time']} to {self.stats['data_end_time']}")
        
        # Aggregate into candles
        candles = self._aggregate_to_candles(tick_data)
        self.stats['candles_generated'] = len(candles)
        
        self.logger.info(f"Generated {len(candles)} candles for processing")
        
        self.stats['simulation_start_time'] = time.time()
        
        # Process candles through trading engine
        for i, candle_data in enumerate(candles):
            # Create CandlestickTick object
            candle = CandlestickTick(
                symbol=candle_data['symbol'],
                timestamp=datetime.fromtimestamp(candle_data['timestamp']),
                open=candle_data['open'],
                high=candle_data['high'],
                low=candle_data['low'],
                close=candle_data['close'],
                volume=candle_data['volume']
            )
            
            # Process through trade engine
            self.trade_engine.process_tick(candle)
            self.stats['ticks_processed'] += 1
            
            # Trigger configurable strategy monitoring
            if self.configurable_strategy:
                self.configurable_strategy.on_candle_completed(candle)
            
            # Record trade events
            recent_trades = self.trade_engine.portfolio.trades[-10:]  # Check last 10 trades
            for trade in recent_trades:
                if not any(te.order_id == trade.id for te in self.trade_events):
                    trade_event = TradeEvent(
                        timestamp=trade.timestamp,
                        symbol=trade.symbol,
                        side=trade.side.value,
                        quantity=trade.quantity,
                        price=trade.price,
                        order_id=trade.id,
                        reason="strategy_execution"
                    )
                    self.trade_events.append(trade_event)
                    self.stats['trades_executed'] += 1
                    
                    if trade.side == OrderSide.BUY:
                        self.stats['positions_opened'] += 1
                    else:
                        self.stats['positions_closed'] += 1
            
            # Pattern actions are now updated directly when trades execute
            
            if enable_detailed_logging and i % 100 == 0:
                self.logger.info(f"Processed {i+1}/{len(candles)} candles")
            
            # Speed control
            if self.speed_multiplier > 1:
                await asyncio.sleep(0.001 / self.speed_multiplier)  # Very fast processing
        
        # FORCE CLOSE ALL POSITIONS - Fix the position selling bug
        self.logger.info("Simulation complete - forcing closure of all open positions...")
        
        portfolio_summary = self.trade_engine.get_portfolio_summary()
        open_positions = portfolio_summary.get('positions', {})
        
        # Close portfolio positions
        positions_closed = 0
        for symbol, position in open_positions.items():
            if position.get('quantity', 0) > 0:
                try:
                    current_price = self.trade_engine.current_prices.get(symbol, position.get('current_price', position.get('avg_cost', 0)))
                    order_id = self.trade_engine.place_market_order(symbol, OrderSide.SELL, position['quantity'])
                    
                    # Record forced closure
                    trade_event = TradeEvent(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        side="sell",
                        quantity=position['quantity'],
                        price=current_price,
                        order_id=order_id,
                        reason="forced_closure"
                    )
                    self.trade_events.append(trade_event)
                    positions_closed += 1
                    self.stats['positions_closed'] += 1
                    self.stats['trades_executed'] += 1
                    
                    self.logger.info(f"FORCED CLOSE: {symbol} - {position['quantity']} shares @ ${current_price:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to close position {symbol}: {e}")
        
        # Close configurable strategy positions
        if self.configurable_strategy and hasattr(self.configurable_strategy, 'active_positions'):
            for symbol in list(self.configurable_strategy.active_positions.keys()):
                try:
                    position = self.configurable_strategy.active_positions[symbol]
                    current_price = self.trade_engine.current_prices.get(symbol, position.get('current_price', position.get('entry_price', 0)))
                    
                    # Use strategy's sell method to properly close
                    self.configurable_strategy._execute_sell(symbol, "simulation_end_forced_closure", current_price)
                    positions_closed += 1
                    
                    self.logger.info(f"CONFIGURABLE STRATEGY FORCED CLOSE: {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to close configurable strategy position {symbol}: {e}")
        
        self.logger.info(f"Forced closure complete - closed {positions_closed} positions")
        
        self.stats['simulation_end_time'] = time.time()
        self.stats['processing_time_seconds'] = self.stats['simulation_end_time'] - self.stats['simulation_start_time']
        
        return self.get_simulation_results()
    
    def _update_pattern_action(self, pattern: PatternMatch, action: str, quantity: int, order_id: str):
        """Update pattern event with executed action"""
        # Find the corresponding pattern event and update it
        for pattern_event in self.pattern_events:
            if (pattern_event.symbol == pattern.symbol and 
                pattern_event.pattern_type == pattern.pattern_type and
                pattern_event.timestamp == pattern.timestamp and
                pattern_event.action_taken == "pending"):
                
                pattern_event.action_taken = action
                pattern_event.quantity = quantity
                pattern_event.order_id = order_id
                self.logger.info(f"✅ Updated pattern action: {pattern.symbol} {action} {quantity} shares")
                break
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results with detailed analytics"""
        if not self.trade_engine:
            return {'error': 'Trading engine not initialized'}
        
        portfolio_summary = self.trade_engine.get_portfolio_summary()
        trading_stats = self.trade_engine.get_trading_statistics()
        
        # Calculate performance metrics
        total_time = self.stats.get('processing_time_seconds', 0)
        data_duration = (self.stats['data_end_time'] - self.stats['data_start_time']).total_seconds() if self.stats.get('data_end_time') else 0
        speedup_achieved = data_duration / max(total_time, 0.001) if total_time > 0 else 0
        
        # Analyze pattern detection
        pattern_analysis = self._analyze_patterns()
        
        # Analyze trades
        trade_analysis = self._analyze_trades()
        
        # Verify all positions are closed
        open_positions = portfolio_summary.get('positions', {})
        remaining_positions = {k: v for k, v in open_positions.items() if v.get('quantity', 0) > 0}
        
        results = {
            'simulation_info': {
                'file_processed': self.tick_file_path,
                'symbols_processed': list(self.stats['symbols_processed']),
                'data_timeframe': {
                    'start': self.stats['data_start_time'].isoformat() if self.stats.get('data_start_time') else None,
                    'end': self.stats['data_end_time'].isoformat() if self.stats.get('data_end_time') else None,
                    'duration_hours': data_duration / 3600 if data_duration else 0
                },
                'processing_stats': {
                    'ticks_processed': self.stats['ticks_processed'],
                    'candles_generated': self.stats['candles_generated'],
                    'processing_time_seconds': total_time,
                    'speedup_achieved': f"{speedup_achieved:.1f}x" if speedup_achieved else "N/A"
                }
            },
            'pattern_analysis': pattern_analysis,
            'trade_analysis': trade_analysis,
            'portfolio_summary': portfolio_summary,
            'position_closure': {
                'positions_opened': self.stats['positions_opened'],
                'positions_closed': self.stats['positions_closed'],
                'remaining_open_positions': len(remaining_positions),
                'all_positions_closed': len(remaining_positions) == 0,
                'remaining_positions': remaining_positions
            },
            'performance': {
                'total_pnl': portfolio_summary.get('total_pnl', 0),
                'realized_pnl': portfolio_summary.get('realized_pnl', 0),
                'unrealized_pnl': portfolio_summary.get('unrealized_pnl', 0),
                'final_balance': portfolio_summary.get('portfolio_value', 0),
                'initial_balance': portfolio_summary.get('initial_balance', 0)
            }
        }
        
        return results
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze pattern detection events"""
        pattern_counts = {}
        pattern_actions = {'buy': 0, 'sell': 0, 'none': 0, 'pending': 0}
        pattern_by_symbol = {}
        pattern_timeline = []
        
        for event in self.pattern_events:
            # Count by pattern type
            pattern_type = event.pattern_type.value
            if pattern_type not in pattern_counts:
                pattern_counts[pattern_type] = 0
            pattern_counts[pattern_type] += 1
            
            # Count actions
            if event.action_taken in pattern_actions:
                pattern_actions[event.action_taken] += 1
            
            # Count by symbol
            if event.symbol not in pattern_by_symbol:
                pattern_by_symbol[event.symbol] = 0
            pattern_by_symbol[event.symbol] += 1
            
            # Timeline entry
            pattern_timeline.append({
                'timestamp': event.timestamp.isoformat(),
                'symbol': event.symbol,
                'pattern_type': pattern_type,
                'confidence': event.confidence,
                'trigger_price': event.trigger_price,
                'action_taken': event.action_taken,
                'quantity': event.quantity
            })
        
        return {
            'total_patterns_detected': len(self.pattern_events),
            'patterns_by_type': pattern_counts,
            'patterns_by_symbol': pattern_by_symbol,
            'actions_taken': pattern_actions,
            'pattern_timeline': pattern_timeline
        }
    
    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze trade execution events"""
        buy_trades = [t for t in self.trade_events if t.side == 'buy']
        sell_trades = [t for t in self.trade_events if t.side == 'sell']
        
        # Match buy/sell pairs for P&L calculation
        winning_trades = 0
        total_pairs = 0
        
        # Simple matching: pair each buy with next sell for same symbol
        trades_by_symbol = {}
        for trade in self.trade_events:
            if trade.symbol not in trades_by_symbol:
                trades_by_symbol[trade.symbol] = {'buys': [], 'sells': []}
            trades_by_symbol[trade.symbol][f"{trade.side}s"].append(trade)
        
        for symbol, trades in trades_by_symbol.items():
            buys = sorted(trades['buys'], key=lambda x: x.timestamp)
            sells = sorted(trades['sells'], key=lambda x: x.timestamp)
            
            for i in range(min(len(buys), len(sells))):
                buy_price = buys[i].price
                sell_price = sells[i].price
                if sell_price > buy_price:
                    winning_trades += 1
                total_pairs += 1
        
        win_rate = (winning_trades / max(total_pairs, 1)) * 100
        
        return {
            'total_trades': len(self.trade_events),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'completed_pairs': total_pairs,
            'winning_trades': winning_trades,
            'losing_trades': total_pairs - winning_trades,
            'win_rate_percent': win_rate,
            'trades_timeline': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'symbol': t.symbol,
                    'side': t.side,
                    'quantity': t.quantity,
                    'price': t.price,
                    'reason': t.reason
                } for t in self.trade_events
            ]
        }
    
    def print_simulation_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of simulation results"""
        sim_info = results['simulation_info']
        pattern_analysis = results['pattern_analysis']
        trade_analysis = results['trade_analysis']
        performance = results['performance']
        position_closure = results['position_closure']
        
        print("\n" + "="*80)
        print("FAST SIMULATION RESULTS")
        print("="*80)
        
        # File and timing info
        print(f"File: {os.path.basename(sim_info['file_processed'])}")
        timeframe = sim_info['data_timeframe']
        print(f"Data period: {timeframe['start']} to {timeframe['end']}")
        print(f"Duration: {timeframe['duration_hours']:.2f} hours")
        print(f"Processing: {sim_info['processing_stats']['processing_time_seconds']:.1f}s ({sim_info['processing_stats']['speedup_achieved']} speedup)")
        print(f"Symbols: {', '.join(sim_info['symbols_processed'])}")
        print(f"Ticks processed: {sim_info['processing_stats']['ticks_processed']:,}")
        print(f"Candles generated: {sim_info['processing_stats']['candles_generated']:,}")
        
        print("\n" + "-"*50)
        print("PATTERN DETECTION")
        print("-"*50)
        print(f"Total patterns detected: {pattern_analysis['total_patterns_detected']}")
        for pattern_type, count in pattern_analysis['patterns_by_type'].items():
            print(f"  {pattern_type}: {count}")
        print("Top symbols by patterns:")
        sorted_symbols = sorted(pattern_analysis['patterns_by_symbol'].items(), key=lambda x: x[1], reverse=True)
        for symbol, count in sorted_symbols[:5]:
            print(f"  {symbol}: {count} patterns")
        
        print("\n" + "-"*50)
        print("TRADING ACTIVITY")
        print("-"*50)
        print(f"Total trades: {trade_analysis['total_trades']} ({trade_analysis['buy_trades']} buys, {trade_analysis['sell_trades']} sells)")
        print(f"Completed trade pairs: {trade_analysis['completed_pairs']}")
        print(f"Winning trades: {trade_analysis['winning_trades']}")
        print(f"Losing trades: {trade_analysis['losing_trades']}")
        print(f"Win rate: {trade_analysis['win_rate_percent']:.1f}%")
        
        print("\n" + "-"*50)
        print("POSITION MANAGEMENT")
        print("-"*50)
        print(f"Positions opened: {position_closure['positions_opened']}")
        print(f"Positions closed: {position_closure['positions_closed']}")
        print(f"All positions closed: {'✓ YES' if position_closure['all_positions_closed'] else '✗ NO'}")
        if position_closure['remaining_open_positions'] > 0:
            print(f"WARNING: {position_closure['remaining_open_positions']} positions still open!")
            for symbol, pos in position_closure['remaining_positions'].items():
                print(f"  {symbol}: {pos.get('quantity', 0)} shares")
        
        print("\n" + "-"*50)
        print("PORTFOLIO PERFORMANCE")
        print("-"*50)
        print(f"Initial balance: ${performance['initial_balance']:,.2f}")
        print(f"Final balance: ${performance['final_balance']:,.2f}")
        print(f"Total P&L: ${performance['total_pnl']:+,.2f}")
        print(f"Realized P&L: ${performance['realized_pnl']:+,.2f}")
        print(f"Unrealized P&L: ${performance['unrealized_pnl']:+,.2f}")
        
        print("="*80)


# CLI Interface
async def main():
    """Main CLI interface for running file-based simulations"""
    import sys
    import argparse
    
    print("Fast Trading Simulation Engine - File-Based Backtesting")
    print("="*60)
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run backtesting simulation on tick data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies:
{chr(10).join(f'  {name}: {get_preset_config(name).description}' for name in list_presets())}

Examples:
  python fast_simulation.py data/ticks.jsonl
  python fast_simulation.py data/ticks.jsonl --speed 100 --strategy aggressive
  python fast_simulation.py data/ticks.jsonl -s conservative --balance 50000
        """
    )
    
    parser.add_argument('tick_file', help='Path to JSON Lines tick data file')
    parser.add_argument('--speed', '-x', type=int, default=60, 
                       help='Speed multiplier (default: 60x)')
    parser.add_argument('--strategy', '-s', default='conservative',
                       choices=list_presets(),
                       help='Trading strategy to use (default: conservative)')
    parser.add_argument('--balance', '-b', type=float, default=100000,
                       help='Initial balance (default: $100,000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable detailed logging')
    
    # Handle both old-style and new-style arguments
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
        # Check if using old format: file [speed]
        if len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2].isdigit()):
            # Convert old format to new format
            args = parser.parse_args([
                sys.argv[1],  # tick_file
                '--speed', sys.argv[2] if len(sys.argv) > 2 else '60'
            ])
        else:
            args = parser.parse_args()
    else:
        args = parser.parse_args()
    
    try:
        # Display selected configuration
        print(f"File: {args.tick_file}")
        print(f"Speed: {args.speed}x")
        print(f"Strategy: {args.strategy}")
        print(f"Initial balance: ${args.balance:,.2f}")
        
        # Show strategy details
        strategy_config = get_preset_config(args.strategy)
        print(f"Strategy details: {strategy_config.description}")
        print(f"  Buy: {strategy_config.buy_pattern_type.value} @ {strategy_config.buy_confidence_threshold:.0%} confidence")
        print(f"  Sell: After {strategy_config.sell_after_candles} candles, Red candle: {strategy_config.sell_on_red_candle}")
        if strategy_config.stop_loss_percent > 0:
            print(f"  Stop loss: {strategy_config.stop_loss_percent:.1%}")
        
        print("\nStarting simulation...")
        
        simulation = FastSimulationEngine(
            tick_file_path=args.tick_file,
            speed_multiplier=args.speed,
            strategy_name=args.strategy
        )
        
        simulation.setup_trading_engine(initial_balance=args.balance)
        results = await simulation.run_simulation(enable_detailed_logging=args.verbose)
        simulation.print_simulation_summary(results)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError running simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())