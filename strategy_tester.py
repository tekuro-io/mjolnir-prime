"""
Interactive Strategy Testing Tool

Allows users to configure and test trading strategies with custom buy/sell conditions,
stop losses, and various simulation parameters.
"""

import asyncio
import sys
import time
from typing import List, Dict, Any
import logging

from strategy_config import (
    StrategyConfig, 
    ConfigurableStrategy, 
    PRESET_CONFIGS, 
    get_preset_config, 
    list_presets
)
from trading_simulator.core.types import PatternType
from fast_simulation import FastSimulationEngine


class StrategyTester:
    """Interactive tool for testing configurable trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def display_welcome(self):
        """Display welcome message and overview"""
        print("\n" + "="*70)
        print("🚀 CONFIGURABLE STRATEGY TESTING TOOL")
        print("="*70)
        print("Test any trading strategy with custom buy/sell conditions!")
        print("\nFeatures:")
        print("• 📊 Pattern-based buy signals (BULLISH_REVERSAL, etc.)")
        print("• ⏱️  Configurable sell conditions (candles, red candle, stop loss)")
        print("• 🛡️  Risk management (position sizing, stop losses)")
        print("• 🎯 Preset strategies or fully custom configurations")
        print("• ⚡ Ultra-fast simulation (3000x+ real-time)")
        print()
    
    def display_presets(self):
        """Display available preset strategies"""
        print("📋 PRESET STRATEGIES:")
        print("-" * 50)
        
        for i, (name, config) in enumerate(PRESET_CONFIGS.items(), 1):
            print(f"{i}. {config.name}")
            print(f"   📝 {config.description}")
            print(f"   🎯 Buy: {config.buy_confidence_threshold:.0%} confidence")
            print(f"   ⏰ Sell: {config.sell_after_candles} candles")
            if config.sell_on_red_candle:
                print(f"   🔴 Red candle exit: YES")
            if config.stop_loss_percent > 0:
                print(f"   🛡️  Stop loss: {config.stop_loss_percent:.1%}")
            print(f"   📈 Quantity: {config.quantity} shares")
            print()
    
    def get_strategy_config(self) -> StrategyConfig:
        """Get strategy configuration from user"""
        print("STRATEGY CONFIGURATION")
        print("=" * 30)
        
        while True:
            print("\nChoose configuration method:")
            print("1. Use preset strategy")
            print("2. Create custom strategy")
            
            choice = input("Enter choice (1-2): ").strip()
            
            if choice == "1":
                return self._get_preset_config()
            elif choice == "2":
                return self._get_custom_config()
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def _get_preset_config(self) -> StrategyConfig:
        """Get a preset strategy configuration"""
        self.display_presets()
        
        presets = list(PRESET_CONFIGS.keys())
        
        while True:
            try:
                choice = int(input(f"Choose preset (1-{len(presets)}): ")) - 1
                if 0 <= choice < len(presets):
                    preset_name = presets[choice]
                    config = get_preset_config(preset_name)
                    print(f"✅ Selected: {config.name}")
                    return config
                else:
                    print(f"❌ Please enter a number between 1 and {len(presets)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _get_custom_config(self) -> StrategyConfig:
        """Get custom strategy configuration from user"""
        print("\n🛠️  CUSTOM STRATEGY BUILDER")
        print("-" * 40)
        
        # Basic info
        name = input("Strategy name (e.g., 'My Strategy'): ").strip() or "Custom Strategy"
        description = input("Description (optional): ").strip() or "User-defined custom strategy"
        
        # Buy conditions
        print("\n📈 BUY CONDITIONS:")
        confidence = self._get_float_input(
            "Confidence threshold (0.0-1.0)", 
            default=0.65, 
            min_val=0.0, 
            max_val=1.0
        )
        
        # Sell conditions
        print("\n📉 SELL CONDITIONS:")
        sell_candles = self._get_int_input(
            "Sell after how many candles?", 
            default=3, 
            min_val=1
        )
        
        sell_red = self._get_bool_input(
            "Sell on red candle? (y/n)", 
            default=True
        )
        
        stop_loss = self._get_float_input(
            "Stop loss percentage (0.0 = disabled)", 
            default=0.0, 
            min_val=0.0, 
            max_val=0.2  # Max 20% stop loss
        )
        
        # Position management
        print("\n💰 POSITION MANAGEMENT:")
        use_percentage = self._get_bool_input(
            "Use portfolio percentage allocation instead of fixed shares? (y/n)", 
            default=False
        )
        
        if use_percentage:
            allocation_percent = self._get_float_input(
                "Portfolio allocation per trade (%)", 
                default=0.1, 
                min_val=0.01, 
                max_val=1.0
            )
            quantity = 100  # Placeholder, will be calculated
            max_risk = allocation_percent
        else:
            quantity = self._get_int_input(
                "Shares per trade", 
                default=100, 
                min_val=1
            )
            
            max_risk = self._get_float_input(
                "Max risk per trade (% of portfolio)", 
                default=0.02, 
                min_val=0.001, 
                max_val=1.0
            )
        
        # Create config
        config = StrategyConfig(
            name=name,
            description=description,
            buy_confidence_threshold=confidence,
            sell_after_candles=sell_candles,
            sell_on_red_candle=sell_red,
            stop_loss_percent=stop_loss,
            quantity=quantity,
            max_risk_per_trade=max_risk,
            use_percentage_allocation=use_percentage
        )
        
        # Validate and display
        errors = config.validate()
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   • {error}")
            return self._get_custom_config()  # Retry
        
        print("\n✅ CUSTOM STRATEGY CREATED:")
        self._display_config_summary(config)
        
        return config
    
    def _get_float_input(self, prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
        """Get float input with validation"""
        while True:
            try:
                value_str = input(f"{prompt} [{default}]: ").strip()
                value = float(value_str) if value_str else default
                
                if min_val is not None and value < min_val:
                    print(f"❌ Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"❌ Value must be <= {max_val}")
                    continue
                    
                return value
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _get_int_input(self, prompt: str, default: int, min_val: int = None, max_val: int = None) -> int:
        """Get integer input with validation"""
        while True:
            try:
                value_str = input(f"{prompt} [{default}]: ").strip()
                value = int(value_str) if value_str else default
                
                if min_val is not None and value < min_val:
                    print(f"❌ Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"❌ Value must be <= {max_val}")
                    continue
                    
                return value
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _get_bool_input(self, prompt: str, default: bool) -> bool:
        """Get boolean input"""
        default_str = "y" if default else "n"
        while True:
            value = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not value:
                return default
            if value in ['y', 'yes', 'true', '1']:
                return True
            elif value in ['n', 'no', 'false', '0']:
                return False
            else:
                print("❌ Please enter y/n")
    
    def _display_config_summary(self, config: StrategyConfig):
        """Display strategy configuration summary"""
        print(f"📋 {config.name}")
        print(f"   📝 {config.description}")
        print(f"   🎯 Buy: {config.buy_confidence_threshold:.0%} confidence on {config.buy_pattern_type.value}")
        print(f"   ⏰ Sell: After {config.sell_after_candles} candles")
        if config.sell_on_red_candle:
            print(f"   🔴 Red candle exit: ENABLED")
        if config.stop_loss_percent > 0:
            print(f"   🛡️  Stop loss: {config.stop_loss_percent:.1%}")
        print(f"   📈 Position: {config.quantity} shares, {config.max_risk_per_trade:.1%} max risk")
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Get simulation parameters from user"""
        print("\n🎮 SIMULATION PARAMETERS")
        print("=" * 30)
        
        # Get JSONL tick data file
        while True:
            tick_file = input("Path to JSONL tick data file: ").strip()
            if not tick_file:
                print("❌ Tick data file path is required")
                continue
            
            # Expand path if needed
            import os
            tick_file = os.path.expanduser(tick_file)
            
            if not os.path.exists(tick_file):
                print(f"❌ File not found: {tick_file}")
                print("   Please check the path and try again")
                continue
            
            if not tick_file.lower().endswith(('.jsonl', '.json')):
                print("❌ File must be a JSON Lines (.jsonl) file")
                continue
                
            break
        
        # Speed multiplier
        speed = self._get_int_input(
            "Speed multiplier (1x = real-time, 60x = 1min/sec)", 
            default=60, 
            min_val=1, 
            max_val=1000
        )
        
        # Initial balance
        balance = self._get_float_input(
            "Initial balance ($)", 
            default=100000, 
            min_val=1000
        )
        
        # Logging detail
        detailed_logging = self._get_bool_input(
            "Enable detailed logging?", 
            default=False
        )
        
        return {
            'tick_file': tick_file,
            'speed': speed,
            'balance': balance,
            'detailed_logging': detailed_logging
        }
    
    async def run_simulation(self, config: StrategyConfig, sim_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the simulation with configured strategy using file-based tick data"""
        print("\n🚀 STARTING SIMULATION...")
        print("=" * 50)
        
        try:
            # Display file information
            import os
            file_size = os.path.getsize(sim_params['tick_file']) / (1024 * 1024)  # MB
            print(f"📁 Processing file: {os.path.basename(sim_params['tick_file'])}")
            print(f"📊 File size: {file_size:.1f} MB")
            print(f"⚡ Speed: {sim_params['speed']}x")
            print(f"💰 Initial balance: ${sim_params['balance']:,.2f}")
            print(f"🎯 Strategy: {config.name}")
            print()
            
            # Create custom strategy name for the simulation
            strategy_name = "custom_interactive"
            
            # Temporarily add the custom config to available strategies
            from strategy_config import PRESET_CONFIGS
            original_presets = PRESET_CONFIGS.copy()
            PRESET_CONFIGS[strategy_name] = config
            
            try:
                # Create file-based simulation engine
                simulation = FastSimulationEngine(
                    tick_file_path=sim_params['tick_file'],
                    speed_multiplier=sim_params['speed'],
                    strategy_name=strategy_name
                )
                
                # Setup trading engine with custom strategy
                simulation.setup_trading_engine(initial_balance=sim_params['balance'])
                
                # Run the simulation
                results = await simulation.run_simulation(
                    enable_detailed_logging=sim_params['detailed_logging']
                )
                
                return results
                
            finally:
                # Restore original presets
                PRESET_CONFIGS.clear()
                PRESET_CONFIGS.update(original_presets)
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            raise
    
    def display_results(self, results: Dict[str, Any], config: StrategyConfig):
        """Display comprehensive simulation results"""
        print("\n" + "="*70)
        print("📊 STRATEGY TESTING RESULTS")
        print("="*70)
        
        sim_info = results['simulation_info']
        pattern_analysis = results.get('pattern_analysis', {})
        trade_analysis = results.get('trade_analysis', {})
        performance = results['performance']
        position_closure = results.get('position_closure', {})
        
        # Strategy info
        print(f"🎯 Strategy: {config.name}")
        print(f"📝 {config.description}")
        print()
        
        # File and simulation overview  
        print("⚡ SIMULATION OVERVIEW:")
        timeframe = sim_info.get('data_timeframe', {})
        if timeframe.get('duration_hours'):
            print(f"   📅 Data period: {timeframe['duration_hours']:.2f} hours")
        print(f"   🏃 Speed: {sim_info['processing_stats']['speedup_achieved']} faster than real-time")
        print(f"   📈 Symbols: {', '.join(sim_info['symbols_processed'])}")
        print(f"   📊 Ticks processed: {sim_info['processing_stats']['ticks_processed']:,}")
        print(f"   📊 Candles generated: {sim_info['processing_stats']['candles_generated']:,}")
        print()
        
        # Pattern detection results
        if pattern_analysis:
            print("🔍 PATTERN DETECTION:")
            print(f"   🎯 Total patterns detected: {pattern_analysis['total_patterns_detected']}")
            for pattern_type, count in pattern_analysis.get('patterns_by_type', {}).items():
                print(f"      • {pattern_type}: {count}")
            print()
        
        # Portfolio performance
        print("💰 PORTFOLIO PERFORMANCE:")
        print(f"   💵 Initial: ${performance['initial_balance']:,.2f}")
        print(f"   💵 Final: ${performance['final_balance']:,.2f}")
        print(f"   📈 Net gain/loss: ${performance['final_balance'] - performance['initial_balance']:+,.2f}")
        if performance['initial_balance'] > 0:
            roi = ((performance['final_balance'] - performance['initial_balance']) / performance['initial_balance']) * 100
            print(f"   📊 ROI: {roi:+.2f}%")
        print(f"   💸 Total P&L: ${performance['total_pnl']:+,.2f}")
        print(f"       • Realized: ${performance['realized_pnl']:+,.2f}")
        print(f"       • Unrealized: ${performance['unrealized_pnl']:+,.2f}")
        print()
        
        # Trading activity
        if trade_analysis:
            print("📊 TRADING ACTIVITY:")
            print(f"   🔄 Total trades: {trade_analysis['total_trades']} ({trade_analysis['completed_pairs']} complete pairs)")
            print(f"   ✅ Winning trades: {trade_analysis['winning_trades']}")
            print(f"   ❌ Losing trades: {trade_analysis['losing_trades']}")
            print(f"   📈 Win rate: {trade_analysis['win_rate_percent']:.1f}%")
            
            if trade_analysis['completed_pairs'] > 0:
                avg_pnl = performance['realized_pnl'] / trade_analysis['completed_pairs']
                print(f"   📊 Average P&L per trade: ${avg_pnl:+.2f}")
            print()
        
        # Position closure verification
        if position_closure:
            print("🔒 POSITION MANAGEMENT:")
            print(f"   📊 Positions opened: {position_closure['positions_opened']}")
            print(f"   📊 Positions closed: {position_closure['positions_closed']}")
            if position_closure['all_positions_closed']:
                print("   ✅ All positions properly closed")
            else:
                print(f"   ⚠️  {position_closure['remaining_open_positions']} positions still open")
            print()
        
        # Open positions (if any remain)
        remaining_positions = position_closure.get('remaining_positions', {})
        if remaining_positions:
            print("🔍 REMAINING OPEN POSITIONS:")
            for symbol, position in remaining_positions.items():
                if position.get('quantity', 0) > 0:
                    print(f"   📈 {symbol}: {position['quantity']} shares @ ${position.get('avg_cost', 0):.2f}")
        
        print("="*70)
    
    async def run_interactive_session(self):
        """Run the main interactive session"""
        try:
            self.display_welcome()
            
            # Get strategy configuration
            config = self.get_strategy_config()
            print(f"\n✅ Strategy configured: {config.name}")
            
            # Get simulation parameters
            sim_params = self.get_simulation_params()
            import os
            file_name = os.path.basename(sim_params['tick_file'])
            print(f"✅ Simulation configured: {file_name} at {sim_params['speed']}x speed")
            
            # Run simulation
            results = await self.run_simulation(config, sim_params)
            
            # Display results
            self.display_results(results, config)
            
            # Ask if user wants to run another test
            if self._get_bool_input("\nRun another test?", default=False):
                await self.run_interactive_session()
            else:
                print("\n👋 Thanks for using the Strategy Tester!")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Session interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()




async def main():
    """Main entry point for strategy tester"""
    tester = StrategyTester()
    await tester.run_interactive_session()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the interactive session
    asyncio.run(main())