"""
Quick test for the configurable strategy system
"""

import asyncio
from strategy_config import StrategyConfig, get_preset_config
from fast_simulation import FastSimulationEngine

async def test_preset_strategy():
    """Test a preset strategy configuration"""
    print("Testing preset strategy: 'aggressive'")
    
    # Get aggressive preset config
    config = get_preset_config('aggressive')
    print(f"Config: {config.name} - {config.description}")
    print(f"Buy: {config.buy_confidence_threshold:.0%} confidence")
    print(f"Sell: {config.sell_after_candles} candles, red candle: {config.sell_on_red_candle}")
    print(f"Stop loss: {config.stop_loss_percent:.1%}")
    
    # Create simulation
    simulation = FastSimulationEngine(
        tickers=["MSFT", "NVDA"],
        simulation_minutes=60,
        speed_multiplier=60
    )
    
    # Setup with configurable strategy
    simulation.setup_trading_engine()
    configurable_strategy = simulation.strategy_factory.create_configurable_setup(config)
    
    print(f"Created strategy: {configurable_strategy.config.name}")
    print("Running 60-minute simulation...")
    
    # Run simulation (simplified version for testing)
    results = await simulation.run_simulation_with_configurable_strategy(
        configurable_strategy=configurable_strategy,
        enable_detailed_logging=False
    )
    
    # Print basic results
    performance = results['performance']
    print(f"\nResults:")
    print(f"Total trades: {performance['total_trades']}")
    print(f"Complete pairs: {performance['total_trade_pairs']}")
    print(f"P&L: ${performance['total_pnl']:.2f}")
    print(f"Win rate: {performance['win_rate']:.1f}%")
    
    return results

async def test_custom_strategy():
    """Test a custom strategy configuration"""
    print("\nTesting custom strategy configuration...")
    
    # Create custom config
    custom_config = StrategyConfig(
        name="Test Custom",
        description="Custom strategy for testing",
        buy_confidence_threshold=0.7,
        sell_after_candles=4,
        sell_on_red_candle=True,
        stop_loss_percent=0.015,  # 1.5% stop loss
        quantity=75
    )
    
    print(f"Custom config: {custom_config.name}")
    print(f"Buy: {custom_config.buy_confidence_threshold:.0%} confidence")
    print(f"Sell: {custom_config.sell_after_candles} candles")
    print(f"Stop loss: {custom_config.stop_loss_percent:.1%}")
    
    # Create simulation
    simulation = FastSimulationEngine(
        tickers=["UBER", "ROKU"],
        simulation_minutes=45,
        speed_multiplier=60
    )
    
    # Setup with custom strategy
    simulation.setup_trading_engine()
    configurable_strategy = simulation.strategy_factory.create_configurable_setup(custom_config)
    
    print("Running 45-minute simulation...")
    
    # Run simulation
    results = await simulation.run_simulation_with_configurable_strategy(
        configurable_strategy=configurable_strategy,
        enable_detailed_logging=False
    )
    
    # Print basic results
    performance = results['performance']
    print(f"\nResults:")
    print(f"Total trades: {performance['total_trades']}")
    print(f"Complete pairs: {performance['total_trade_pairs']}")
    print(f"P&L: ${performance['total_pnl']:.2f}")
    print(f"Win rate: {performance['win_rate']:.1f}%")
    
    return results

async def main():
    print("="*60)
    print("CONFIGURABLE STRATEGY SYSTEM TEST")
    print("="*60)
    
    try:
        # Test preset strategy
        await test_preset_strategy()
        
        # Test custom strategy
        await test_custom_strategy()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The configurable strategy system is working!")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())