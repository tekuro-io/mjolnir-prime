# ===== FILE: tests/portfolio/test_portfolio.py =====
"""
Tests for portfolio management.
"""

import pytest
from trading_simulator.portfolio.portfolio import Portfolio
from trading_simulator.portfolio.position import Position
from trading_simulator.core.exceptions import InsufficientFundsError, InsufficientSharesError


class TestPosition:
    def test_position_creation(self):
        """Test position creation"""
        pos = Position('AAPL')
        assert pos.symbol == 'AAPL'
        assert pos.quantity == 0
        assert pos.avg_cost == 0.0
    
    def test_add_shares(self):
        """Test adding shares to position"""
        pos = Position('AAPL')
        pos.add_shares(100, 150.0)
        
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
    
    def test_add_shares_average_cost(self):
        """Test average cost calculation when adding shares"""
        pos = Position('AAPL')
        pos.add_shares(100, 150.0)  # 100 @ $150
        pos.add_shares(100, 160.0)  # 100 @ $160
        
        assert pos.quantity == 200
        assert pos.avg_cost == 155.0  # (100*150 + 100*160) / 200
    
    def test_remove_shares(self):
        """Test removing shares from position"""
        pos = Position('AAPL')
        pos.add_shares(100, 150.0)
        
        success = pos.remove_shares(50)
        assert success
        assert pos.quantity == 50
        assert pos.avg_cost == 150.0
    
    def test_remove_more_shares_than_available(self):
        """Test removing more shares than available"""
        pos = Position('AAPL')
        pos.add_shares(50, 150.0)
        
        success = pos.remove_shares(100)
        assert not success
        assert pos.quantity == 50


class TestPortfolio:
    def test_portfolio_creation(self):
        """Test portfolio creation"""
        portfolio = Portfolio(100000.0)
        assert portfolio.cash == 100000.0
        assert portfolio.initial_balance == 100000.0
        assert len(portfolio.positions) == 0
    
    def test_execute_buy_success(self):
        """Test successful buy execution"""
        portfolio = Portfolio(100000.0)
        
        success = portfolio.execute_buy('AAPL', 100, 150.0)
        assert success
        assert portfolio.cash == 85000.0  # 100000 - 100*150
        
        position = portfolio.get_position('AAPL')
        assert position.quantity == 100
        assert position.avg_cost == 150.0
    
    def test_execute_buy_insufficient_funds(self):
        """Test buy execution with insufficient funds"""
        portfolio = Portfolio(1000.0)
        
        with pytest.raises(InsufficientFundsError):
            portfolio.execute_buy('AAPL', 100, 150.0)
    
    def test_execute_sell_success(self):
        """Test successful sell execution"""
        portfolio = Portfolio(100000.0)
        portfolio.execute_buy('AAPL', 100, 150.0)
        
        success = portfolio.execute_sell('AAPL', 50, 160.0)
        assert success
        assert portfolio.cash == 93000.0  # 85000 + 50*160
        
        position = portfolio.get_position('AAPL')
        assert position.quantity == 50
    
    def test_execute_sell_insufficient_shares(self):
        """Test sell execution with insufficient shares"""
        portfolio = Portfolio(100000.0)
        portfolio.execute_buy('AAPL', 50, 150.0)
        
        with pytest.raises(InsufficientSharesError):
            portfolio.execute_sell('AAPL', 100, 160.0)
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        portfolio = Portfolio(100000.0)
        portfolio.execute_buy('AAPL', 100, 150.0)
        portfolio.execute_buy('GOOGL', 10, 2800.0)
        
        current_prices = {'AAPL': 160.0, 'GOOGL': 2900.0}
        portfolio_value = portfolio.get_portfolio_value(current_prices)
        
        expected_value = 57000.0 + 100*160.0 + 10*2900.0  # cash + AAPL + GOOGL
        assert portfolio_value == expected_value
    
    def test_pnl_calculation(self):
        """Test P&L calculation"""
        portfolio = Portfolio(100000.0)
        portfolio.execute_buy('AAPL', 100, 150.0)
        
        current_prices = {'AAPL': 160.0}
        pnl = portfolio.get_pnl(current_prices)
        
        expected_pnl = 1000.0  # 100 shares * $10 gain
        assert pnl == expected_pnl
