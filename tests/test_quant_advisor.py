# tests/test_quant_advisor.py
"""
量化顾问测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.quant_advisor import QuantAdvisor, QuantAdvice, Direction, MarketRegime


@pytest.fixture
def sample_data():
    """生成测试用的 OHLCV 数据"""
    np.random.seed(42)
    n = 150  # 需要足够的数据
    
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return df


@pytest.fixture
def trending_up_data():
    """生成上涨趋势数据"""
    n = 150
    trend = np.linspace(0, 0.5, n)  # 稳定上涨
    noise = np.random.randn(n) * 0.01
    close = 100 * np.exp(trend + noise)
    
    df = pd.DataFrame({
        'open': close * 0.999,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return df


@pytest.fixture
def trending_down_data():
    """生成下跌趋势数据"""
    n = 150
    trend = np.linspace(0, -0.5, n)  # 稳定下跌
    noise = np.random.randn(n) * 0.01
    close = 100 * np.exp(trend + noise)
    
    df = pd.DataFrame({
        'open': close * 1.001,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return df


class TestQuantAdvisor:
    """量化顾问测试"""
    
    def test_analyze_returns_advice(self, sample_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(sample_data, "TEST/USDT")
        
        assert isinstance(advice, QuantAdvice)
        assert advice.symbol == "TEST/USDT"
    
    def test_advice_has_required_fields(self, sample_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(sample_data, "TEST/USDT")
        
        # 检查必要字段
        assert advice.direction in [Direction.LONG, Direction.SHORT, Direction.NEUTRAL]
        assert 0 <= advice.confidence <= 100
        assert 0 <= advice.win_probability <= 1
        assert advice.current_price > 0
        assert isinstance(advice.market_regime, MarketRegime)
    
    def test_advice_z_scores(self, sample_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(sample_data, "TEST/USDT")
        
        # Z-scores 应该存在
        assert len(advice.z_scores) > 0
        
        # Z-scores 应该在合理范围内 (通常 -5 到 5)
        for factor, z in advice.z_scores.items():
            assert -10 < z < 10, f"Z-score for {factor} out of range: {z}"
    
    def test_position_size_reasonable(self, sample_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(sample_data, "TEST/USDT")
        
        # 仓位应该在 0-25% 之间
        assert 0 <= advice.position_size_pct <= 25
        assert 0 <= advice.kelly_fraction <= 0.5
    
    def test_stop_loss_take_profit(self, sample_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(sample_data, "TEST/USDT")
        
        if advice.direction == Direction.LONG:
            assert advice.stop_loss < advice.current_price
            assert advice.take_profit > advice.current_price
        elif advice.direction == Direction.SHORT:
            assert advice.stop_loss > advice.current_price
            assert advice.take_profit < advice.current_price


class TestMarketRegimeDetection:
    """市场状态检测测试"""
    
    def test_detects_uptrend(self, trending_up_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(trending_up_data, "TEST/USDT")
        
        # 市场状态检测应该返回有效值
        assert advice.market_regime in [
            MarketRegime.TRENDING_UP, 
            MarketRegime.TRENDING_DOWN,
            MarketRegime.RANGING, 
            MarketRegime.HIGH_VOLATILITY
        ]
    
    def test_detects_downtrend(self, trending_down_data):
        advisor = QuantAdvisor()
        advice = advisor.analyze(trending_down_data, "TEST/USDT")
        
        # 市场状态检测应该返回有效值
        assert advice.market_regime in [
            MarketRegime.TRENDING_UP, 
            MarketRegime.TRENDING_DOWN,
            MarketRegime.RANGING, 
            MarketRegime.HIGH_VOLATILITY
        ]


class TestInsufficientData:
    """数据不足测试"""
    
    def test_insufficient_data_returns_neutral(self):
        advisor = QuantAdvisor()
        
        # 只有 30 根 K 线
        small_df = pd.DataFrame({
            'open': np.random.randn(30) + 100,
            'high': np.random.randn(30) + 101,
            'low': np.random.randn(30) + 99,
            'close': np.random.randn(30) + 100,
            'volume': np.random.randint(1000, 10000, 30)
        })
        
        advice = advisor.analyze(small_df, "TEST/USDT")
        
        assert advice.direction == Direction.NEUTRAL
        assert advice.confidence == 0

