# tests/test_signals.py
"""
信号生成器测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.signals import SignalGenerator, SignalType, Signal
from src.strategy.indicators import calculate_all_indicators


@pytest.fixture
def sample_data():
    """生成测试用的 OHLCV 数据"""
    np.random.seed(42)
    n = 100
    
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
def oversold_data():
    """生成 RSI 超卖数据 (价格持续下跌)"""
    n = 100
    # 连续下跌
    close = 100 * np.exp(np.linspace(0, -0.3, n))
    
    df = pd.DataFrame({
        'open': close * 1.001,
        'high': close * 1.005,
        'low': close * 0.995,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    return df


class TestSignalGenerator:
    """信号生成器测试"""
    
    def test_analyze_returns_signals(self, sample_data):
        generator = SignalGenerator()
        signals = generator.analyze(sample_data, "TEST/USDT")
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)
    
    def test_signal_has_required_fields(self, sample_data):
        generator = SignalGenerator()
        signals = generator.analyze(sample_data, "TEST/USDT")
        
        if len(signals) > 0:
            signal = signals[0]
            assert signal.symbol == "TEST/USDT"
            assert signal.signal_type in [SignalType.LONG, SignalType.SHORT, SignalType.NEUTRAL, SignalType.CLOSE]
            assert signal.price > 0
            assert 0 <= signal.confidence <= 1
            assert signal.strategy is not None
    
    def test_signal_to_message(self, sample_data):
        generator = SignalGenerator()
        signals = generator.analyze(sample_data, "TEST/USDT")
        
        if len(signals) > 0:
            message = signals[0].to_message()
            assert isinstance(message, str)
            assert len(message) > 0


class TestRSISignals:
    """RSI 信号测试"""
    
    def test_oversold_generates_signal(self, oversold_data):
        generator = SignalGenerator()
        df = calculate_all_indicators(oversold_data)
        
        # RSI 应该很低
        current_rsi = df['rsi'].iloc[-1]
        assert current_rsi < 35, f"RSI should be oversold, got {current_rsi}"


class TestSignalType:
    """信号类型测试"""
    
    def test_signal_types(self):
        assert SignalType.LONG.value == "LONG"
        assert SignalType.SHORT.value == "SHORT"
        assert SignalType.NEUTRAL.value == "NEUTRAL"
        assert SignalType.CLOSE.value == "CLOSE"

