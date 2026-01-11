# tests/test_indicators.py
"""
技术指标测试
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.indicators import TechnicalIndicators, calculate_all_indicators


@pytest.fixture
def sample_data():
    """生成测试用的 OHLCV 数据"""
    np.random.seed(42)
    n = 100
    
    # 生成随机价格序列
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


class TestTechnicalIndicators:
    """单个指标测试"""
    
    def test_sma(self, sample_data):
        ind = TechnicalIndicators()
        sma = ind.sma(sample_data['close'], 20)
        
        assert len(sma) == len(sample_data)
        assert sma.isna().sum() == 19  # 前19个值应该是 NaN
        assert not sma.iloc[-1:].isna().any()
    
    def test_ema(self, sample_data):
        ind = TechnicalIndicators()
        ema = ind.ema(sample_data['close'], 12)
        
        assert len(ema) == len(sample_data)
        assert not ema.iloc[-1:].isna().any()
    
    def test_rsi(self, sample_data):
        ind = TechnicalIndicators()
        rsi = ind.rsi(sample_data['close'], 14)
        
        assert len(rsi) == len(sample_data)
        # RSI 应该在 0-100 之间
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_data):
        ind = TechnicalIndicators()
        macd = ind.macd(sample_data['close'])
        
        assert 'macd' in macd
        assert 'signal' in macd
        assert 'histogram' in macd
        assert len(macd['macd']) == len(sample_data)
    
    def test_bollinger_bands(self, sample_data):
        ind = TechnicalIndicators()
        bb = ind.bollinger_bands(sample_data['close'], 20, 2.0)
        
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        
        # 上轨应该 > 中轨 > 下轨
        valid_idx = ~bb['upper'].isna()
        assert (bb['upper'][valid_idx] >= bb['middle'][valid_idx]).all()
        assert (bb['middle'][valid_idx] >= bb['lower'][valid_idx]).all()
    
    def test_atr(self, sample_data):
        ind = TechnicalIndicators()
        atr = ind.atr(sample_data['high'], sample_data['low'], sample_data['close'], 14)
        
        assert len(atr) == len(sample_data)
        # ATR 应该为正
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_stochastic(self, sample_data):
        ind = TechnicalIndicators()
        stoch = ind.stochastic(sample_data['high'], sample_data['low'], sample_data['close'])
        
        assert 'k' in stoch
        assert 'd' in stoch
        
        # K 和 D 应该在 0-100 之间
        valid_k = stoch['k'].dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()


class TestCalculateAllIndicators:
    """综合指标计算测试"""
    
    def test_calculate_all(self, sample_data):
        df = calculate_all_indicators(sample_data.copy())
        
        # 检查所有指标列存在
        expected_columns = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'stoch_k', 'stoch_d'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_indicators_have_values(self, sample_data):
        df = calculate_all_indicators(sample_data.copy())
        
        # 最后一行应该有值 (除了需要更多数据的指标)
        last_row = df.iloc[-1]
        
        assert not pd.isna(last_row['sma_20'])
        assert not pd.isna(last_row['rsi'])
        assert not pd.isna(last_row['macd'])


