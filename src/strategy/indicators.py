# src/strategy/indicators.py
"""
技术指标计算模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均线"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均线"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        相对强弱指数 (RSI)
        
        RSI > 70: 超买
        RSI < 30: 超卖
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD 指标
        
        返回:
            macd: MACD 线
            signal: 信号线
            histogram: 柱状图
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        布林带
        
        返回:
            upper: 上轨
            middle: 中轨
            lower: 下轨
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        平均真实波幅 (ATR)
        用于衡量波动性
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        随机指标 (KDJ)
        
        K > 80: 超买
        K < 20: 超卖
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return {'k': k, 'd': d}
    
    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """成交量移动平均"""
        return volume.rolling(window=period).mean()


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有常用指标
    
    Args:
        df: 包含 open, high, low, close, volume 的 DataFrame
        
    Returns:
        添加了指标列的 DataFrame
    """
    ind = TechnicalIndicators()
    
    # 移动平均线
    df['sma_20'] = ind.sma(df['close'], 20)
    df['sma_50'] = ind.sma(df['close'], 50)
    df['ema_12'] = ind.ema(df['close'], 12)
    df['ema_26'] = ind.ema(df['close'], 26)
    
    # RSI
    df['rsi'] = ind.rsi(df['close'], 14)
    
    # MACD
    macd = ind.macd(df['close'])
    df['macd'] = macd['macd']
    df['macd_signal'] = macd['signal']
    df['macd_hist'] = macd['histogram']
    
    # 布林带
    bb = ind.bollinger_bands(df['close'])
    df['bb_upper'] = bb['upper']
    df['bb_middle'] = bb['middle']
    df['bb_lower'] = bb['lower']
    
    # ATR
    df['atr'] = ind.atr(df['high'], df['low'], df['close'])
    
    # 随机指标
    stoch = ind.stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['k']
    df['stoch_d'] = stoch['d']
    
    return df

