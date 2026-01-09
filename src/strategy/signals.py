# src/strategy/signals.py
"""
交易信号生成模块
"""
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import TechnicalIndicators, calculate_all_indicators
from src.utils.logger import log


class SignalType(Enum):
    """信号类型"""
    LONG = "LONG"       # 做多
    SHORT = "SHORT"     # 做空
    CLOSE = "CLOSE"     # 平仓
    NEUTRAL = "NEUTRAL" # 无信号


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strategy: str
    price: float
    timestamp: datetime
    reason: str
    confidence: float  # 0-1
    indicators: Dict[str, float]
    
    def to_message(self) -> str:
        """转换为消息格式"""
        emoji = {
            SignalType.LONG: "[LONG]",
            SignalType.SHORT: "[SHORT]",
            SignalType.CLOSE: "[CLOSE]",
            SignalType.NEUTRAL: "[---]"
        }
        
        lines = [
            f"{emoji[self.signal_type]} {self.symbol}",
            f"Strategy: {self.strategy}",
            f"Price: ${self.price:,.2f}",
            f"Reason: {self.reason}",
            f"Confidence: {self.confidence:.0%}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Indicators:"
        ]
        
        for name, value in self.indicators.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.2f}")
            else:
                lines.append(f"  {name}: {value}")
        
        return "\n".join(lines)


class SignalGenerator:
    """信号生成器"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        分析数据并生成信号
        
        Args:
            df: K线数据
            symbol: 交易对
            
        Returns:
            信号列表
        """
        # 计算指标
        df = calculate_all_indicators(df)
        
        signals = []
        
        # 检查各种策略
        signals.extend(self._check_rsi_signal(df, symbol))
        signals.extend(self._check_ma_cross_signal(df, symbol))
        signals.extend(self._check_macd_signal(df, symbol))
        signals.extend(self._check_bollinger_signal(df, symbol))
        
        return signals
    
    def _check_rsi_signal(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """RSI 超买超卖信号"""
        signals = []
        
        if len(df) < 2:
            return signals
        
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        current_price = df['close'].iloc[-1]
        
        indicators = {
            'RSI': current_rsi,
            'RSI_prev': prev_rsi
        }
        
        # RSI 从超卖区回升 (< 30 -> > 30)
        if prev_rsi < 30 and current_rsi > 30:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strategy="RSI_Oversold",
                price=current_price,
                timestamp=datetime.now(),
                reason=f"RSI crossed above 30 (from {prev_rsi:.1f} to {current_rsi:.1f})",
                confidence=0.6,
                indicators=indicators
            ))
        
        # RSI 从超买区回落 (> 70 -> < 70)
        elif prev_rsi > 70 and current_rsi < 70:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strategy="RSI_Overbought",
                price=current_price,
                timestamp=datetime.now(),
                reason=f"RSI crossed below 70 (from {prev_rsi:.1f} to {current_rsi:.1f})",
                confidence=0.6,
                indicators=indicators
            ))
        
        # 极端超卖警告
        elif current_rsi < 20:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                strategy="RSI_Extreme",
                price=current_price,
                timestamp=datetime.now(),
                reason=f"RSI extremely oversold at {current_rsi:.1f}",
                confidence=0.5,
                indicators=indicators
            ))
        
        # 极端超买警告
        elif current_rsi > 80:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.NEUTRAL,
                strategy="RSI_Extreme",
                price=current_price,
                timestamp=datetime.now(),
                reason=f"RSI extremely overbought at {current_rsi:.1f}",
                confidence=0.5,
                indicators=indicators
            ))
        
        return signals
    
    def _check_ma_cross_signal(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """均线交叉信号"""
        signals = []
        
        if len(df) < 2:
            return signals
        
        # 当前和前一根K线的均线值
        current_sma20 = df['sma_20'].iloc[-1]
        current_sma50 = df['sma_50'].iloc[-1]
        prev_sma20 = df['sma_20'].iloc[-2]
        prev_sma50 = df['sma_50'].iloc[-2]
        current_price = df['close'].iloc[-1]
        
        if pd.isna(current_sma50) or pd.isna(prev_sma50):
            return signals
        
        indicators = {
            'SMA20': current_sma20,
            'SMA50': current_sma50,
            'Price': current_price
        }
        
        # 金叉: SMA20 上穿 SMA50
        if prev_sma20 <= prev_sma50 and current_sma20 > current_sma50:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strategy="MA_Golden_Cross",
                price=current_price,
                timestamp=datetime.now(),
                reason="SMA20 crossed above SMA50 (Golden Cross)",
                confidence=0.7,
                indicators=indicators
            ))
        
        # 死叉: SMA20 下穿 SMA50
        elif prev_sma20 >= prev_sma50 and current_sma20 < current_sma50:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strategy="MA_Death_Cross",
                price=current_price,
                timestamp=datetime.now(),
                reason="SMA20 crossed below SMA50 (Death Cross)",
                confidence=0.7,
                indicators=indicators
            ))
        
        return signals
    
    def _check_macd_signal(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """MACD 信号"""
        signals = []
        
        if len(df) < 2:
            return signals
        
        current_macd = df['macd'].iloc[-1]
        current_signal = df['macd_signal'].iloc[-1]
        prev_macd = df['macd'].iloc[-2]
        prev_signal = df['macd_signal'].iloc[-2]
        current_hist = df['macd_hist'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if pd.isna(current_signal) or pd.isna(prev_signal):
            return signals
        
        indicators = {
            'MACD': current_macd,
            'Signal': current_signal,
            'Histogram': current_hist
        }
        
        # MACD 上穿信号线
        if prev_macd <= prev_signal and current_macd > current_signal:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strategy="MACD_Bullish",
                price=current_price,
                timestamp=datetime.now(),
                reason="MACD crossed above signal line",
                confidence=0.65,
                indicators=indicators
            ))
        
        # MACD 下穿信号线
        elif prev_macd >= prev_signal and current_macd < current_signal:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strategy="MACD_Bearish",
                price=current_price,
                timestamp=datetime.now(),
                reason="MACD crossed below signal line",
                confidence=0.65,
                indicators=indicators
            ))
        
        return signals
    
    def _check_bollinger_signal(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """布林带信号"""
        signals = []
        
        if len(df) < 2:
            return signals
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        current_upper = df['bb_upper'].iloc[-1]
        current_lower = df['bb_lower'].iloc[-1]
        current_middle = df['bb_middle'].iloc[-1]
        prev_lower = df['bb_lower'].iloc[-2]
        prev_upper = df['bb_upper'].iloc[-2]
        
        if pd.isna(current_upper):
            return signals
        
        indicators = {
            'BB_Upper': current_upper,
            'BB_Middle': current_middle,
            'BB_Lower': current_lower,
            'Price': current_price
        }
        
        # 价格触及下轨后反弹
        if prev_price <= prev_lower and current_price > current_lower:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strategy="BB_Lower_Bounce",
                price=current_price,
                timestamp=datetime.now(),
                reason="Price bounced from lower Bollinger Band",
                confidence=0.6,
                indicators=indicators
            ))
        
        # 价格触及上轨后回落
        elif prev_price >= prev_upper and current_price < current_upper:
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strategy="BB_Upper_Rejection",
                price=current_price,
                timestamp=datetime.now(),
                reason="Price rejected from upper Bollinger Band",
                confidence=0.6,
                indicators=indicators
            ))
        
        return signals

