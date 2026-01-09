# src/strategy/advisor.py
"""
交易建议生成器
综合分析多个指标，给出具体的交易建议
"""
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import TechnicalIndicators, calculate_all_indicators
from src.utils.logger import log


class Direction(Enum):
    """交易方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class TradeAdvice:
    """交易建议"""
    symbol: str
    direction: Direction
    confidence: float  # 0-100
    risk_level: RiskLevel
    
    # 价格相关
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit_1: float  # 第一目标
    take_profit_2: float  # 第二目标
    take_profit_3: float  # 第三目标
    
    # 分析结果
    trend_score: int      # -100 到 100
    momentum_score: int   # -100 到 100
    volatility_score: int # 0 到 100
    
    # 详细分析
    analysis: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_risk_reward_ratio(self) -> float:
        """计算风险收益比"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - self.entry_price)
        if risk == 0:
            return 0
        return reward / risk
    
    def to_discord_embed_data(self) -> Dict[str, Any]:
        """转换为 Discord Embed 数据"""
        direction_text = {
            Direction.LONG: "LONG (Buy)",
            Direction.SHORT: "SHORT (Sell)",
            Direction.NEUTRAL: "NEUTRAL (Wait)"
        }
        
        risk_color = {
            RiskLevel.LOW: 0x00FF00,      # 绿色
            RiskLevel.MEDIUM: 0xFFFF00,   # 黄色
            RiskLevel.HIGH: 0xFF0000      # 红色
        }
        
        return {
            'direction': direction_text[self.direction],
            'confidence': self.confidence,
            'risk_level': self.risk_level.value,
            'color': risk_color[self.risk_level],
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'risk_reward': self.get_risk_reward_ratio(),
            'trend_score': self.trend_score,
            'momentum_score': self.momentum_score,
            'volatility_score': self.volatility_score,
            'reasons': self.reasons,
            'warnings': self.warnings
        }


class TradingAdvisor:
    """交易顾问"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> TradeAdvice:
        """
        综合分析并生成交易建议
        
        Args:
            df: K线数据 (需要足够的历史数据，建议100根以上)
            symbol: 交易对
            
        Returns:
            TradeAdvice 对象
        """
        # 确保有足够数据
        if len(df) < 50:
            log.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            return self._create_neutral_advice(symbol, df)
        
        # 计算所有指标
        df = calculate_all_indicators(df)
        
        # 获取最新数据
        current = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = current['close']
        
        # 分析各个维度
        trend_analysis = self._analyze_trend(df)
        momentum_analysis = self._analyze_momentum(df)
        volatility_analysis = self._analyze_volatility(df)
        support_resistance = self._find_support_resistance(df)
        
        # 综合评分
        total_score = (
            trend_analysis['score'] * 0.4 +
            momentum_analysis['score'] * 0.4 +
            (50 - volatility_analysis['score']) * 0.2  # 波动性越低越好
        )
        
        # 确定方向
        if total_score >= 25:
            direction = Direction.LONG
        elif total_score <= -25:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL
        
        # 计算置信度 (0-100)
        confidence = min(100, abs(total_score))
        
        # 计算风险等级
        risk_level = self._calculate_risk_level(
            volatility_analysis['score'],
            confidence,
            df
        )
        
        # 计算入场、止损、止盈
        atr = current['atr'] if not pd.isna(current['atr']) else current_price * 0.02
        
        if direction == Direction.LONG:
            entry_price = current_price
            stop_loss = entry_price - (atr * 2)
            take_profit_1 = entry_price + (atr * 2)    # 1:1
            take_profit_2 = entry_price + (atr * 3)    # 1.5:1
            take_profit_3 = entry_price + (atr * 4)    # 2:1
        elif direction == Direction.SHORT:
            entry_price = current_price
            stop_loss = entry_price + (atr * 2)
            take_profit_1 = entry_price - (atr * 2)
            take_profit_2 = entry_price - (atr * 3)
            take_profit_3 = entry_price - (atr * 4)
        else:
            entry_price = current_price
            stop_loss = current_price
            take_profit_1 = current_price
            take_profit_2 = current_price
            take_profit_3 = current_price
        
        # 收集分析理由 (双语)
        reasons = []
        warnings = []
        
        # 趋势分析理由
        if trend_analysis['score'] > 30:
            reasons.append("Strong uptrend / 强势上涨趋势: Price above MA20 & MA50 / 价格在MA20和MA50之上")
        elif trend_analysis['score'] < -30:
            reasons.append("Strong downtrend / 强势下跌趋势: Price below MA20 & MA50 / 价格在MA20和MA50之下")
        
        if trend_analysis['ma_cross'] == 'golden':
            reasons.append("Golden Cross / 金叉: MA20 crossed above MA50 / MA20上穿MA50")
        elif trend_analysis['ma_cross'] == 'death':
            reasons.append("Death Cross / 死叉: MA20 crossed below MA50 / MA20下穿MA50")
        
        # 动量分析理由
        rsi = current['rsi']
        if not pd.isna(rsi):
            if rsi < 30:
                reasons.append(f"RSI oversold / RSI超卖 ({rsi:.1f})")
            elif rsi > 70:
                reasons.append(f"RSI overbought / RSI超买 ({rsi:.1f})")
            elif 30 <= rsi <= 40 and direction == Direction.LONG:
                reasons.append(f"RSI recovering / RSI从超卖区回升 ({rsi:.1f})")
            elif 60 <= rsi <= 70 and direction == Direction.SHORT:
                reasons.append(f"RSI declining / RSI从超买区回落 ({rsi:.1f})")
        
        if momentum_analysis['macd_cross'] == 'bullish':
            reasons.append("MACD bullish crossover / MACD金叉看涨")
        elif momentum_analysis['macd_cross'] == 'bearish':
            reasons.append("MACD bearish crossover / MACD死叉看跌")
        
        # 布林带分析
        if not pd.isna(current['bb_lower']):
            bb_position = (current_price - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'])
            if bb_position < 0.2:
                reasons.append("Price near lower BB / 价格接近布林带下轨")
            elif bb_position > 0.8:
                reasons.append("Price near upper BB / 价格接近布林带上轨")
        
        # 警告 (双语)
        if volatility_analysis['score'] > 70:
            warnings.append("High volatility / 高波动性 - Reduce position / 建议减小仓位")
        
        if confidence < 40:
            warnings.append("Low confidence / 置信度低 - Wait for better setup / 建议等待更好机会")
        
        if direction != Direction.NEUTRAL and self.get_risk_reward_ratio(entry_price, stop_loss, take_profit_1) < 1.5:
            warnings.append("R/R below 1.5:1 / 风险收益比低于1.5:1")
        
        # 汇总分析数据
        analysis = {
            'trend': trend_analysis,
            'momentum': momentum_analysis,
            'volatility': volatility_analysis,
            'support_resistance': support_resistance,
            'indicators': {
                'rsi': float(current['rsi']) if not pd.isna(current['rsi']) else None,
                'macd': float(current['macd']) if not pd.isna(current['macd']) else None,
                'macd_signal': float(current['macd_signal']) if not pd.isna(current['macd_signal']) else None,
                'sma_20': float(current['sma_20']) if not pd.isna(current['sma_20']) else None,
                'sma_50': float(current['sma_50']) if not pd.isna(current['sma_50']) else None,
                'bb_upper': float(current['bb_upper']) if not pd.isna(current['bb_upper']) else None,
                'bb_lower': float(current['bb_lower']) if not pd.isna(current['bb_lower']) else None,
                'atr': float(current['atr']) if not pd.isna(current['atr']) else None,
                'stoch_k': float(current['stoch_k']) if not pd.isna(current['stoch_k']) else None,
            }
        }
        
        return TradeAdvice(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            risk_level=risk_level,
            current_price=current_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            trend_score=int(trend_analysis['score']),
            momentum_score=int(momentum_analysis['score']),
            volatility_score=int(volatility_analysis['score']),
            analysis=analysis,
            reasons=reasons,
            warnings=warnings
        )
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析趋势"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current['close']
        
        score = 0
        ma_cross = None
        
        # MA 位置分析
        sma_20 = current['sma_20']
        sma_50 = current['sma_50']
        
        if not pd.isna(sma_20):
            if price > sma_20:
                score += 20
            else:
                score -= 20
        
        if not pd.isna(sma_50):
            if price > sma_50:
                score += 20
            else:
                score -= 20
        
        # MA 交叉
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            prev_sma_20 = prev['sma_20']
            prev_sma_50 = prev['sma_50']
            
            if not pd.isna(prev_sma_20) and not pd.isna(prev_sma_50):
                if prev_sma_20 <= prev_sma_50 and sma_20 > sma_50:
                    ma_cross = 'golden'
                    score += 30
                elif prev_sma_20 >= prev_sma_50 and sma_20 < sma_50:
                    ma_cross = 'death'
                    score -= 30
        
        # MA 斜率
        if not pd.isna(sma_20) and len(df) >= 5:
            sma_5_ago = df['sma_20'].iloc[-5]
            if not pd.isna(sma_5_ago):
                slope = (sma_20 - sma_5_ago) / sma_5_ago * 100
                score += min(20, max(-20, slope * 10))
        
        return {
            'score': max(-100, min(100, score)),
            'ma_cross': ma_cross,
            'price_vs_ma20': 'above' if price > sma_20 else 'below' if not pd.isna(sma_20) else None,
            'price_vs_ma50': 'above' if price > sma_50 else 'below' if not pd.isna(sma_50) else None,
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析动量"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        macd_cross = None
        
        # RSI 分析
        rsi = current['rsi']
        if not pd.isna(rsi):
            if rsi < 30:
                score += 30  # 超卖 - 潜在买入
            elif rsi > 70:
                score -= 30  # 超买 - 潜在卖出
            elif rsi < 50:
                score -= 10
            else:
                score += 10
        
        # MACD 分析
        macd = current['macd']
        macd_signal = current['macd_signal']
        prev_macd = prev['macd']
        prev_signal = prev['macd_signal']
        
        if not pd.isna(macd) and not pd.isna(macd_signal):
            # MACD 位置
            if macd > macd_signal:
                score += 15
            else:
                score -= 15
            
            # MACD 交叉
            if not pd.isna(prev_macd) and not pd.isna(prev_signal):
                if prev_macd <= prev_signal and macd > macd_signal:
                    macd_cross = 'bullish'
                    score += 25
                elif prev_macd >= prev_signal and macd < macd_signal:
                    macd_cross = 'bearish'
                    score -= 25
        
        # Stochastic 分析
        stoch_k = current['stoch_k']
        if not pd.isna(stoch_k):
            if stoch_k < 20:
                score += 15
            elif stoch_k > 80:
                score -= 15
        
        return {
            'score': max(-100, min(100, score)),
            'macd_cross': macd_cross,
            'rsi': float(rsi) if not pd.isna(rsi) else None,
            'stoch_k': float(stoch_k) if not pd.isna(stoch_k) else None,
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析波动性"""
        current = df.iloc[-1]
        
        # ATR 相对波动率
        atr = current['atr']
        price = current['close']
        
        if not pd.isna(atr) and price > 0:
            atr_percent = (atr / price) * 100
            # 波动率评分 (0-100)
            volatility_score = min(100, atr_percent * 20)
        else:
            volatility_score = 50
        
        # 布林带宽度
        bb_width = None
        if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
            bb_width = (current['bb_upper'] - current['bb_lower']) / current['bb_middle'] * 100
        
        return {
            'score': volatility_score,
            'atr_percent': atr_percent if not pd.isna(atr) else None,
            'bb_width': bb_width,
        }
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """找出支撑和阻力位"""
        recent = df.tail(20)
        
        highs = recent['high'].values
        lows = recent['low'].values
        
        resistance = float(highs.max())
        support = float(lows.min())
        
        return {
            'resistance': resistance,
            'support': support,
        }
    
    def _calculate_risk_level(self, volatility_score: float, confidence: float, df: pd.DataFrame) -> RiskLevel:
        """计算风险等级"""
        # 综合考虑波动率和置信度
        if volatility_score > 70 or confidence < 30:
            return RiskLevel.HIGH
        elif volatility_score > 50 or confidence < 50:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _create_neutral_advice(self, symbol: str, df: pd.DataFrame) -> TradeAdvice:
        """创建中性建议 (数据不足时)"""
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        return TradeAdvice(
            symbol=symbol,
            direction=Direction.NEUTRAL,
            confidence=0,
            risk_level=RiskLevel.HIGH,
            current_price=current_price,
            entry_price=current_price,
            stop_loss=current_price,
            take_profit_1=current_price,
            take_profit_2=current_price,
            take_profit_3=current_price,
            trend_score=0,
            momentum_score=0,
            volatility_score=50,
            reasons=["Insufficient data for analysis"],
            warnings=["Need at least 50 candles for reliable analysis"]
        )
    
    @staticmethod
    def get_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
        """计算风险收益比"""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk == 0:
            return 0
        return reward / risk


# 测试
if __name__ == "__main__":
    from src.data.tradingview import TradingViewDataFetcher
    
    fetcher = TradingViewDataFetcher()
    advisor = TradingAdvisor()
    
    # 获取数据
    df = fetcher.get_klines("BTC/USDT:USDT", "1h", 100)
    
    if df is not None:
        advice = advisor.analyze(df, "BTC/USDT:USDT")
        
        print(f"\n{'='*50}")
        print(f"Trading Advice for {advice.symbol}")
        print(f"{'='*50}")
        print(f"Direction: {advice.direction.value}")
        print(f"Confidence: {advice.confidence:.0f}%")
        print(f"Risk Level: {advice.risk_level.value}")
        print(f"\nPrice Levels:")
        print(f"  Current:     ${advice.current_price:,.2f}")
        print(f"  Entry:       ${advice.entry_price:,.2f}")
        print(f"  Stop Loss:   ${advice.stop_loss:,.2f}")
        print(f"  TP1:         ${advice.take_profit_1:,.2f}")
        print(f"  TP2:         ${advice.take_profit_2:,.2f}")
        print(f"  TP3:         ${advice.take_profit_3:,.2f}")
        print(f"  R/R Ratio:   {advice.get_risk_reward_ratio():.2f}")
        print(f"\nScores:")
        print(f"  Trend:       {advice.trend_score}")
        print(f"  Momentum:    {advice.momentum_score}")
        print(f"  Volatility:  {advice.volatility_score}")
        print(f"\nReasons:")
        for r in advice.reasons:
            print(f"  - {r}")
        if advice.warnings:
            print(f"\nWarnings:")
            for w in advice.warnings:
                print(f"  ! {w}")

