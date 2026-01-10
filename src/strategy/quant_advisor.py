# src/strategy/quant_advisor.py
"""
量化交易建议生成器
使用统计学方法和因子模型，而非硬编码规则
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import TechnicalIndicators, calculate_all_indicators
from src.utils.logger import log


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class MarketRegime(Enum):
    """市场状态"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class QuantAdvice:
    """量化交易建议"""
    symbol: str
    direction: Direction
    
    # 统计置信度
    confidence: float           # 0-100
    statistical_edge: float     # 期望收益 (expected return)
    win_probability: float      # 预测胜率
    
    # 价格水平
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # 仓位建议
    position_size_pct: float    # 建议仓位 (占总资金%)
    kelly_fraction: float       # Kelly criterion
    
    # 市场分析
    market_regime: MarketRegime
    volatility_percentile: float  # 当前波动率在历史中的百分位
    
    # 因子分析
    factors: Dict[str, float] = field(default_factory=dict)
    z_scores: Dict[str, float] = field(default_factory=dict)
    
    # 信号
    signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)


class QuantAdvisor:
    """
    量化交易顾问
    
    核心理念:
    1. 一切基于统计显著性，不用硬编码阈值
    2. 动态适应市场状态
    3. 基于风险调整的仓位管理
    4. 多因子综合评分
    """
    
    def __init__(self, lookback_period: int = 100):
        self.lookback = lookback_period
        self.z_threshold = 2.0  # 统计显著性阈值
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> QuantAdvice:
        """
        量化分析
        
        步骤:
        1. 识别市场状态 (Regime Detection)
        2. 计算因子 Z-Score
        3. 评估统计边际 (Statistical Edge)
        4. 计算最优仓位 (Kelly Criterion)
        5. 生成建议
        """
        if len(df) < self.lookback:
            return self._create_neutral_advice(symbol, df)
        
        # 计算指标
        df = calculate_all_indicators(df)
        current = df.iloc[-1]
        current_price = current['close']
        
        # 1. 识别市场状态
        regime = self._detect_regime(df)
        
        # 2. 计算因子和 Z-Score
        factors, z_scores = self._calculate_factors(df)
        
        # 3. 综合因子评分
        composite_score = self._calculate_composite_score(z_scores, regime)
        
        # 4. 估计胜率和期望收益
        win_prob, expected_return = self._estimate_edge(df, composite_score, regime)
        
        # 5. 确定方向
        if composite_score > self.z_threshold and win_prob > 0.5:
            direction = Direction.LONG
        elif composite_score < -self.z_threshold and win_prob > 0.5:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL
        
        # 6. 计算止损止盈 (基于波动率)
        atr = current['atr'] if not pd.isna(current['atr']) else current_price * 0.02
        volatility_adjusted_atr = self._adjust_for_regime(atr, regime)
        
        if direction == Direction.LONG:
            stop_loss = current_price - (volatility_adjusted_atr * 2)
            take_profit = current_price + (volatility_adjusted_atr * 3)
        elif direction == Direction.SHORT:
            stop_loss = current_price + (volatility_adjusted_atr * 2)
            take_profit = current_price - (volatility_adjusted_atr * 3)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        # 7. Kelly Criterion 仓位
        kelly = self._kelly_criterion(win_prob, expected_return, volatility_adjusted_atr / current_price)
        position_size = min(kelly * 100, 10)  # 最大 10%
        
        # 8. 波动率百分位
        vol_percentile = self._calculate_volatility_percentile(df)
        
        # 9. 生成信号说明
        signals, warnings = self._generate_signals(factors, z_scores, regime, vol_percentile)
        
        # 置信度 = 综合评分的绝对值 * 胜率
        confidence = min(100, abs(composite_score) * 20 * win_prob)
        
        return QuantAdvice(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            statistical_edge=expected_return,
            win_probability=win_prob,
            current_price=current_price,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            kelly_fraction=kelly,
            market_regime=regime,
            volatility_percentile=vol_percentile,
            factors=factors,
            z_scores=z_scores,
            signals=signals,
            warnings=warnings
        )
    
    def _detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        市场状态检测
        
        使用:
        - 趋势强度 (ADX 概念)
        - 波动率水平
        - 价格与均线关系
        """
        close = df['close']
        
        # 计算趋势强度
        returns = close.pct_change()
        rolling_return = returns.rolling(20).mean().iloc[-1]
        rolling_std = returns.rolling(20).std().iloc[-1]
        
        # 趋势显著性
        trend_t_stat = rolling_return / (rolling_std / np.sqrt(20)) if rolling_std > 0 else 0
        
        # 波动率状态
        current_vol = rolling_std
        historical_vol = returns.rolling(100).std().iloc[-1]
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        # 判断状态
        if vol_ratio > 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif trend_t_stat > 2:
            return MarketRegime.TRENDING_UP
        elif trend_t_stat < -2:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def _calculate_factors(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        计算量化因子和 Z-Score
        
        因子:
        1. Momentum (动量)
        2. Mean Reversion (均值回归)
        3. Volatility (波动率)
        4. Volume (成交量)
        """
        current = df.iloc[-1]
        factors = {}
        z_scores = {}
        
        # === 动量因子 ===
        # 短期动量: 5日收益率
        returns = df['close'].pct_change()
        mom_5 = returns.rolling(5).sum().iloc[-1]
        mom_5_mean = returns.rolling(5).sum().rolling(50).mean().iloc[-1]
        mom_5_std = returns.rolling(5).sum().rolling(50).std().iloc[-1]
        
        factors['momentum_5d'] = mom_5
        z_scores['momentum_5d'] = (mom_5 - mom_5_mean) / mom_5_std if mom_5_std > 0 else 0
        
        # 中期动量: 20日收益率
        mom_20 = returns.rolling(20).sum().iloc[-1]
        mom_20_mean = returns.rolling(20).sum().rolling(50).mean().iloc[-1]
        mom_20_std = returns.rolling(20).sum().rolling(50).std().iloc[-1]
        
        factors['momentum_20d'] = mom_20
        z_scores['momentum_20d'] = (mom_20 - mom_20_mean) / mom_20_std if mom_20_std > 0 else 0
        
        # === 均值回归因子 ===
        # RSI Z-Score
        rsi = current['rsi']
        if not pd.isna(rsi):
            rsi_mean = df['rsi'].rolling(50).mean().iloc[-1]
            rsi_std = df['rsi'].rolling(50).std().iloc[-1]
            factors['rsi'] = rsi
            z_scores['rsi'] = (rsi - rsi_mean) / rsi_std if rsi_std > 0 else 0
        
        # 布林带位置 (0-1 标准化)
        if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
            bb_position = (current['close'] - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'])
            factors['bb_position'] = bb_position
            # Z-Score: 0.5 是中性，偏离越大越极端
            z_scores['bb_position'] = (bb_position - 0.5) * 4  # 缩放到 -2 到 +2
        
        # === MACD 因子 ===
        if not pd.isna(current['macd']) and not pd.isna(current['macd_signal']):
            macd_diff = current['macd'] - current['macd_signal']
            macd_diff_series = df['macd'] - df['macd_signal']
            macd_mean = macd_diff_series.rolling(50).mean().iloc[-1]
            macd_std = macd_diff_series.rolling(50).std().iloc[-1]
            
            factors['macd_diff'] = macd_diff
            z_scores['macd_diff'] = (macd_diff - macd_mean) / macd_std if macd_std > 0 else 0
        
        # === 成交量因子 ===
        if 'volume' in df.columns:
            vol_ratio = current['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            factors['volume_ratio'] = vol_ratio
            
            vol_mean = (df['volume'] / df['volume'].rolling(20).mean()).rolling(50).mean().iloc[-1]
            vol_std = (df['volume'] / df['volume'].rolling(20).mean()).rolling(50).std().iloc[-1]
            z_scores['volume_ratio'] = (vol_ratio - vol_mean) / vol_std if vol_std > 0 else 0
        
        return factors, z_scores
    
    def _calculate_composite_score(self, z_scores: Dict[str, float], regime: MarketRegime) -> float:
        """
        综合评分
        
        根据市场状态调整因子权重:
        - 趋势市场: 动量因子权重高
        - 震荡市场: 均值回归因子权重高
        """
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # 趋势市场: 重动量
            weights = {
                'momentum_5d': 0.3,
                'momentum_20d': 0.3,
                'macd_diff': 0.25,
                'rsi': 0.1,
                'bb_position': 0.05,
            }
        elif regime == MarketRegime.RANGING:
            # 震荡市场: 重均值回归
            weights = {
                'momentum_5d': 0.1,
                'momentum_20d': 0.1,
                'macd_diff': 0.2,
                'rsi': 0.35,        # RSI 更重要
                'bb_position': 0.25, # 布林带更重要
            }
        else:  # HIGH_VOLATILITY
            # 高波动: 保守，均衡
            weights = {
                'momentum_5d': 0.15,
                'momentum_20d': 0.15,
                'macd_diff': 0.2,
                'rsi': 0.25,
                'bb_position': 0.25,
            }
        
        score = 0
        for factor, weight in weights.items():
            if factor in z_scores:
                # 均值回归因子要反转方向
                if factor in ['rsi', 'bb_position']:
                    score -= z_scores[factor] * weight  # 超买做空，超卖做多
                else:
                    score += z_scores[factor] * weight
        
        return score
    
    def _estimate_edge(self, df: pd.DataFrame, composite_score: float, regime: MarketRegime) -> Tuple[float, float]:
        """
        估计统计边际
        
        使用历史数据估计:
        - 胜率 (win probability)
        - 期望收益 (expected return)
        """
        # 简化版: 基于综合评分估计
        # 实际应该用回测数据
        
        # 评分越极端，信号越强
        signal_strength = min(abs(composite_score) / 3, 1)  # 0-1
        
        # 基础胜率 50%，根据信号强度调整
        base_win_rate = 0.5
        win_prob = base_win_rate + signal_strength * 0.15  # 最高 65%
        
        # 期望收益: 基于波动率和胜率
        returns = df['close'].pct_change()
        avg_move = returns.abs().rolling(20).mean().iloc[-1]
        
        # 期望收益 = 胜率 * 平均盈利 - (1-胜率) * 平均亏损
        # 假设盈亏比 1.5:1
        rr_ratio = 1.5
        expected_return = win_prob * avg_move * rr_ratio - (1 - win_prob) * avg_move
        
        return win_prob, expected_return
    
    def _kelly_criterion(self, win_prob: float, expected_return: float, risk_per_trade: float) -> float:
        """
        Kelly Criterion 仓位计算
        
        f* = (p * b - q) / b
        
        p = 胜率
        q = 1 - p
        b = 盈亏比
        """
        if risk_per_trade <= 0:
            return 0
        
        q = 1 - win_prob
        b = 1.5  # 假设盈亏比
        
        kelly = (win_prob * b - q) / b
        
        # 使用半 Kelly，更保守
        half_kelly = kelly * 0.5
        
        return max(0, min(0.25, half_kelly))  # 最大 25%
    
    def _adjust_for_regime(self, atr: float, regime: MarketRegime) -> float:
        """根据市场状态调整 ATR"""
        multipliers = {
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 1.0,
            MarketRegime.RANGING: 0.8,
            MarketRegime.HIGH_VOLATILITY: 1.5,
        }
        return atr * multipliers.get(regime, 1.0)
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """计算当前波动率在历史中的百分位"""
        returns = df['close'].pct_change()
        current_vol = returns.rolling(20).std().iloc[-1]
        historical_vols = returns.rolling(20).std().dropna()
        
        percentile = stats.percentileofscore(historical_vols, current_vol)
        return percentile
    
    def _generate_signals(self, factors: Dict, z_scores: Dict, regime: MarketRegime, vol_pct: float) -> Tuple[List[str], List[str]]:
        """生成信号说明和警告"""
        signals = []
        warnings = []
        
        # 市场状态
        regime_names = {
            MarketRegime.TRENDING_UP: "Uptrend / 上涨趋势",
            MarketRegime.TRENDING_DOWN: "Downtrend / 下跌趋势",
            MarketRegime.RANGING: "Ranging / 震荡市",
            MarketRegime.HIGH_VOLATILITY: "High Volatility / 高波动",
        }
        signals.append(f"Market Regime / 市场状态: {regime_names[regime]}")
        
        # 显著因子
        for factor, z in z_scores.items():
            if abs(z) > 2:
                direction = "bullish / 看涨" if z > 0 else "bearish / 看跌"
                if factor in ['rsi', 'bb_position']:
                    direction = "bearish / 看跌" if z > 0 else "bullish / 看涨"
                signals.append(f"{factor}: Z={z:.2f} ({direction})")
        
        # 警告
        if vol_pct > 80:
            warnings.append(f"Volatility at {vol_pct:.0f}th percentile / 波动率处于{vol_pct:.0f}%分位 - Reduce size / 减小仓位")
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append("High volatility regime / 高波动状态 - Use tight stops / 使用紧止损")
        
        return signals, warnings
    
    def _create_neutral_advice(self, symbol: str, df: pd.DataFrame) -> QuantAdvice:
        """创建中性建议"""
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        return QuantAdvice(
            symbol=symbol,
            direction=Direction.NEUTRAL,
            confidence=0,
            statistical_edge=0,
            win_probability=0.5,
            current_price=current_price,
            entry_price=current_price,
            stop_loss=current_price,
            take_profit=current_price,
            position_size_pct=0,
            kelly_fraction=0,
            market_regime=MarketRegime.RANGING,
            volatility_percentile=50,
            signals=["Insufficient data / 数据不足"],
            warnings=["Need more historical data / 需要更多历史数据"]
        )


# 测试
if __name__ == "__main__":
    from src.data.tradingview import TradingViewDataFetcher
    
    fetcher = TradingViewDataFetcher()
    advisor = QuantAdvisor()
    
    df = fetcher.get_klines("BTC/USDT:USDT", "1h", 150)
    
    if df is not None:
        advice = advisor.analyze(df, "BTC/USDT:USDT")
        
        print(f"\n{'='*60}")
        print(f"QUANT ANALYSIS / 量化分析: {advice.symbol}")
        print(f"{'='*60}")
        print(f"Direction / 方向: {advice.direction.value}")
        print(f"Confidence / 置信度: {advice.confidence:.1f}%")
        print(f"Win Probability / 胜率: {advice.win_probability:.1%}")
        print(f"Expected Edge / 期望收益: {advice.statistical_edge:.2%}")
        print(f"\nMarket Regime / 市场状态: {advice.market_regime.value}")
        print(f"Volatility Percentile / 波动率分位: {advice.volatility_percentile:.0f}%")
        print(f"\nPosition Size / 建议仓位: {advice.position_size_pct:.1f}%")
        print(f"Kelly Fraction: {advice.kelly_fraction:.2%}")
        print(f"\nPrice Levels / 价格:")
        print(f"  Current / 当前: ${advice.current_price:,.2f}")
        print(f"  Stop Loss / 止损: ${advice.stop_loss:,.2f}")
        print(f"  Take Profit / 止盈: ${advice.take_profit:,.2f}")
        print(f"\nZ-Scores:")
        for factor, z in advice.z_scores.items():
            print(f"  {factor}: {z:+.2f}")
        print(f"\nSignals / 信号:")
        for s in advice.signals:
            print(f"  - {s}")
        if advice.warnings:
            print(f"\nWarnings / 警告:")
            for w in advice.warnings:
                print(f"  ! {w}")

