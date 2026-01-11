# src/strategy/quant_advisor.py
"""
量化交易建议生成器
使用统计学方法和因子模型，而非硬编码规则
支持 AR (自回归) 模型预测因子 (可选，需要 PyTorch)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from scipy import stats

# PyTorch 是可选的
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import TechnicalIndicators, calculate_all_indicators
from src.utils.logger import log

# AR 模型也是可选的
if TORCH_AVAILABLE:
    try:
        from src.strategy.ar_model import LinearARModel, load_ar_model
    except ImportError:
        LinearARModel = None
        load_ar_model = None
else:
    LinearARModel = None
    load_ar_model = None


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
    5. 支持 AR 模型预测因子 (可选)
    """
    
    def __init__(
        self, 
        lookback_period: int = 100,
        ar_model: Optional[LinearARModel] = None,
        ar_model_path: Optional[str] = None,
        ar_n_lags: int = 3
    ):
        """
        初始化量化顾问
        
        Args:
            lookback_period: 回看周期
            ar_model: 预训练的 AR 模型 (可选)
            ar_model_path: AR 模型文件路径 (可选)
            ar_n_lags: AR 模型滞后期数 (默认 3)
        """
        self.lookback = lookback_period
        self.z_threshold = 2.0  # 统计显著性阈值
        self.ar_n_lags = ar_n_lags
        
        # 加载 AR 模型 (需要 PyTorch)
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available - AR model features disabled")
            self.ar_model = None
        elif ar_model is not None:
            self.ar_model = ar_model
        elif ar_model_path is not None and load_ar_model is not None:
            try:
                self.ar_model = load_ar_model(ar_model_path)
                log.info(f"Loaded AR model from {ar_model_path}")
            except Exception as e:
                log.warning(f"Failed to load AR model: {e}")
                self.ar_model = None
        else:
            self.ar_model = None
    
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
        
        # 5. 确定方向 (改进版 - 考虑多个因素)
        direction = self._determine_direction(
            composite_score, win_prob, expected_return, regime, z_scores
        )
        
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
        
        因子 (基于 K 线数量，适用于任何 timeframe):
        1. Momentum (动量) - 短期/中期/微观
        2. Mean Reversion (均值回归) - RSI, BB
        3. Trend (趋势) - MACD, 价格位置
        4. Acceleration (加速度) - 动量变化率
        5. Volume (成交量)
        """
        current = df.iloc[-1]
        factors = {}
        z_scores = {}
        
        returns = df['close'].pct_change()
        
        # === 动量因子 (基于 K 线数量，非固定天数) ===
        
        # 微观动量: 3 根 K 线 (适合即时决策)
        mom_micro = returns.rolling(3).sum().iloc[-1]
        mom_micro_mean = returns.rolling(3).sum().rolling(30).mean().iloc[-1]
        mom_micro_std = returns.rolling(3).sum().rolling(30).std().iloc[-1]
        factors['momentum_micro'] = mom_micro
        z_scores['momentum_micro'] = (mom_micro - mom_micro_mean) / mom_micro_std if mom_micro_std > 0 else 0
        
        # 短期动量: 5 根 K 线
        mom_short = returns.rolling(5).sum().iloc[-1]
        mom_short_mean = returns.rolling(5).sum().rolling(50).mean().iloc[-1]
        mom_short_std = returns.rolling(5).sum().rolling(50).std().iloc[-1]
        factors['momentum_short'] = mom_short
        z_scores['momentum_short'] = (mom_short - mom_short_mean) / mom_short_std if mom_short_std > 0 else 0
        
        # 中期动量: 20 根 K 线
        mom_medium = returns.rolling(20).sum().iloc[-1]
        mom_medium_mean = returns.rolling(20).sum().rolling(50).mean().iloc[-1]
        mom_medium_std = returns.rolling(20).sum().rolling(50).std().iloc[-1]
        factors['momentum_medium'] = mom_medium
        z_scores['momentum_medium'] = (mom_medium - mom_medium_mean) / mom_medium_std if mom_medium_std > 0 else 0
        
        # === 加速度因子 (动量的变化率，捕捉趋势加速/减速) ===
        mom_series = returns.rolling(5).sum()
        acceleration = mom_series.diff(3).iloc[-1]  # 动量 3 根 K 线的变化
        acc_mean = mom_series.diff(3).rolling(30).mean().iloc[-1]
        acc_std = mom_series.diff(3).rolling(30).std().iloc[-1]
        factors['acceleration'] = acceleration
        z_scores['acceleration'] = (acceleration - acc_mean) / acc_std if acc_std > 0 else 0
        
        # === ROC (价格变化率) ===
        roc_5 = (current['close'] / df['close'].iloc[-6] - 1) if len(df) > 5 else 0
        roc_series = df['close'].pct_change(5)
        roc_mean = roc_series.rolling(30).mean().iloc[-1]
        roc_std = roc_series.rolling(30).std().iloc[-1]
        factors['roc'] = roc_5
        z_scores['roc'] = (roc_5 - roc_mean) / roc_std if roc_std > 0 else 0
        
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
            bb_range = current['bb_upper'] - current['bb_lower']
            if bb_range > 0:
                bb_position = (current['close'] - current['bb_lower']) / bb_range
                factors['bb_position'] = bb_position
                # Z-Score: 0.5 是中性，偏离越大越极端
                z_scores['bb_position'] = (bb_position - 0.5) * 4  # 缩放到 -2 到 +2
        
        # === 趋势因子 ===
        # MACD
        if not pd.isna(current['macd']) and not pd.isna(current['macd_signal']):
            macd_diff = current['macd'] - current['macd_signal']
            macd_diff_series = df['macd'] - df['macd_signal']
            macd_mean = macd_diff_series.rolling(50).mean().iloc[-1]
            macd_std = macd_diff_series.rolling(50).std().iloc[-1]
            factors['macd_diff'] = macd_diff
            z_scores['macd_diff'] = (macd_diff - macd_mean) / macd_std if macd_std > 0 else 0
        
        # 价格相对短期均线位置
        if not pd.isna(current['sma_20']):
            price_vs_ma = (current['close'] - current['sma_20']) / current['sma_20']
            price_vs_ma_series = (df['close'] - df['sma_20']) / df['sma_20']
            pma_mean = price_vs_ma_series.rolling(50).mean().iloc[-1]
            pma_std = price_vs_ma_series.rolling(50).std().iloc[-1]
            factors['price_vs_ma'] = price_vs_ma
            z_scores['price_vs_ma'] = (price_vs_ma - pma_mean) / pma_std if pma_std > 0 else 0
        
        # === 成交量因子 ===
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            if vol_ma > 0:
                vol_ratio = current['volume'] / vol_ma
                factors['volume_ratio'] = vol_ratio
                
                vol_ratio_series = df['volume'] / df['volume'].rolling(20).mean()
                vol_mean = vol_ratio_series.rolling(50).mean().iloc[-1]
                vol_std = vol_ratio_series.rolling(50).std().iloc[-1]
                z_scores['volume_ratio'] = (vol_ratio - vol_mean) / vol_std if vol_std > 0 else 0
        
        # === AR 模型预测因子 ===
        if self.ar_model is not None:
            ar_prediction = self._calculate_ar_prediction(returns)
            if ar_prediction is not None:
                factors['ar_prediction'] = ar_prediction
                # Z-Score: 预测值相对于历史波动的标准化
                recent_std = returns.rolling(20).std().iloc[-1]
                if recent_std > 0:
                    z_scores['ar_prediction'] = ar_prediction / recent_std
                else:
                    z_scores['ar_prediction'] = 0
        
        return factors, z_scores
    
    def _calculate_ar_prediction(self, returns: pd.Series) -> Optional[float]:
        """
        使用 AR 模型计算预测值

        Args:
            returns: 收益率序列

        Returns:
            预测的下一期 log return
        """
        if not TORCH_AVAILABLE or self.ar_model is None:
            return None
        
        # 获取最近 n_lags 个 log returns
        log_returns = np.log(1 + returns).dropna()
        
        if len(log_returns) < self.ar_n_lags:
            return None
        
        # 获取最近的 log returns (最新在前)
        recent_log_rets = log_returns.iloc[-self.ar_n_lags:].values[::-1].copy()
        
        # 预测
        X = torch.tensor(recent_log_rets, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.ar_model(X).item()
        
        return prediction
    
    def _determine_direction(
        self, 
        composite_score: float, 
        win_prob: float, 
        expected_return: float,
        regime: MarketRegime,
        z_scores: Dict[str, float]
    ) -> Direction:
        """
        智能方向判断 - 综合多个因素
        
        考虑:
        1. 综合评分强度
        2. 胜率阈值 (根据市场状态调整)
        3. 期望收益是否为正
        4. AR 模型预测 (如果有)
        5. 因子一致性
        """
        # 根据市场状态调整阈值
        if regime == MarketRegime.HIGH_VOLATILITY:
            score_threshold = 2.5  # 高波动需要更强信号
            win_threshold = 0.52
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            score_threshold = 1.5  # 趋势市可以更激进
            win_threshold = 0.50
        else:  # RANGING
            score_threshold = 2.0
            win_threshold = 0.53
        
        # 检查期望收益
        if expected_return <= 0:
            return Direction.NEUTRAL
        
        # 检查因子一致性 (至少 3 个主要因子同向)
        momentum_factors = ['momentum_micro', 'momentum_short', 'momentum_medium', 'ar_prediction']
        bullish_count = sum(1 for f in momentum_factors if f in z_scores and z_scores[f] > 0.5)
        bearish_count = sum(1 for f in momentum_factors if f in z_scores and z_scores[f] < -0.5)
        
        # AR 模型预测权重更高
        ar_signal = z_scores.get('ar_prediction', 0)
        if not pd.isna(ar_signal) and abs(ar_signal) > 1:
            if ar_signal > 0:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # 判断方向
        if composite_score > score_threshold and win_prob > win_threshold:
            if bullish_count >= 2 or (self.ar_model is not None and ar_signal > 0.5):
                return Direction.LONG
        elif composite_score < -score_threshold and win_prob > win_threshold:
            if bearish_count >= 2 or (self.ar_model is not None and ar_signal < -0.5):
                return Direction.SHORT
        
        # 特殊情况：AR 模型信号很强
        if self.ar_model is not None and abs(ar_signal) > 2:
            if ar_signal > 2 and win_prob > 0.52:
                return Direction.LONG
            elif ar_signal < -2 and win_prob > 0.52:
                return Direction.SHORT
        
        return Direction.NEUTRAL
    
    def _calculate_composite_score(self, z_scores: Dict[str, float], regime: MarketRegime) -> float:
        """
        综合评分
        
        根据市场状态调整因子权重:
        - 趋势市场: 动量因子权重高
        - 震荡市场: 均值回归因子权重高
        
        因子分类:
        - 趋势跟随: momentum_micro, momentum_short, momentum_medium, macd_diff, price_vs_ma, roc, acceleration
        - 均值回归: rsi, bb_position (反向)
        - AR 预测: ar_prediction (如果可用)
        """
        # 检查是否有 AR 预测
        has_ar = 'ar_prediction' in z_scores and not pd.isna(z_scores.get('ar_prediction'))
        
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # 趋势市场: 重动量和趋势因子
            if has_ar:
                weights = {
                    'ar_prediction': 0.25,     # AR 模型预测
                    'momentum_micro': 0.12,
                    'momentum_short': 0.12,
                    'momentum_medium': 0.10,
                    'acceleration': 0.08,
                    'roc': 0.08,
                    'macd_diff': 0.10,
                    'price_vs_ma': 0.08,
                    'rsi': 0.04,
                    'bb_position': 0.03,
                }
            else:
                weights = {
                    'momentum_micro': 0.15,
                    'momentum_short': 0.15,
                    'momentum_medium': 0.15,
                    'acceleration': 0.10,
                    'roc': 0.10,
                    'macd_diff': 0.15,
                    'price_vs_ma': 0.10,
                    'rsi': 0.05,
                    'bb_position': 0.05,
                }
        elif regime == MarketRegime.RANGING:
            # 震荡市场: 重均值回归, AR 模型的均值回归特性更适合
            if has_ar:
                weights = {
                    'ar_prediction': 0.30,     # AR 模型在震荡市更重要
                    'momentum_micro': 0.04,
                    'momentum_short': 0.04,
                    'momentum_medium': 0.04,
                    'acceleration': 0.08,
                    'roc': 0.04,
                    'macd_diff': 0.10,
                    'price_vs_ma': 0.08,
                    'rsi': 0.16,
                    'bb_position': 0.12,
                }
            else:
                weights = {
                    'momentum_micro': 0.05,
                    'momentum_short': 0.05,
                    'momentum_medium': 0.05,
                    'acceleration': 0.10,
                    'roc': 0.05,
                    'macd_diff': 0.15,
                    'price_vs_ma': 0.10,
                    'rsi': 0.25,
                    'bb_position': 0.20,
                }
        else:  # HIGH_VOLATILITY
            # 高波动: 微观因子更重要，快速反应
            if has_ar:
                weights = {
                    'ar_prediction': 0.20,
                    'momentum_micro': 0.18,
                    'momentum_short': 0.08,
                    'momentum_medium': 0.04,
                    'acceleration': 0.12,
                    'roc': 0.08,
                    'macd_diff': 0.08,
                    'price_vs_ma': 0.04,
                    'rsi': 0.10,
                    'bb_position': 0.08,
                }
            else:
                weights = {
                    'momentum_micro': 0.20,
                    'momentum_short': 0.10,
                    'momentum_medium': 0.05,
                    'acceleration': 0.15,
                    'roc': 0.10,
                    'macd_diff': 0.10,
                    'price_vs_ma': 0.05,
                    'rsi': 0.15,
                    'bb_position': 0.10,
                }
        
        score = 0
        for factor, weight in weights.items():
            if factor in z_scores:
                z = z_scores[factor]
                # 处理 NaN
                if pd.isna(z):
                    continue
                # 均值回归因子要反转方向 (超买做空，超卖做多)
                if factor in ['rsi', 'bb_position']:
                    score -= z * weight
                else:
                    score += z * weight
        
        return score
    
    def _estimate_edge(self, df: pd.DataFrame, composite_score: float, regime: MarketRegime) -> Tuple[float, float]:
        """
        估计统计边际 - 改进版
        
        基于历史数据验证信号有效性:
        - 回测最近 N 个信号的实际胜率
        - 计算实际期望收益
        """
        returns = df['close'].pct_change()
        
        # 如果有 AR 模型，用模型预测验证
        if self.ar_model is not None:
            win_prob, expected_return = self._validate_ar_signals(df, returns)
            if win_prob is not None:
                return win_prob, expected_return
        
        # 否则用传统方法：基于因子信号的历史验证
        # 计算历史上类似信号的表现
        lookback = min(100, len(df) - 20)
        if lookback < 30:
            # 数据不足，返回保守估计
            return 0.5, 0.0
        
        # 模拟历史信号
        signal_strength = min(abs(composite_score) / 3, 1)
        
        # 根据市场状态调整基础胜率
        regime_base_rates = {
            MarketRegime.TRENDING_UP: 0.52 if composite_score > 0 else 0.48,
            MarketRegime.TRENDING_DOWN: 0.48 if composite_score > 0 else 0.52,
            MarketRegime.RANGING: 0.50,
            MarketRegime.HIGH_VOLATILITY: 0.45,  # 高波动更难预测
        }
        base_win_rate = regime_base_rates.get(regime, 0.5)
        
        # 信号强度调整 (更保守)
        win_prob = base_win_rate + signal_strength * 0.08  # 最高只加 8%
        win_prob = min(0.60, max(0.40, win_prob))  # 限制在 40-60%
        
        # 计算实际波动率
        recent_vol = returns.rolling(20).std().iloc[-1]
        avg_move = returns.abs().rolling(20).mean().iloc[-1]
        
        # 基于 ATR 的动态盈亏比
        if regime == MarketRegime.TRENDING_UP or regime == MarketRegime.TRENDING_DOWN:
            rr_ratio = 1.8  # 趋势市可以追求更高盈亏比
        elif regime == MarketRegime.RANGING:
            rr_ratio = 1.2  # 震荡市盈亏比降低
        else:
            rr_ratio = 1.0  # 高波动市保守
        
        # 期望收益
        expected_return = win_prob * avg_move * rr_ratio - (1 - win_prob) * avg_move
        
        return win_prob, expected_return
    
    def _validate_ar_signals(self, df: pd.DataFrame, returns: pd.Series) -> Tuple[Optional[float], Optional[float]]:
        """
        用 AR 模型验证历史信号

        回测最近 N 个预测的实际表现
        """
        if not TORCH_AVAILABLE or self.ar_model is None or len(df) < 50:
            return None, None
        
        try:
            # 计算 log returns
            log_returns = np.log(1 + returns).dropna()
            if len(log_returns) < self.ar_n_lags + 30:
                return None, None
            
            # 回测最近 30 个预测
            predictions = []
            actuals = []
            
            for i in range(30, 0, -1):
                idx = -i - 1
                if abs(idx) > len(log_returns) - self.ar_n_lags:
                    continue
                
                # 获取特征
                features = log_returns.iloc[idx - self.ar_n_lags:idx].values[::-1].copy()
                if len(features) < self.ar_n_lags:
                    continue
                
                # 预测
                X = torch.tensor(features, dtype=torch.float32)
                with torch.no_grad():
                    pred = self.ar_model(X).item()
                
                # 实际值
                actual = log_returns.iloc[idx] if idx < -1 else log_returns.iloc[-1]
                
                predictions.append(pred)
                actuals.append(actual)
            
            if len(predictions) < 20:
                return None, None
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # 计算胜率
            correct = (np.sign(predictions) == np.sign(actuals))
            win_rate = correct.mean()
            
            # 计算实际期望收益
            trade_returns = np.sign(predictions) * actuals
            expected_return = trade_returns.mean()
            
            return win_rate, expected_return
            
        except Exception as e:
            log.warning(f"AR validation failed: {e}")
            return None, None
    
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
        
        # 因子名称映射 (更友好的显示)
        factor_names = {
            'momentum_micro': 'Micro Mom / 微观动量',
            'momentum_short': 'Short Mom / 短期动量',
            'momentum_medium': 'Medium Mom / 中期动量',
            'acceleration': 'Acceleration / 加速度',
            'roc': 'ROC / 变化率',
            'rsi': 'RSI',
            'bb_position': 'BB Position / 布林带',
            'macd_diff': 'MACD',
            'price_vs_ma': 'Price vs MA / 均线偏离',
            'volume_ratio': 'Volume / 成交量',
            'ar_prediction': 'AR Model / AR模型预测',
        }
        
        # 均值回归因子 (反向解读)
        mean_reversion_factors = ['rsi', 'bb_position']
        
        # 市场状态
        regime_names = {
            MarketRegime.TRENDING_UP: "Uptrend / 上涨趋势",
            MarketRegime.TRENDING_DOWN: "Downtrend / 下跌趋势",
            MarketRegime.RANGING: "Ranging / 震荡市",
            MarketRegime.HIGH_VOLATILITY: "High Volatility / 高波动",
        }
        signals.append(f"Market Regime / 市场状态: {regime_names[regime]}")
        
        # 显著因子 (Z > 1.5)
        for factor, z in z_scores.items():
            if pd.isna(z):
                continue
            if abs(z) > 1.5:
                display_name = factor_names.get(factor, factor)
                # 均值回归因子反向解读
                if factor in mean_reversion_factors:
                    direction = "bearish / 看跌" if z > 0 else "bullish / 看涨"
                else:
                    direction = "bullish / 看涨" if z > 0 else "bearish / 看跌"
                signals.append(f"{display_name}: Z={z:.2f} ({direction})")
        
        # 警告
        if vol_pct > 80:
            warnings.append(f"Volatility at {vol_pct:.0f}th percentile / 波动率处于{vol_pct:.0f}%分位 - Reduce size / 减小仓位")
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            warnings.append("High volatility regime / 高波动状态 - Use tight stops / 使用紧止损")
        
        # 加速度警告
        if 'acceleration' in z_scores and abs(z_scores['acceleration']) > 2:
            if z_scores['acceleration'] > 2:
                warnings.append("Strong acceleration / 强加速 - Momentum building / 动量增强")
            else:
                warnings.append("Strong deceleration / 强减速 - Reversal possible / 可能反转")
        
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

