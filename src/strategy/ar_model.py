# src/strategy/ar_model.py
"""
自回归 (AR) 模型用于预测 log return
基于 build-a-quant-trading-strategy 教程
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, List, Tuple
from dataclasses import dataclass


class LinearARModel(nn.Module):
    """
    简单线性 AR 模型
    y_hat = w1 * lag_1 + w2 * lag_2 + ... + wn * lag_n + bias
    """
    def __init__(self, n_lags: int):
        super().__init__()
        self.n_lags = n_lags
        self.linear = nn.Linear(n_lags, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def get_weights(self) -> Tuple[np.ndarray, float]:
        """获取权重和偏置"""
        weights = self.linear.weight.detach().cpu().numpy().flatten()
        bias = self.linear.bias.detach().cpu().numpy().item()
        return weights, bias


class StreamingLogReturn:
    """流式计算 log return"""
    
    def __init__(self):
        self._prev_price: Optional[float] = None
    
    def on_price(self, price: float) -> Optional[float]:
        """输入价格，返回 log return"""
        if self._prev_price is None:
            self._prev_price = price
            return None
        
        log_ret = np.log(price / self._prev_price)
        self._prev_price = price
        return log_ret
    
    def reset(self):
        self._prev_price = None


class StreamingARPredictor:
    """
    流式 AR 预测器
    实时处理价格数据并生成预测
    """
    
    def __init__(self, model: LinearARModel):
        self.model = model
        self.model.eval()
        self.n_lags = model.n_lags
        
        # 滑动窗口存储 log returns
        self._log_returns = deque(maxlen=self.n_lags)
        self._log_return_calc = StreamingLogReturn()
    
    def on_price(self, price: float) -> Optional[float]:
        """
        处理新价格，返回预测的下一期 log return
        
        Args:
            price: 当前价格
            
        Returns:
            预测的 log return，如果数据不足返回 None
        """
        log_ret = self._log_return_calc.on_price(price)
        
        if log_ret is None:
            return None
        
        # 最新的 log return 放在最前面 (lag_1)
        self._log_returns.appendleft(log_ret)
        
        if len(self._log_returns) < self.n_lags:
            return None
        
        # 预测
        X = torch.tensor(list(self._log_returns), dtype=torch.float32)
        with torch.no_grad():
            y_hat = self.model(X)
        
        return y_hat.item()
    
    def reset(self):
        """重置状态"""
        self._log_returns.clear()
        self._log_return_calc.reset()
    
    def is_ready(self) -> bool:
        """是否有足够数据进行预测"""
        return len(self._log_returns) >= self.n_lags


@dataclass
class ARTrainingResult:
    """AR 模型训练结果"""
    model: LinearARModel
    train_loss: float
    test_loss: float
    weights: np.ndarray
    bias: float
    win_rate: float
    sharpe: float
    n_lags: int


def prepare_ar_features(
    prices: np.ndarray, 
    n_lags: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备 AR 模型的特征和目标
    
    Args:
        prices: 价格序列
        n_lags: 滞后期数
        forecast_horizon: 预测周期
        
    Returns:
        X: 特征矩阵 (n_samples, n_lags)
        y: 目标向量 (n_samples,)
    """
    # 计算 log returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # 创建滞后特征
    X_list = []
    y_list = []
    
    for i in range(n_lags, len(log_returns) - forecast_horizon + 1):
        # 特征: [lag_1, lag_2, ..., lag_n] (最新在前)
        features = log_returns[i-n_lags:i][::-1]  # 反转使最新在前
        target = log_returns[i + forecast_horizon - 1]
        
        X_list.append(features)
        y_list.append(target)
    
    return np.array(X_list), np.array(y_list)


def train_ar_model(
    prices: np.ndarray,
    n_lags: int = 3,
    forecast_horizon: int = 1,
    test_size: float = 0.25,
    n_epochs: int = 1000,
    lr: float = 0.01,
    verbose: bool = True
) -> ARTrainingResult:
    """
    训练 AR 模型
    
    Args:
        prices: 价格序列
        n_lags: 滞后期数
        forecast_horizon: 预测周期
        test_size: 测试集比例
        n_epochs: 训练轮数
        lr: 学习率
        verbose: 是否打印训练过程
        
    Returns:
        ARTrainingResult
    """
    # 准备数据
    X, y = prepare_ar_features(prices, n_lags, forecast_horizon)
    
    # 时间序列分割
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    # 创建模型
    model = LinearARModel(n_lags)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    train_losses = []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if verbose and (epoch + 1) % (n_epochs // 10) == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}")
    
    # 评估
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t)
        test_pred = model(X_test_t)
        
        train_loss = criterion(train_pred, y_train_t).item()
        test_loss = criterion(test_pred, y_test_t).item()
    
    # 计算交易指标
    y_hat = test_pred.numpy().flatten()
    y_true = y_test
    
    # 胜率: 预测方向正确的比例
    correct_direction = (np.sign(y_hat) == np.sign(y_true))
    win_rate = correct_direction.mean()
    
    # 交易收益: 按预测方向交易的收益
    trade_returns = np.sign(y_hat) * y_true
    
    # Sharpe ratio (假设无风险利率为0)
    if trade_returns.std() > 0:
        sharpe = trade_returns.mean() / trade_returns.std() * np.sqrt(252 * 24)  # 年化 (假设1小时数据)
    else:
        sharpe = 0
    
    weights, bias = model.get_weights()
    
    if verbose:
        print(f"\n训练完成:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Weights: {weights}")
        print(f"  Bias: {bias:.6f}")
    
    return ARTrainingResult(
        model=model,
        train_loss=train_loss,
        test_loss=test_loss,
        weights=weights,
        bias=bias,
        win_rate=win_rate,
        sharpe=sharpe,
        n_lags=n_lags
    )


def save_ar_model(model: LinearARModel, path: str):
    """保存模型"""
    torch.save({
        'state_dict': model.state_dict(),
        'n_lags': model.n_lags
    }, path)


def load_ar_model(path: str) -> LinearARModel:
    """加载模型"""
    checkpoint = torch.load(path, weights_only=True)
    model = LinearARModel(checkpoint['n_lags'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
