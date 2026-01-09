# src/utils/charts.py
"""
图表生成模块 - 生成 K 线图和技术指标图表
"""
import io
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategy.indicators import calculate_all_indicators


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    show_volume: bool = True,
    show_indicators: bool = True,
    style: str = "nightclouds"
) -> io.BytesIO:
    """
    生成 K 线图
    
    Args:
        df: OHLCV DataFrame (需要 DatetimeIndex)
        symbol: 交易对名称
        timeframe: 时间周期
        show_volume: 是否显示成交量
        show_indicators: 是否显示技术指标
        style: 图表样式 (nightclouds, charles, yahoo, etc.)
        
    Returns:
        BytesIO 图片数据
    """
    # 确保有正确的列名
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # 计算指标
    if show_indicators:
        df = calculate_all_indicators(df)
    
    # 创建附加图表
    addplots = []
    
    if show_indicators:
        # 布林带
        if 'bb_upper' in df.columns:
            addplots.append(mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--', width=0.7))
            addplots.append(mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--', width=0.7))
        
        # 均线
        if 'sma_20' in df.columns:
            addplots.append(mpf.make_addplot(df['sma_20'], color='orange', width=1))
        if 'sma_50' in df.columns:
            addplots.append(mpf.make_addplot(df['sma_50'], color='blue', width=1))
    
    # 自定义样式
    mc = mpf.make_marketcolors(
        up='#26a69a',      # 上涨颜色 (绿)
        down='#ef5350',    # 下跌颜色 (红)
        edge='inherit',
        wick='inherit',
        volume='inherit'
    )
    
    s = mpf.make_mpf_style(
        base_mpf_style=style,
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#2a2a2a',
        facecolor='#1a1a2e',
        figcolor='#1a1a2e',
        rc={
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white'
        }
    )
    
    # 创建图表
    buf = io.BytesIO()
    
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=s,
        title=f'\n{symbol} - {timeframe}',
        ylabel='Price (USDT)',
        ylabel_lower='Volume',
        volume=show_volume,
        addplot=addplots if addplots else None,
        figsize=(12, 8),
        returnfig=True,
        tight_layout=True
    )
    
    # 保存到内存
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_indicator_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h"
) -> io.BytesIO:
    """
    生成带 RSI 和 MACD 的完整图表
    
    Returns:
        BytesIO 图片数据
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = calculate_all_indicators(df)
    
    # 创建子图
    fig = plt.figure(figsize=(14, 10), facecolor='#1a1a2e')
    
    # 价格图 (占 50%)
    ax1 = fig.add_axes([0.1, 0.55, 0.85, 0.4])
    ax1.set_facecolor('#1a1a2e')
    
    # 绘制 K 线
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]
    
    # 上涨 K 线
    ax1.bar(up.index, up['close'] - up['open'], bottom=up['open'], color='#26a69a', width=0.02)
    ax1.bar(up.index, up['high'] - up['close'], bottom=up['close'], color='#26a69a', width=0.005)
    ax1.bar(up.index, up['low'] - up['open'], bottom=up['open'], color='#26a69a', width=0.005)
    
    # 下跌 K 线
    ax1.bar(down.index, down['close'] - down['open'], bottom=down['open'], color='#ef5350', width=0.02)
    ax1.bar(down.index, down['high'] - down['open'], bottom=down['open'], color='#ef5350', width=0.005)
    ax1.bar(down.index, down['low'] - down['close'], bottom=down['close'], color='#ef5350', width=0.005)
    
    # 均线
    ax1.plot(df.index, df['sma_20'], color='orange', linewidth=1, label='SMA20')
    ax1.plot(df.index, df['sma_50'], color='#2196f3', linewidth=1, label='SMA50')
    
    # 布林带
    ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
    
    ax1.set_title(f'{symbol} - {timeframe}', color='white', fontsize=14)
    ax1.set_ylabel('Price', color='white')
    ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2)
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    
    # RSI 图 (占 20%)
    ax2 = fig.add_axes([0.1, 0.30, 0.85, 0.2])
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(df.index, df['rsi'], color='#9c27b0', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', color='white')
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2)
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    
    # MACD 图 (占 20%)
    ax3 = fig.add_axes([0.1, 0.05, 0.85, 0.2])
    ax3.set_facecolor('#1a1a2e')
    ax3.plot(df.index, df['macd'], color='#2196f3', linewidth=1, label='MACD')
    ax3.plot(df.index, df['macd_signal'], color='orange', linewidth=1, label='Signal')
    
    # MACD 柱状图
    colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df['macd_hist']]
    ax3.bar(df.index, df['macd_hist'], color=colors, width=0.02, alpha=0.7)
    
    ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    ax3.set_ylabel('MACD', color='white')
    ax3.set_xlabel('Time', color='white')
    ax3.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2)
    ax3.spines['bottom'].set_color('white')
    ax3.spines['top'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['right'].set_color('white')
    
    # 保存到内存
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def create_simple_price_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h"
) -> io.BytesIO:
    """
    生成简单价格线图
    """
    df = df.copy()
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    ax.plot(df.index, df['close'], color='#2196f3', linewidth=1.5)
    ax.fill_between(df.index, df['close'].min(), df['close'], alpha=0.2, color='#2196f3')
    
    ax.set_title(f'{symbol} - {timeframe}', color='white', fontsize=14)
    ax.set_ylabel('Price (USDT)', color='white')
    ax.set_xlabel('Time', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    plt.close(fig)
    
    return buf

