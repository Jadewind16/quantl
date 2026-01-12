# src/discord_bot.py
"""
Discord Trading Bot
支持命令交互 + 自动信号提醒
"""
import os
import asyncio
import discord
from discord.ext import commands, tasks
from discord import app_commands
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tradingview import TradingViewDataFetcher
from src.strategy.signals import SignalGenerator, SignalType
from src.strategy.indicators import calculate_all_indicators
from src.strategy.advisor import TradingAdvisor, Direction, RiskLevel
from src.strategy.quant_advisor import QuantAdvisor, MarketRegime
from src.utils.logger import log

# AR 模型功能是可选的 (需要 PyTorch)
try:
    from src.strategy.ar_model import LinearARModel, load_ar_model, train_ar_model, save_ar_model
    AR_MODEL_AVAILABLE = True
except ImportError:
    AR_MODEL_AVAILABLE = False
    LinearARModel = None
    load_ar_model = None
    train_ar_model = None
    save_ar_model = None
from src.utils.charts import create_candlestick_chart, create_indicator_chart

# 加载环境变量
load_dotenv()


def fmt_price(price: float, sig_figs: int = 5) -> str:
    """
    格式化价格，显示 4-6 位有效数字
    
    Examples:
        94123.456 -> $94,123
        0.00012345 -> $0.0001235
        1.2345 -> $1.2345
    """
    if price == 0:
        return "$0"
    
    abs_price = abs(price)
    
    if abs_price >= 10000:
        # 大数：显示整数部分 + 1-2位小数
        return f"${price:,.1f}"
    elif abs_price >= 100:
        # 中等数：2位小数
        return f"${price:,.2f}"
    elif abs_price >= 1:
        # 1-100：4位小数
        return f"${price:.4f}"
    elif abs_price >= 0.01:
        # 小数：5位小数
        return f"${price:.5f}"
    else:
        # 非常小的数：用有效数字格式
        return f"${price:.{sig_figs}g}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """格式化百分比"""
    return f"{value:.{decimals}f}%"


class TradingBot(commands.Bot):
    """Discord Trading Bot"""
    
    def __init__(self):
        # 设置 intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            description="Perpetual Futures Signal Bot"
        )
        
        # 交易组件 - 使用 TradingView 数据源
        self.fetcher = TradingViewDataFetcher()
        self.signal_generator = SignalGenerator()
        self.advisor = TradingAdvisor()
        
        # AR 模型路径
        self.ar_model_path = Path(__file__).parent.parent / "data" / "ar_model.pth"
        self.ar_model: Optional[LinearARModel] = None
        self.ar_n_lags = 3
        
        # 尝试加载 AR 模型
        self._load_ar_model()
        
        # 初始化 QuantAdvisor (带 AR 模型)
        self.quant_advisor = QuantAdvisor(
            ar_model=self.ar_model,
            ar_n_lags=self.ar_n_lags
        )
        
        # 监控配置
        self.watched_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        self.alert_channel_id: Optional[int] = None
        self.sent_signals = {}
        self.signal_cooldown = 3600
    
    def _load_ar_model(self) -> bool:
        """加载 AR 模型"""
        if not AR_MODEL_AVAILABLE:
            log.warning("PyTorch not available - AR model features disabled")
            self.ar_model = None
            return False
        
        try:
            if self.ar_model_path.exists():
                self.ar_model = load_ar_model(str(self.ar_model_path))
                self.ar_n_lags = self.ar_model.n_lags
                log.info(f"Loaded AR model from {self.ar_model_path} (n_lags={self.ar_n_lags})")
                return True
            else:
                log.warning(f"AR model not found at {self.ar_model_path}")
                self.ar_model = None
                return False
        except Exception as e:
            log.error(f"Failed to load AR model: {e}")
            self.ar_model = None
            return False
    
    def reload_ar_model(self) -> bool:
        """重新加载 AR 模型并更新 QuantAdvisor"""
        success = self._load_ar_model()
        if success:
            self.quant_advisor = QuantAdvisor(
                ar_model=self.ar_model,
                ar_n_lags=self.ar_n_lags
            )
            log.info("QuantAdvisor updated with new AR model")
        return success
    
    def train_new_model(self, symbol: str, timeframe: str, n_lags: int = 3, limit: int = 500) -> dict:
        """训练新的 AR 模型"""
        if not AR_MODEL_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed - cannot train model"}
        
        try:
            # 获取数据
            df = self.fetcher.get_klines(symbol, timeframe, limit=limit)
            if df is None or len(df) < 100:
                return {"success": False, "error": "Insufficient data"}
            
            prices = df['close'].values
            
            # 训练模型
            result = train_ar_model(
                prices=prices,
                n_lags=n_lags,
                forecast_horizon=1,
                test_size=0.25,
                n_epochs=1000,
                lr=0.01,
                verbose=False
            )
            
            # 保存模型
            save_ar_model(result.model, str(self.ar_model_path))
            
            # 重新加载
            self.reload_ar_model()
            
            return {
                "success": True,
                "win_rate": result.win_rate,
                "sharpe": result.sharpe,
                "weights": result.weights.tolist(),
                "bias": result.bias,
                "n_lags": n_lags
            }
        except Exception as e:
            log.error(f"Failed to train model: {e}")
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> dict:
        """获取当前模型信息"""
        if self.ar_model is None:
            return {"loaded": False}
        
        weights, bias = self.ar_model.get_weights()
        return {
            "loaded": True,
            "n_lags": self.ar_model.n_lags,
            "weights": weights.tolist(),
            "bias": float(bias),
            "path": str(self.ar_model_path)
        }
    
    async def setup_hook(self):
        """Bot 启动时执行"""
        # 添加 Cog
        await self.add_cog(TradingCommands(self))
        
        log.info("Bot setup complete")
    
    async def on_ready(self):
        """Bot 连接成功"""
        log.info(f"Logged in as {self.user.name} ({self.user.id})")
        log.info(f"Connected to {len(self.guilds)} guild(s)")
        
        # 同步命令到所有已加入的服务器 (立即生效)
        for guild in self.guilds:
            try:
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                log.info(f"Synced commands to guild: {guild.name}")
            except Exception as e:
                log.error(f"Failed to sync to {guild.name}: {e}")
        
        # 启动自动检查任务
        if not self.auto_check_signals.is_running():
            self.auto_check_signals.start()
    
    @tasks.loop(minutes=60)
    async def auto_check_signals(self):
        """每小时自动检查信号"""
        if not self.alert_channel_id:
            return
        
        channel = self.get_channel(self.alert_channel_id)
        if not channel:
            return
        
        log.info("Running auto signal check...")
        
        for symbol in self.watched_symbols:
            try:
                signals = await self._check_symbol_signals(symbol)
                
                for signal in signals:
                    if self._should_send_signal(signal):
                        embed = self._create_signal_embed(signal)
                        await channel.send(embed=embed)
                        self._mark_signal_sent(signal)
                        
            except Exception as e:
                log.error(f"Auto check error for {symbol}: {e}")
    
    async def _check_symbol_signals(self, symbol: str):
        """检查单个交易对信号"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行阻塞操作
        df = await loop.run_in_executor(
            None, 
            lambda: self.fetcher.get_klines(symbol, "1h", limit=100)
        )
        
        signals = self.signal_generator.analyze(df, symbol)
        return [s for s in signals if s.signal_type != SignalType.NEUTRAL]
    
    def _should_send_signal(self, signal) -> bool:
        """检查是否应该发送信号"""
        key = f"{signal.symbol}_{signal.strategy}_{signal.signal_type.value}"
        
        if key in self.sent_signals:
            last_sent = self.sent_signals[key]
            elapsed = (datetime.now() - last_sent).total_seconds()
            if elapsed < self.signal_cooldown:
                return False
        
        return True
    
    def _mark_signal_sent(self, signal):
        """标记信号已发送"""
        key = f"{signal.symbol}_{signal.strategy}_{signal.signal_type.value}"
        self.sent_signals[key] = datetime.now()
    
    def _create_signal_embed(self, signal) -> discord.Embed:
        """创建信号 Embed"""
        colors = {
            SignalType.LONG: discord.Color.green(),
            SignalType.SHORT: discord.Color.red(),
            SignalType.CLOSE: discord.Color.yellow(),
            SignalType.NEUTRAL: discord.Color.greyple()
        }
        
        embed = discord.Embed(
            title=f"[{signal.signal_type.value}] {signal.symbol}",
            color=colors.get(signal.signal_type, discord.Color.blue()),
            timestamp=signal.timestamp
        )
        
        embed.add_field(name="Strategy", value=signal.strategy, inline=True)
        embed.add_field(name="Price", value=fmt_price(signal.price), inline=True)
        embed.add_field(name="Confidence", value=f"{signal.confidence:.0%}", inline=True)
        embed.add_field(name="Reason", value=signal.reason, inline=False)
        
        # 指标
        indicator_text = "\n".join([
            f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in signal.indicators.items()
        ])
        embed.add_field(name="Indicators", value=f"```\n{indicator_text}\n```", inline=False)
        
        embed.set_footer(text="Trading Signal Bot")
        
        return embed


class TradingCommands(commands.Cog):
    """交易命令"""
    
    def __init__(self, bot: TradingBot):
        self.bot = bot
    
    @app_commands.command(name="price", description="Get current price")
    @app_commands.describe(symbol="Trading pair (e.g., BTC, ETH)")
    async def price(self, interaction: discord.Interaction, symbol: str = "BTC"):
        """获取当前价格"""
        await interaction.response.defer()
        
        # 格式化交易对
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        try:
            loop = asyncio.get_event_loop()
            price_data = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_price(formatted_symbol)
            )
            
            embed = discord.Embed(
                title=f"{symbol.upper()}/USDT",
                color=discord.Color.blue()
            )
            
            change_color = "green" if price_data['change_24h'] >= 0 else "red"
            change_sign = "+" if price_data['change_24h'] >= 0 else ""
            
            embed.add_field(
                name="Price", 
                value=fmt_price(price_data['price']), 
                inline=True
            )
            embed.add_field(
                name="24h Change", 
                value=f"{change_sign}{price_data['change_24h']:.2f}%", 
                inline=True
            )
            embed.add_field(
                name="24h High", 
                value=fmt_price(price_data['high_24h']), 
                inline=True
            )
            embed.add_field(
                name="24h Low", 
                value=fmt_price(price_data['low_24h']), 
                inline=True
            )
            
            embed.set_footer(text=f"OKX | {price_data['timestamp']}")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="funding", description="Get funding rate")
    @app_commands.describe(symbol="Trading pair (e.g., BTC, ETH)")
    async def funding(self, interaction: discord.Interaction, symbol: str = "BTC"):
        """获取资金费率"""
        await interaction.response.defer()
        
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        try:
            loop = asyncio.get_event_loop()
            funding_data = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_funding_rate(formatted_symbol)
            )
            
            sentiment_emoji = {
                "bullish": "Bull",
                "bearish": "Bear",
                "neutral": "Neutral"
            }
            
            embed = discord.Embed(
                title=f"Funding Rate - {symbol.upper()}/USDT",
                color=discord.Color.gold()
            )
            
            embed.add_field(
                name="Rate", 
                value=f"{funding_data['funding_rate_percent']:.4f}%", 
                inline=True
            )
            embed.add_field(
                name="Sentiment", 
                value=sentiment_emoji.get(funding_data['sentiment'], funding_data['sentiment']), 
                inline=True
            )
            
            if funding_data['next_funding_time']:
                embed.add_field(
                    name="Next Funding", 
                    value=funding_data['next_funding_time'], 
                    inline=False
                )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="signals", description="Check current signals")
    @app_commands.describe(symbol="Trading pair (e.g., BTC, ETH)")
    async def signals(self, interaction: discord.Interaction, symbol: str = "BTC"):
        """检查当前信号"""
        await interaction.response.defer()
        
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        try:
            signals = await self.bot._check_symbol_signals(formatted_symbol)
            
            if not signals:
                await interaction.followup.send(f"No signals for {symbol.upper()}/USDT at the moment.")
                return
            
            for signal in signals[:5]:  # 最多显示5个
                embed = self.bot._create_signal_embed(signal)
                await interaction.followup.send(embed=embed)
                
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="indicators", description="Get technical indicators")
    @app_commands.describe(symbol="Trading pair (e.g., BTC, ETH)")
    async def indicators(self, interaction: discord.Interaction, symbol: str = "BTC"):
        """获取技术指标"""
        await interaction.response.defer()
        
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_klines(formatted_symbol, "1h", limit=100)
            )
            
            df = calculate_all_indicators(df)
            latest = df.iloc[-1]
            
            embed = discord.Embed(
                title=f"Technical Indicators - {symbol.upper()}/USDT (1H)",
                color=discord.Color.purple()
            )
            
            # 趋势指标
            embed.add_field(
                name="Moving Averages",
                value=f"SMA20: {fmt_price(latest['sma_20'])}\nSMA50: {fmt_price(latest['sma_50'])}\nEMA12: {fmt_price(latest['ema_12'])}",
                inline=True
            )
            
            # 动量指标
            embed.add_field(
                name="Momentum",
                value=f"RSI: {latest['rsi']:.1f}\nStoch K: {latest['stoch_k']:.1f}\nStoch D: {latest['stoch_d']:.1f}",
                inline=True
            )
            
            # MACD
            embed.add_field(
                name="MACD",
                value=f"MACD: {latest['macd']:.2f}\nSignal: {latest['macd_signal']:.2f}\nHist: {latest['macd_hist']:.2f}",
                inline=True
            )
            
            # 布林带
            embed.add_field(
                name="Bollinger Bands",
                value=f"Upper: {fmt_price(latest['bb_upper'])}\nMiddle: {fmt_price(latest['bb_middle'])}\nLower: {fmt_price(latest['bb_lower'])}",
                inline=True
            )
            
            # ATR
            embed.add_field(
                name="Volatility",
                value=f"ATR: {fmt_price(latest['atr'])}",
                inline=True
            )
            
            # 当前价格
            embed.add_field(
                name="Current Price",
                value=fmt_price(latest['close']),
                inline=True
            )
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="watch", description="Add symbol to watchlist")
    @app_commands.describe(symbol="Trading pair to watch (e.g., BTC, ETH)")
    async def watch(self, interaction: discord.Interaction, symbol: str):
        """添加到监控列表"""
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        if formatted_symbol not in self.bot.watched_symbols:
            self.bot.watched_symbols.append(formatted_symbol)
            await interaction.response.send_message(f"Added {symbol.upper()}/USDT to watchlist.")
        else:
            await interaction.response.send_message(f"{symbol.upper()}/USDT is already in watchlist.")
    
    @app_commands.command(name="unwatch", description="Remove symbol from watchlist")
    @app_commands.describe(symbol="Trading pair to remove (e.g., BTC, ETH)")
    async def unwatch(self, interaction: discord.Interaction, symbol: str):
        """从监控列表移除"""
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        if formatted_symbol in self.bot.watched_symbols:
            self.bot.watched_symbols.remove(formatted_symbol)
            await interaction.response.send_message(f"Removed {symbol.upper()}/USDT from watchlist.")
        else:
            await interaction.response.send_message(f"{symbol.upper()}/USDT is not in watchlist.")
    
    @app_commands.command(name="watchlist", description="Show current watchlist")
    async def watchlist(self, interaction: discord.Interaction):
        """显示监控列表"""
        if not self.bot.watched_symbols:
            await interaction.response.send_message("Watchlist is empty.")
            return
        
        symbols = [s.replace("/USDT:USDT", "") for s in self.bot.watched_symbols]
        await interaction.response.send_message(f"Watchlist: {', '.join(symbols)}")
    
    @app_commands.command(name="setalert", description="Set this channel for auto alerts")
    async def setalert(self, interaction: discord.Interaction):
        """设置自动提醒频道"""
        self.bot.alert_channel_id = interaction.channel_id
        await interaction.response.send_message(
            f"Alert channel set to #{interaction.channel.name}. "
            f"You will receive automatic signals every hour."
        )
    
    @app_commands.command(name="stopalert", description="Stop auto alerts")
    async def stopalert(self, interaction: discord.Interaction):
        """停止自动提醒"""
        self.bot.alert_channel_id = None
        await interaction.response.send_message("Auto alerts disabled.")
    
    @app_commands.command(name="chart", description="Get candlestick chart")
    @app_commands.describe(
        symbol="Trading pair (e.g., BTC, ETH)",
        timeframe="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)",
        limit="Number of candles (default: 100)"
    )
    async def chart(
        self, 
        interaction: discord.Interaction, 
        symbol: str = "BTC",
        timeframe: str = "1h",
        limit: int = 100
    ):
        """生成 K 线图"""
        await interaction.response.defer()
        
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_klines(formatted_symbol, timeframe, limit=limit)
            )
            
            # 生成图表
            chart_buf = await loop.run_in_executor(
                None,
                lambda: create_candlestick_chart(df, f"{symbol.upper()}/USDT", timeframe)
            )
            
            # 发送图片
            file = discord.File(chart_buf, filename=f"{symbol}_chart.png")
            await interaction.followup.send(file=file)
            
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="analysis", description="Get full technical analysis chart")
    @app_commands.describe(
        symbol="Trading pair (e.g., BTC, ETH)",
        timeframe="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)"
    )
    async def analysis(
        self, 
        interaction: discord.Interaction, 
        symbol: str = "BTC",
        timeframe: str = "1h"
    ):
        """生成完整技术分析图表 (K线 + RSI + MACD)"""
        await interaction.response.defer()
        
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_klines(formatted_symbol, timeframe, limit=100)
            )
            
            # 生成图表
            chart_buf = await loop.run_in_executor(
                None,
                lambda: create_indicator_chart(df, f"{symbol.upper()}/USDT", timeframe)
            )
            
            # 发送图片
            file = discord.File(chart_buf, filename=f"{symbol}_analysis.png")
            await interaction.followup.send(file=file)
            
        except Exception as e:
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="advice", description="Get trading advice / 获取交易建议")
    @app_commands.describe(
        symbol="Trading pair (e.g., BTC, ETH)",
        timeframe="Timeframe for analysis (1h, 4h, 1d)"
    )
    async def advice(self, interaction: discord.Interaction, symbol: str = "BTC", timeframe: str = "1h"):
        """获取交易建议"""
        await interaction.response.defer()
        
        try:
            # 格式化交易对
            formatted_symbol = f"{symbol.upper()}/USDT:USDT"
            
            # 获取数据
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_klines(formatted_symbol, timeframe, limit=100)
            )
            
            if df is None or len(df) < 50:
                await interaction.followup.send("Insufficient data / 数据不足. Need at least 50 candles / 需要至少50根K线")
                return
            
            # 生成交易建议
            advice = self.bot.advisor.analyze(df, formatted_symbol)
            data = advice.to_discord_embed_data()
            
            # 根据方向选择颜色和文字 (双语)
            if advice.direction == Direction.LONG:
                color = 0x00FF00  # 绿色
                direction_emoji = "[LONG]"
                direction_text = "LONG (Buy) / 做多"
            elif advice.direction == Direction.SHORT:
                color = 0xFF0000  # 红色
                direction_emoji = "[SHORT]"
                direction_text = "SHORT (Sell) / 做空"
            else:
                color = 0x808080  # 灰色
                direction_emoji = "[WAIT]"
                direction_text = "NEUTRAL (Wait) / 观望"
            
            # 风险等级双语
            risk_text = {
                "LOW": "LOW / 低风险",
                "MEDIUM": "MEDIUM / 中风险",
                "HIGH": "HIGH / 高风险"
            }
            
            # 创建 Embed
            embed = discord.Embed(
                title=f"{direction_emoji} {symbol.upper()}/USDT Trading Advice / 交易建议",
                color=color,
                timestamp=datetime.now()
            )
            
            # 基本信息
            embed.add_field(
                name="Direction / 方向",
                value=f"**{direction_text}**",
                inline=True
            )
            embed.add_field(
                name="Confidence / 置信度",
                value=f"**{data['confidence']:.0f}%**",
                inline=True
            )
            embed.add_field(
                name="Risk / 风险",
                value=f"**{risk_text.get(data['risk_level'], data['risk_level'])}**",
                inline=True
            )
            
            # 价格信息
            if advice.direction != Direction.NEUTRAL:
                price_info = (
                    f"```\n"
                    f"Current/当前:  {fmt_price(data['current_price']):>14}\n"
                    f"Entry/入场:    {fmt_price(data['entry_price']):>14}\n"
                    f"Stop/止损:     {fmt_price(data['stop_loss']):>14}\n"
                    f"TP1/止盈1:     {fmt_price(data['take_profit_1']):>14}\n"
                    f"TP2/止盈2:     {fmt_price(data['take_profit_2']):>14}\n"
                    f"TP3/止盈3:     {fmt_price(data['take_profit_3']):>14}\n"
                    f"```"
                )
                embed.add_field(
                    name="Price Levels / 价格位置",
                    value=price_info,
                    inline=False
                )
                
                # 风险收益比
                embed.add_field(
                    name="R/R Ratio / 盈亏比",
                    value=f"**{data['risk_reward']:.1f}:1**",
                    inline=True
                )
            else:
                embed.add_field(
                    name="Current Price / 当前价格",
                    value=f"**{fmt_price(data['current_price'])}**",
                    inline=False
                )
            
            # 分析评分
            score_info = (
                f"```\n"
                f"Trend/趋势:     {data['trend_score']:>4} {'[+]' if data['trend_score'] > 0 else '[-]' if data['trend_score'] < 0 else '[=]'}\n"
                f"Momentum/动量:  {data['momentum_score']:>4} {'[+]' if data['momentum_score'] > 0 else '[-]' if data['momentum_score'] < 0 else '[=]'}\n"
                f"Volatility/波动:{data['volatility_score']:>4} {'[!]' if data['volatility_score'] > 50 else '[OK]'}\n"
                f"```"
            )
            embed.add_field(
                name="Scores / 评分",
                value=score_info,
                inline=False
            )
            
            # 分析理由
            if data['reasons']:
                reasons_text = "\n".join([f"- {r}" for r in data['reasons'][:5]])
                embed.add_field(
                    name="Analysis / 分析",
                    value=reasons_text,
                    inline=False
                )
            
            # 警告
            if data['warnings']:
                warnings_text = "\n".join([f"- {w}" for w in data['warnings']])
                embed.add_field(
                    name="Warnings / 警告",
                    value=warnings_text,
                    inline=False
                )
            
            # 免责声明 (双语)
            embed.set_footer(text=f"Timeframe: {timeframe} | Not financial advice / 非投资建议，风险自担")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            log.error(f"Error in advice command: {e}")
            await interaction.followup.send(f"Error: {e}")
    
    @app_commands.command(name="quant", description="Quantitative analysis / 量化分析")
    @app_commands.describe(
        symbol="Trading pair (e.g., BTC, ETH)",
        timeframe="Timeframe for analysis (1h, 4h, 1d)"
    )
    async def quant(self, interaction: discord.Interaction, symbol: str = "BTC", timeframe: str = "1h"):
        """量化分析建议"""
        await interaction.response.defer()
        
        try:
            formatted_symbol = f"{symbol.upper()}/USDT:USDT"
            
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: self.bot.fetcher.get_klines(formatted_symbol, timeframe, limit=150)
            )
            
            if df is None or len(df) < 100:
                await interaction.followup.send("Insufficient data / 数据不足. Need 100+ candles / 需要100根以上K线")
                return
            
            advice = self.bot.quant_advisor.analyze(df, formatted_symbol)
            
            # 颜色
            if advice.direction.value == "LONG":
                color = 0x00FF00
                direction_text = "LONG / 做多"
            elif advice.direction.value == "SHORT":
                color = 0xFF0000
                direction_text = "SHORT / 做空"
            else:
                color = 0x808080
                direction_text = "NEUTRAL / 观望"
            
            # 市场状态
            regime_text = {
                MarketRegime.TRENDING_UP: "Uptrend / 上涨趋势",
                MarketRegime.TRENDING_DOWN: "Downtrend / 下跌趋势",
                MarketRegime.RANGING: "Ranging / 震荡市",
                MarketRegime.HIGH_VOLATILITY: "High Vol / 高波动",
            }
            
            embed = discord.Embed(
                title=f"[QUANT] {symbol.upper()}/USDT Analysis / 量化分析",
                color=color,
                timestamp=datetime.now()
            )
            
            # 方向和置信度
            embed.add_field(name="Direction / 方向", value=f"**{direction_text}**", inline=True)
            embed.add_field(name="Confidence / 置信度", value=f"**{advice.confidence:.1f}%**", inline=True)
            embed.add_field(name="Win Prob / 胜率", value=f"**{advice.win_probability:.1%}**", inline=True)
            
            # 市场状态
            embed.add_field(
                name="Market Regime / 市场状态",
                value=f"**{regime_text.get(advice.market_regime, 'Unknown')}**",
                inline=True
            )
            embed.add_field(
                name="Volatility / 波动率",
                value=f"**{advice.volatility_percentile:.0f}th percentile**",
                inline=True
            )
            embed.add_field(
                name="Position Size / 仓位",
                value=f"**{advice.position_size_pct:.1f}%**",
                inline=True
            )
            
            # 入场和止损止盈
            if advice.direction.value != "NEUTRAL":
                # 入场范围
                entry_info = (
                    f"```\n"
                    f"Entry Range / 入场范围:\n"
                    f"  {fmt_price(advice.entry_low)} - {fmt_price(advice.entry_high)}\n"
                    f"Current / 当前: {fmt_price(advice.current_price)}\n"
                    f"```"
                )
                embed.add_field(name="📍 Entry / 入场", value=entry_info, inline=True)
                
                # 止损
                sl_pct = abs(advice.stop_loss - advice.current_price) / advice.current_price * 100
                sl_info = f"```\n{fmt_price(advice.stop_loss)}\n({sl_pct:.2f}% risk)\n```"
                embed.add_field(name="🛑 Stop Loss / 止损", value=sl_info, inline=True)
                
                # 风险收益比
                rr_info = f"```\nR:R = 1:{advice.risk_reward_ratio:.2f}\n```"
                embed.add_field(name="⚖️ Risk/Reward", value=rr_info, inline=True)
                
                # 多级止盈
                tp_info = (
                    f"```\n"
                    f"TP1 (保守): {fmt_price(advice.tp1)}\n"
                    f"TP2 (标准): {fmt_price(advice.tp2)}\n"
                    f"TP3 (激进): {fmt_price(advice.tp3)}\n"
                    f"```"
                )
                embed.add_field(name="🎯 Take Profit / 止盈", value=tp_info, inline=False)
                
                # 支撑压力位
                if advice.supports or advice.resistances:
                    sr_text = "```\n"
                    if advice.resistances:
                        sr_text += "Resistance / 压力位:\n"
                        for r in advice.resistances[:3]:
                            sr_text += f"  🔴 {fmt_price(r)}\n"
                    sr_text += f"  ➡️ {fmt_price(advice.current_price)} (current)\n"
                    if advice.supports:
                        sr_text += "Support / 支撑位:\n"
                        for s in advice.supports[:3]:
                            sr_text += f"  🟢 {fmt_price(s)}\n"
                    sr_text += "```"
                    embed.add_field(name="📊 S/R Levels / 支撑压力", value=sr_text, inline=False)
            
            # Z-Scores
            z_text = "```\n"
            for factor, z in list(advice.z_scores.items())[:5]:
                bar = "+" * min(int(abs(z) * 2), 5) if z > 0 else "-" * min(int(abs(z) * 2), 5)
                z_text += f"{factor:15}: {z:+.2f} {bar}\n"
            z_text += "```"
            embed.add_field(name="Factor Z-Scores / 因子评分", value=z_text, inline=False)
            
            # 信号
            if advice.signals:
                signals_text = "\n".join([f"- {s}" for s in advice.signals[:4]])
                embed.add_field(name="Signals / 信号", value=signals_text, inline=False)
            
            # 警告
            if advice.warnings:
                warnings_text = "\n".join([f"- {w}" for w in advice.warnings])
                embed.add_field(name="Warnings / 警告", value=warnings_text, inline=False)
            
            embed.set_footer(text=f"Kelly: {advice.kelly_fraction:.1%} | Edge: {advice.statistical_edge:.2%} | {timeframe}")
            
            await interaction.followup.send(embed=embed)
            
        except Exception as e:
            log.error(f"Error in quant command: {e}")
            await interaction.followup.send(f"Error: {e}")

    @app_commands.command(name="model", description="Show AR model info / 显示模型信息")
    async def model_info(self, interaction: discord.Interaction):
        """显示当前 AR 模型信息"""
        info = self.bot.get_model_info()
        
        if not info["loaded"]:
            embed = discord.Embed(
                title="🤖 AR Model Status",
                description="No model loaded / 模型未加载",
                color=0xFF0000
            )
            embed.add_field(
                name="Hint / 提示",
                value="Use `/train_model` to train a new model\n使用 `/train_model` 训练新模型",
                inline=False
            )
        else:
            embed = discord.Embed(
                title="🤖 AR Model Info",
                color=0x00FF00
            )
            embed.add_field(name="Status / 状态", value="✅ Loaded / 已加载", inline=True)
            embed.add_field(name="Lags / 滞后期", value=f"**{info['n_lags']}**", inline=True)
            embed.add_field(name="Bias / 偏置", value=f"**{info['bias']:.6f}**", inline=True)
            
            # 权重解读
            weights_text = ""
            for i, w in enumerate(info['weights'], 1):
                effect = "均值回归 ↩️" if w < 0 else "动量 ➡️"
                weights_text += f"Lag {i}: `{w:+.4f}` ({effect})\n"
            embed.add_field(name="Weights / 权重", value=weights_text, inline=False)
            
            embed.set_footer(text=f"Path: {info['path']}")
        
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="reload_model", description="Reload AR model / 重新加载模型")
    async def reload_model(self, interaction: discord.Interaction):
        """重新加载 AR 模型"""
        await interaction.response.defer()
        
        success = self.bot.reload_ar_model()
        
        if success:
            info = self.bot.get_model_info()
            embed = discord.Embed(
                title="🔄 Model Reloaded / 模型已重载",
                color=0x00FF00
            )
            embed.add_field(name="Lags", value=f"**{info['n_lags']}**", inline=True)
            embed.add_field(name="Status", value="✅ Ready", inline=True)
        else:
            embed = discord.Embed(
                title="❌ Reload Failed / 重载失败",
                description="Model file not found or corrupted\n模型文件不存在或损坏",
                color=0xFF0000
            )
        
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="train_model", description="Train new AR model / 训练新模型")
    @app_commands.describe(
        symbol="Training data symbol (e.g., BTC, ETH)",
        timeframe="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)",
        lags="Number of lags (default: 3)",
        limit="Data limit (default: 500)"
    )
    async def train_model(
        self, 
        interaction: discord.Interaction, 
        symbol: str = "BTC",
        timeframe: str = "1h",
        lags: int = 3,
        limit: int = 500
    ):
        """训练新的 AR 模型"""
        await interaction.response.defer()
        
        formatted_symbol = f"{symbol.upper()}/USDT:USDT"
        
        # 发送训练中消息
        training_embed = discord.Embed(
            title="🏋️ Training Model / 训练中...",
            description=f"Symbol: {formatted_symbol}\nTimeframe: {timeframe}\nLags: {lags}\nData: {limit} candles",
            color=0xFFFF00
        )
        await interaction.followup.send(embed=training_embed)
        
        # 在后台训练
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.bot.train_new_model(formatted_symbol, timeframe, lags, limit)
        )
        
        if result["success"]:
            embed = discord.Embed(
                title="✅ Model Trained / 训练完成",
                color=0x00FF00
            )
            embed.add_field(name="Win Rate / 胜率", value=f"**{result['win_rate']:.1%}**", inline=True)
            embed.add_field(name="Sharpe Ratio", value=f"**{result['sharpe']:.2f}**", inline=True)
            embed.add_field(name="Lags", value=f"**{result['n_lags']}**", inline=True)
            
            # 权重
            weights_text = ""
            for i, w in enumerate(result['weights'], 1):
                effect = "Mean Rev" if w < 0 else "Momentum"
                weights_text += f"Lag {i}: `{w:+.4f}` ({effect})\n"
            embed.add_field(name="Weights", value=weights_text, inline=False)
            
            embed.set_footer(text="Model saved and loaded / 模型已保存并加载")
        else:
            embed = discord.Embed(
                title="❌ Training Failed / 训练失败",
                description=f"Error: {result.get('error', 'Unknown error')}",
                color=0xFF0000
            )
        
        await interaction.channel.send(embed=embed)

    @app_commands.command(name="help", description="Show all commands")
    async def help_command(self, interaction: discord.Interaction):
        """帮助命令"""
        embed = discord.Embed(
            title="Trading Bot Commands",
            color=discord.Color.blue()
        )
        
        commands_list = [
            ("/price [symbol]", "Get current price (default: BTC)"),
            ("/funding [symbol]", "Get funding rate"),
            ("/signals [symbol]", "Check current trading signals"),
            ("/indicators [symbol]", "Get technical indicators"),
            ("/chart [symbol] [timeframe]", "Get candlestick chart"),
            ("/analysis [symbol] [timeframe]", "Full analysis (K-line + RSI + MACD)"),
            ("/advice [symbol] [timeframe]", "Rule-based trading advice / 规则型建议"),
            ("/quant [symbol] [timeframe]", "Quantitative analysis / 量化分析"),
            ("/model", "Show AR model info / 显示模型信息"),
            ("/reload_model", "Reload AR model / 重新加载模型"),
            ("/train_model [symbol] [timeframe]", "Train new AR model / 训练新模型"),
            ("/watch [symbol]", "Add to watchlist"),
            ("/unwatch [symbol]", "Remove from watchlist"),
            ("/watchlist", "Show watchlist"),
            ("/setalert", "Enable auto alerts in current channel"),
            ("/stopalert", "Disable auto alerts"),
        ]
        
        for cmd, desc in commands_list:
            embed.add_field(name=cmd, value=desc, inline=False)
        
        await interaction.response.send_message(embed=embed)


def main():
    """启动 Bot"""
    token = os.getenv("DISCORD_BOT_TOKEN")
    
    if not token:
        print("Error: DISCORD_BOT_TOKEN not found in environment variables")
        print("\nTo set up:")
        print("1. Go to https://discord.com/developers/applications")
        print("2. Create a new application")
        print("3. Go to Bot section, create bot and copy token")
        print("4. Add DISCORD_BOT_TOKEN=your_token to .env file")
        print("5. Enable MESSAGE CONTENT INTENT in Bot settings")
        print("6. Invite bot using OAuth2 URL Generator (bot + applications.commands)")
        return
    
    bot = TradingBot()
    bot.run(token)


if __name__ == "__main__":
    main()

