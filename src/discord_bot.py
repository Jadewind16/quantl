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
from src.utils.logger import log
from src.utils.charts import create_candlestick_chart, create_indicator_chart

# 加载环境变量
load_dotenv()


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
        
        # 监控配置
        self.watched_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        self.alert_channel_id: Optional[int] = None
        self.sent_signals = {}
        self.signal_cooldown = 3600
    
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
        embed.add_field(name="Price", value=f"${signal.price:,.2f}", inline=True)
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
                value=f"${price_data['price']:,.2f}", 
                inline=True
            )
            embed.add_field(
                name="24h Change", 
                value=f"{change_sign}{price_data['change_24h']:.2f}%", 
                inline=True
            )
            embed.add_field(
                name="24h High", 
                value=f"${price_data['high_24h']:,.2f}", 
                inline=True
            )
            embed.add_field(
                name="24h Low", 
                value=f"${price_data['low_24h']:,.2f}", 
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
                value=f"SMA20: ${latest['sma_20']:,.2f}\nSMA50: ${latest['sma_50']:,.2f}\nEMA12: ${latest['ema_12']:,.2f}",
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
                value=f"Upper: ${latest['bb_upper']:,.2f}\nMiddle: ${latest['bb_middle']:,.2f}\nLower: ${latest['bb_lower']:,.2f}",
                inline=True
            )
            
            # ATR
            embed.add_field(
                name="Volatility",
                value=f"ATR: ${latest['atr']:.2f}",
                inline=True
            )
            
            # 当前价格
            embed.add_field(
                name="Current Price",
                value=f"${latest['close']:,.2f}",
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
                    f"Current/当前:  ${data['current_price']:>12,.2f}\n"
                    f"Entry/入场:    ${data['entry_price']:>12,.2f}\n"
                    f"Stop/止损:     ${data['stop_loss']:>12,.2f}\n"
                    f"TP1/止盈1:     ${data['take_profit_1']:>12,.2f}\n"
                    f"TP2/止盈2:     ${data['take_profit_2']:>12,.2f}\n"
                    f"TP3/止盈3:     ${data['take_profit_3']:>12,.2f}\n"
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
                    value=f"**{data['risk_reward']:.2f}:1**",
                    inline=True
                )
            else:
                embed.add_field(
                    name="Current Price / 当前价格",
                    value=f"**${data['current_price']:,.2f}**",
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
            ("/advice [symbol] [timeframe]", "Get trading advice with Entry/SL/TP"),
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

