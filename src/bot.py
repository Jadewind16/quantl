# src/bot.py
"""
信号提醒 Bot 主程序
"""
import time
import schedule
from typing import List, Dict
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.perpetual import PerpetualDataFetcher
from src.strategy.signals import SignalGenerator, Signal, SignalType
from src.utils.notifications import get_notifier
from src.utils.logger import log


class SignalBot:
    """
    信号提醒 Bot
    
    功能:
    - 定时获取市场数据
    - 计算技术指标
    - 检测交易信号
    - 发送 Telegram 通知
    """
    
    def __init__(
        self,
        exchange: str = "okx",
        symbols: List[str] = None,
        timeframe: str = "1h",
        notifier_type: str = "console"
    ):
        """
        初始化 Bot
        
        Args:
            exchange: 交易所名称
            symbols: 监控的交易对列表
            timeframe: K线周期
            notifier_type: 通知方式 (console, discord, email, telegram)
        """
        self.exchange = exchange
        self.symbols = symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        self.timeframe = timeframe
        
        # 初始化组件
        self.fetcher = PerpetualDataFetcher(exchange)
        self.signal_generator = SignalGenerator()
        self.notifier = get_notifier(notifier_type)
        
        # 记录已发送的信号 (避免重复)
        self.sent_signals: Dict[str, datetime] = {}
        self.signal_cooldown = 3600  # 同一信号冷却时间 (秒)
        
        log.info(f"SignalBot initialized")
        log.info(f"  Exchange: {exchange}")
        log.info(f"  Symbols: {symbols}")
        log.info(f"  Timeframe: {timeframe}")
    
    def check_signals(self):
        """检查所有交易对的信号"""
        log.info(f"Checking signals at {datetime.now()}")
        
        for symbol in self.symbols:
            try:
                signals = self._analyze_symbol(symbol)
                
                for signal in signals:
                    if self._should_send_signal(signal):
                        self._send_signal(signal)
                        
            except Exception as e:
                log.error(f"Error analyzing {symbol}: {e}")
    
    def _analyze_symbol(self, symbol: str) -> List[Signal]:
        """分析单个交易对"""
        # 获取 K 线数据
        df = self.fetcher.get_klines(symbol, self.timeframe, limit=100)
        
        # 生成信号
        signals = self.signal_generator.analyze(df, symbol)
        
        # 只返回有效信号 (非 NEUTRAL)
        return [s for s in signals if s.signal_type != SignalType.NEUTRAL]
    
    def _should_send_signal(self, signal: Signal) -> bool:
        """检查是否应该发送信号 (避免重复)"""
        key = f"{signal.symbol}_{signal.strategy}_{signal.signal_type.value}"
        
        if key in self.sent_signals:
            last_sent = self.sent_signals[key]
            elapsed = (datetime.now() - last_sent).total_seconds()
            
            if elapsed < self.signal_cooldown:
                log.debug(f"Signal {key} on cooldown ({elapsed:.0f}s < {self.signal_cooldown}s)")
                return False
        
        return True
    
    def _send_signal(self, signal: Signal):
        """发送信号通知"""
        key = f"{signal.symbol}_{signal.strategy}_{signal.signal_type.value}"
        
        log.info(f"Sending signal: {signal.signal_type.value} {signal.symbol} ({signal.strategy})")
        
        self.notifier.send_signal(signal)
        self.sent_signals[key] = datetime.now()
    
    def get_market_summary(self) -> str:
        """获取市场概览"""
        lines = ["Market Summary", "=" * 40]
        
        for symbol in self.symbols:
            try:
                price_data = self.fetcher.get_price(symbol)
                funding = self.fetcher.get_funding_rate(symbol)
                
                lines.append(f"\n{symbol}")
                lines.append(f"  Price: ${price_data['price']:,.2f}")
                lines.append(f"  24h Change: {price_data['change_24h']:.2f}%")
                lines.append(f"  Funding Rate: {funding['funding_rate_percent']:.4f}%")
                
            except Exception as e:
                lines.append(f"\n{symbol}: Error - {e}")
        
        return "\n".join(lines)
    
    def send_daily_summary(self):
        """发送每日市场摘要"""
        summary = self.get_market_summary()
        self.notifier.send_status(summary)
    
    def run_once(self):
        """运行一次检查"""
        log.info("Running single check...")
        self.check_signals()
        log.info("Check complete")
    
    def run(self, interval_minutes: int = 60):
        """
        启动 Bot (持续运行)
        
        Args:
            interval_minutes: 检查间隔 (分钟)
        """
        log.info(f"Starting SignalBot (interval: {interval_minutes}m)")
        
        # 立即运行一次
        self.check_signals()
        
        # 设置定时任务
        schedule.every(interval_minutes).minutes.do(self.check_signals)
        
        # 每天 8:00 发送市场摘要
        schedule.every().day.at("08:00").do(self.send_daily_summary)
        
        # 启动通知
        self.notifier.send_status(f"SignalBot started\nMonitoring: {', '.join(self.symbols)}")
        
        # 主循环
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Bot stopped by user")
            self.notifier.send_status("SignalBot stopped")


def main():
    """主函数"""
    # 配置
    SYMBOLS = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]
    
    # 创建 Bot
    # notifier_type 选项: "console", "discord", "email", "telegram"
    bot = SignalBot(
        exchange="okx",
        symbols=SYMBOLS,
        timeframe="1h",
        notifier_type="console"  # 改为 "discord" 或 "email" 启用通知
    )
    
    # 显示市场概览
    print(bot.get_market_summary())
    print()
    
    # 运行一次检查
    bot.run_once()
    
    # 如果要持续运行:
    # bot.run(interval_minutes=60)


if __name__ == "__main__":
    main()

