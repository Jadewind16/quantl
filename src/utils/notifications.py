# src/utils/notifications.py
"""
通知模块 - 支持多种通知渠道
"""
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import log


class DiscordNotifier:
    """Discord Webhook 通知器"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        初始化 Discord 通知器
        
        Args:
            webhook_url: Discord Webhook URL
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        
        if not self.webhook_url:
            log.warning("Discord webhook not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            log.info("Discord notifier initialized")
    
    def send_message(self, message: str, **kwargs) -> bool:
        """发送消息到 Discord"""
        if not self.enabled:
            log.info(f"[Discord Disabled] {message[:100]}...")
            return False
        
        payload = {"content": message}
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code in [200, 204]:
                log.debug("Discord message sent successfully")
                return True
            else:
                log.error(f"Discord API error: {response.status_code}")
                return False
                
        except Exception as e:
            log.error(f"Failed to send Discord message: {e}")
            return False
    
    def send_signal(self, signal) -> bool:
        """发送交易信号"""
        message = self._format_signal_message(signal)
        return self.send_message(message)
    
    def _format_signal_message(self, signal) -> str:
        """格式化信号消息"""
        signal_type = signal.signal_type.value
        
        lines = [
            f"**[{signal_type}] {signal.symbol}**",
            "",
            f"**Strategy:** {signal.strategy}",
            f"**Price:** ${signal.price:,.2f}",
            f"**Confidence:** {signal.confidence:.0%}",
            "",
            f"**Reason:** {signal.reason}",
            "",
            "**Indicators:**"
        ]
        
        for name, value in signal.indicators.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.2f}")
            else:
                lines.append(f"  {name}: {value}")
        
        lines.append("")
        lines.append(f"`{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`")
        
        return "\n".join(lines)
    
    def send_alert(self, title: str, message: str) -> bool:
        return self.send_message(f"**[ALERT] {title}**\n\n{message}")
    
    def send_error(self, error: str) -> bool:
        return self.send_message(f"**[ERROR]**\n```\n{error}\n```")
    
    def send_status(self, status: str) -> bool:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return self.send_message(f"**[STATUS]**\n\n{status}\n\n`{timestamp}`")


class EmailNotifier:
    """Email 通知器"""
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        recipient_email: Optional[str] = None
    ):
        """
        初始化 Email 通知器
        
        常用 SMTP 服务器:
        - Gmail: smtp.gmail.com, 587
        - QQ邮箱: smtp.qq.com, 587
        - 163邮箱: smtp.163.com, 25
        """
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD")
        self.recipient_email = recipient_email or os.getenv("RECIPIENT_EMAIL")
        
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            log.warning("Email credentials not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            log.info("Email notifier initialized")
    
    def send_message(self, message: str, subject: str = "Trading Signal") -> bool:
        """发送邮件"""
        if not self.enabled:
            log.info(f"[Email Disabled] {message[:100]}...")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            log.debug("Email sent successfully")
            return True
            
        except Exception as e:
            log.error(f"Failed to send email: {e}")
            return False
    
    def send_signal(self, signal) -> bool:
        """发送交易信号"""
        subject = f"[{signal.signal_type.value}] {signal.symbol}"
        message = signal.to_message()
        return self.send_message(message, subject)
    
    def send_alert(self, title: str, message: str) -> bool:
        return self.send_message(message, f"[ALERT] {title}")
    
    def send_error(self, error: str) -> bool:
        return self.send_message(error, "[ERROR] Trading Bot")
    
    def send_status(self, status: str) -> bool:
        return self.send_message(status, "[STATUS] Trading Bot")


class TelegramNotifier:
    """Telegram 通知器"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        初始化 Telegram 通知器
        
        Args:
            bot_token: Telegram Bot Token (从 @BotFather 获取)
            chat_id: 接收消息的 Chat ID
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            log.warning("Telegram credentials not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            log.info("Telegram notifier initialized")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        发送消息
        
        Args:
            message: 消息内容
            parse_mode: 解析模式 (HTML, Markdown, MarkdownV2)
            
        Returns:
            是否发送成功
        """
        if not self.enabled:
            log.info(f"[Telegram Disabled] {message[:100]}...")
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                log.debug("Telegram message sent successfully")
                return True
            else:
                log.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            log.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_signal(self, signal) -> bool:
        """
        发送交易信号
        
        Args:
            signal: Signal 对象
        """
        message = self._format_signal_message(signal)
        return self.send_message(message)
    
    def _format_signal_message(self, signal) -> str:
        """格式化信号消息为 HTML"""
        type_emoji = {
            "LONG": "LONG",
            "SHORT": "SHORT", 
            "CLOSE": "CLOSE",
            "NEUTRAL": "INFO"
        }
        
        signal_type = signal.signal_type.value
        emoji = type_emoji.get(signal_type, "")
        
        lines = [
            f"<b>[{emoji}] {signal.symbol}</b>",
            "",
            f"<b>Strategy:</b> {signal.strategy}",
            f"<b>Price:</b> ${signal.price:,.2f}",
            f"<b>Confidence:</b> {signal.confidence:.0%}",
            "",
            f"<b>Reason:</b>",
            f"<i>{signal.reason}</i>",
            "",
            "<b>Indicators:</b>"
        ]
        
        for name, value in signal.indicators.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.2f}")
            else:
                lines.append(f"  {name}: {value}")
        
        lines.append("")
        lines.append(f"<code>{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</code>")
        
        return "\n".join(lines)
    
    def send_alert(self, title: str, message: str) -> bool:
        """发送普通警报"""
        formatted = f"<b>[ALERT] {title}</b>\n\n{message}"
        return self.send_message(formatted)
    
    def send_error(self, error: str) -> bool:
        """发送错误通知"""
        formatted = f"<b>[ERROR]</b>\n\n<code>{error}</code>"
        return self.send_message(formatted)
    
    def send_status(self, status: str) -> bool:
        """发送状态更新"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"<b>[STATUS]</b>\n\n{status}\n\n<code>{timestamp}</code>"
        return self.send_message(formatted)


class ConsoleNotifier:
    """控制台通知器 (用于测试)"""
    
    def __init__(self):
        self.enabled = True
        log.info("Console notifier initialized")
    
    def send_message(self, message: str, **kwargs) -> bool:
        print("\n" + "=" * 50)
        print(message)
        print("=" * 50 + "\n")
        return True
    
    def send_signal(self, signal) -> bool:
        message = signal.to_message()
        return self.send_message(message)
    
    def send_alert(self, title: str, message: str) -> bool:
        return self.send_message(f"[ALERT] {title}\n{message}")
    
    def send_error(self, error: str) -> bool:
        return self.send_message(f"[ERROR] {error}")
    
    def send_status(self, status: str) -> bool:
        return self.send_message(f"[STATUS] {status}")


def get_notifier(notifier_type: str = "console"):
    """
    获取通知器实例
    
    Args:
        notifier_type: 通知类型
            - "console": 控制台输出 (测试用)
            - "discord": Discord Webhook
            - "email": 邮件通知
            - "telegram": Telegram Bot
    
    Returns:
        通知器实例
    """
    notifier_type = notifier_type.lower()
    
    if notifier_type == "discord":
        notifier = DiscordNotifier()
        if notifier.enabled:
            return notifier
        log.warning("Discord not configured, falling back to console")
    
    elif notifier_type == "email":
        notifier = EmailNotifier()
        if notifier.enabled:
            return notifier
        log.warning("Email not configured, falling back to console")
    
    elif notifier_type == "telegram":
        notifier = TelegramNotifier()
        if notifier.enabled:
            return notifier
        log.warning("Telegram not configured, falling back to console")
    
    return ConsoleNotifier()

