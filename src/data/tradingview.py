# src/data/tradingview.py
"""
TradingView 数据获取模块
通过 TradingView 获取各交易所的 K 线数据
"""
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import log


class TradingViewDataFetcher:
    """TradingView 数据获取器"""
    
    # 时间周期映射
    INTERVAL_MAP = {
        '1m': Interval.in_1_minute,
        '5m': Interval.in_5_minute,
        '15m': Interval.in_15_minute,
        '30m': Interval.in_30_minute,
        '1h': Interval.in_1_hour,
        '2h': Interval.in_2_hour,
        '4h': Interval.in_4_hour,
        '1d': Interval.in_daily,
        '1w': Interval.in_weekly,
        '1M': Interval.in_monthly,
    }
    
    # 无需硬编码列表，_convert_symbol 会智能处理
    
    def __init__(self, username: str = None, password: str = None):
        """
        初始化 TradingView 数据获取器
        
        Args:
            username: TradingView 用户名 (可选，登录后可获取更多数据)
            password: TradingView 密码
        """
        if username and password:
            self.tv = TvDatafeed(username, password)
            log.info("TradingView: Logged in as user")
        else:
            self.tv = TvDatafeed()
            log.info("TradingView: Using anonymous access (limited data)")
    
    def _convert_symbol(self, symbol: str, exchange: str = None) -> tuple:
        """
        转换符号格式
        
        支持格式:
            1. 直接 TradingView 格式: AAPL, BTCUSDT.P, EURUSD
            2. 指定交易所: AAPL:NASDAQ, BTCUSDT.P:BINANCE
            3. 加密货币简写: BTC, ETH (自动转为 BTCUSDT.P@BINANCE)
            
        Returns:
            (tv_symbol, exchange) 元组
        """
        symbol = symbol.upper().strip()
        
        # 1. 用户指定了交易所 (格式: SYMBOL:EXCHANGE)
        if ':' in symbol and not symbol.endswith(':USDT'):
            parts = symbol.split(':')
            if len(parts) == 2:
                return (parts[0], parts[1])
        
        # 2. 参数传入交易所
        if exchange:
            return (symbol, exchange.upper())
        
        # 3. 处理加密货币格式: BTC/USDT:USDT -> BTC
        base = symbol
        if ':USDT' in base:
            base = base.split(':')[0]
        if '/USDT' in base:
            base = base.split('/')[0]
        if base.endswith('USDT') and len(base) > 4:
            base = base[:-4]
        
        # 4. 智能猜测资产类型
        # 常见加密货币 -> BINANCE 永续
        crypto_symbols = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 
                         'DOT', 'MATIC', 'SHIB', 'LTC', 'ATOM', 'UNI', 'ARB', 'OP',
                         'APT', 'SUI', 'NEAR', 'FIL', 'INJ', 'TIA', 'SEI', 'PEPE', 'WIF'}
        if base in crypto_symbols:
            return (f"{base}USDT.P", 'BINANCE')
        
        # 常见美股 -> NASDAQ/NYSE
        us_stocks = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
                    'NFLX', 'COIN', 'MSTR', 'INTC', 'PYPL', 'CRM', 'ADBE', 'ORCL'}
        if base in us_stocks:
            return (base, 'NASDAQ')
        
        # 常见 ETF -> AMEX
        etfs = {'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'ARKK', 'VTI', 'VOO'}
        if base in etfs:
            return (base, 'AMEX')
        
        # 外汇对 -> FX
        if len(base) == 6 and base[:3] in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF']:
            return (base, 'FX')
        
        # 黄金白银
        if base in ['XAUUSD', 'XAGUSD', 'GOLD', 'SILVER']:
            if base == 'GOLD':
                base = 'XAUUSD'
            if base == 'SILVER':
                base = 'XAGUSD'
            return (base, 'OANDA')
        
        # 5. 默认: 假设是加密货币永续
        return (f"{base}USDT.P", 'BINANCE')
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                    limit: int = 100) -> Optional[pd.DataFrame]:
        """
        获取 K 线数据
        
        Args:
            symbol: 交易对 (ccxt 格式，如 BTC/USDT:USDT)
            timeframe: 时间周期 (1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w, 1M)
            limit: K 线数量
            
        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        try:
            tv_symbol, exchange = self._convert_symbol(symbol)
            interval = self.INTERVAL_MAP.get(timeframe, Interval.in_1_hour)
            
            log.debug(f"Fetching {symbol} ({tv_symbol}@{exchange}) {timeframe} x{limit}")
            
            df = self.tv.get_hist(
                symbol=tv_symbol,
                exchange=exchange,
                interval=interval,
                n_bars=limit
            )
            
            if df is None or df.empty:
                log.warning(f"No data returned for {symbol}")
                return None
            
            # 重命名列以匹配 ccxt 格式
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # 确保索引是 datetime
            df.index = pd.to_datetime(df.index)
            
            log.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            log.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取最新价格信息
        
        Args:
            symbol: 交易对
            
        Returns:
            包含价格信息的字典
        """
        try:
            # 获取最近2根K线来计算变化
            df = self.fetch_ohlcv(symbol, '1h', limit=2)
            
            if df is None or len(df) < 2:
                return None
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            change = current['close'] - previous['close']
            change_pct = (change / previous['close']) * 100
            
            return {
                'symbol': symbol,
                'last': current['close'],
                'high': current['high'],
                'low': current['low'],
                'open': current['open'],
                'close': current['close'],
                'change': change,
                'percentage': change_pct,
                'volume': current['volume'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            log.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def fetch_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        TradingView 不提供资金费率数据
        返回 None
        """
        log.warning("TradingView does not provide funding rate data")
        return None
    
    def get_available_symbols(self) -> List[str]:
        """获取可用的交易对列表"""
        return list(self.SYMBOL_MAP.keys())
    
    # ========== 兼容 PerpetualDataFetcher 接口 ==========
    
    def get_klines(self, symbol: str, timeframe: str = '1h', 
                   limit: int = 100) -> Optional[pd.DataFrame]:
        """
        获取 K 线数据 (兼容接口)
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: K 线数量
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        return self.fetch_ohlcv(symbol, timeframe, limit)
    
    def get_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取最新价格 (兼容接口)
        
        Returns:
            {
                'price': float,
                'change_24h': float,
                'volume_24h': float,
                'high_24h': float,
                'low_24h': float
            }
        """
        ticker = self.fetch_ticker(symbol)
        if ticker is None:
            return None
        
        return {
            'price': ticker['last'],
            'change_24h': ticker['percentage'],
            'volume_24h': ticker['volume'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low']
        }
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取资金费率 (TradingView 不支持)
        
        Returns:
            None (TradingView 不提供资金费率数据)
        """
        return None


# 测试
if __name__ == "__main__":
    fetcher = TradingViewDataFetcher()
    
    # 测试获取 K 线
    print("\n=== OHLCV Data ===")
    df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", 10)
    if df is not None:
        print(df.tail())
    
    # 测试获取价格
    print("\n=== Ticker ===")
    ticker = fetcher.fetch_ticker("BTC/USDT:USDT")
    if ticker:
        print(f"Price: ${ticker['last']:,.2f}")
        print(f"Change: {ticker['percentage']:.2f}%")
    
    # 测试 ETH
    print("\n=== ETH Data ===")
    df = fetcher.fetch_ohlcv("ETH/USDT:USDT", "4h", 5)
    if df is not None:
        print(df)

