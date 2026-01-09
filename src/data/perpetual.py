 # src/data/perpetual.py
"""
永续合约数据获取模块
专注于获取 Perpetual Futures 相关数据
"""
import ccxt
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import log


class PerpetualDataFetcher:
    """
    永续合约数据获取器
    
    支持功能：
    - 获取实时价格
    - 获取 K 线数据
    - 获取资金费率
    - 获取持仓量
    - 获取买卖盘深度
    """
    
    def __init__(self, exchange_name: str = "binance", testnet: bool = False):
        """
        初始化数据获取器
        
        Args:
            exchange_name: 交易所名称 (binance, bybit, okx)
            testnet: 是否使用测试网
        """
        self.exchange_name = exchange_name
        self.testnet = testnet
        
        # 创建交易所实例
        self.exchange = self._create_exchange()
        
        log.info(f"永续合约数据获取器初始化完成: {exchange_name}")
    
    def _create_exchange(self) -> ccxt.Exchange:
        """创建 CCXT 交易所实例"""
        exchange_class = getattr(ccxt, self.exchange_name)
        
        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 默认使用期货市场
            }
        }
        
        # 测试网配置
        if self.testnet:
            if self.exchange_name == 'binance':
                config['options']['defaultType'] = 'future'
                config['options']['testnet'] = True
            elif self.exchange_name == 'bybit':
                config['options']['testnet'] = True
        
        return exchange_class(config)
    
    # ==================== 价格数据 ====================
    
    def get_price(self, symbol: str = "BTC/USDT:USDT") -> Dict[str, Any]:
        """
        获取当前价格信息
        
        Args:
            symbol: 交易对，永续合约格式如 "BTC/USDT:USDT"
            
        Returns:
            {
                'symbol': 'BTC/USDT:USDT',
                'price': 50000.0,
                'change_24h': 2.5,
                'high_24h': 51000.0,
                'low_24h': 49000.0,
                'volume_24h': 1000000.0,
                'timestamp': '2024-01-01 12:00:00'
            }
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'change_24h': ticker.get('percentage', 0),
                'high_24h': ticker.get('high', 0),
                'low_24h': ticker.get('low', 0),
                'volume_24h': ticker.get('quoteVolume', 0),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            log.error(f"获取价格失败 {symbol}: {e}")
            raise
    
    def get_prices(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        批量获取多个交易对价格
        
        Args:
            symbols: 交易对列表，如 ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        """
        if symbols is None:
            symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        
        results = []
        for symbol in symbols:
            try:
                price_data = self.get_price(symbol)
                results.append(price_data)
            except Exception as e:
                log.warning(f"获取 {symbol} 价格失败: {e}")
        
        return results
    
    # ==================== K线数据 ====================
    
    def get_klines(
        self, 
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        获取 K 线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期 (1m, 5m, 15m, 1h, 4h, 1d)
            limit: 获取数量
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            log.debug(f"获取 {symbol} {timeframe} K线 {len(df)} 条")
            return df
            
        except Exception as e:
            log.error(f"获取K线失败 {symbol}: {e}")
            raise
    
    # ==================== 资金费率 ====================
    
    def get_funding_rate(self, symbol: str = "BTC/USDT:USDT") -> Dict[str, Any]:
        """
        获取资金费率
        
        资金费率是永续合约的重要指标：
        - 正值：多头付给空头（市场看多）
        - 负值：空头付给多头（市场看空）
        
        Returns:
            {
                'symbol': 'BTC/USDT:USDT',
                'funding_rate': 0.0001,  # 0.01%
                'funding_rate_percent': 0.01,
                'next_funding_time': '2024-01-01 08:00:00',
                'sentiment': 'bullish'  # bullish/bearish/neutral
            }
        """
        try:
            # 获取资金费率
            funding = self.exchange.fetch_funding_rate(symbol)
            
            rate = funding.get('fundingRate', 0)
            rate_percent = rate * 100 if rate else 0
            
            # 判断市场情绪
            if rate_percent > 0.01:
                sentiment = 'bullish'  # 看多
            elif rate_percent < -0.01:
                sentiment = 'bearish'  # 看空
            else:
                sentiment = 'neutral'  # 中性
            
            # 下次结算时间
            next_time = funding.get('fundingTimestamp')
            if next_time:
                next_time = datetime.fromtimestamp(next_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'symbol': symbol,
                'funding_rate': rate,
                'funding_rate_percent': rate_percent,
                'next_funding_time': next_time,
                'sentiment': sentiment
            }
            
        except Exception as e:
            log.error(f"获取资金费率失败 {symbol}: {e}")
            raise
    
    # ==================== 持仓量 ====================
    
    def get_open_interest(self, symbol: str = "BTC/USDT:USDT") -> Dict[str, Any]:
        """
        获取持仓量 (Open Interest)
        
        持仓量表示市场上未平仓的合约数量
        - 增加：新资金进入市场
        - 减少：资金离开市场
        
        Returns:
            {
                'symbol': 'BTC/USDT:USDT',
                'open_interest': 100000.0,  # BTC
                'open_interest_value': 5000000000.0,  # USDT
            }
        """
        try:
            # 注意：不是所有交易所都支持此功能
            if hasattr(self.exchange, 'fetch_open_interest'):
                oi = self.exchange.fetch_open_interest(symbol)
                
                return {
                    'symbol': symbol,
                    'open_interest': oi.get('openInterest', 0),
                    'open_interest_value': oi.get('openInterestValue', 0),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                log.warning(f"{self.exchange_name} 不支持获取持仓量")
                return None
                
        except Exception as e:
            log.error(f"获取持仓量失败 {symbol}: {e}")
            return None
    
    # ==================== 买卖盘 ====================
    
    def get_orderbook(
        self, 
        symbol: str = "BTC/USDT:USDT",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        获取买卖盘深度
        
        Returns:
            {
                'symbol': 'BTC/USDT:USDT',
                'bids': [[price, amount], ...],  # 买单
                'asks': [[price, amount], ...],  # 卖单
                'spread': 0.5,  # 买卖价差
                'spread_percent': 0.001  # 价差百分比
            }
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            spread = best_ask - best_bid
            spread_percent = (spread / best_bid * 100) if best_bid else 0
            
            return {
                'symbol': symbol,
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_percent': spread_percent,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            log.error(f"获取买卖盘失败 {symbol}: {e}")
            raise
    
    # ==================== 市场列表 ====================
    
    def get_perpetual_markets(self) -> List[str]:
        """
        获取所有永续合约交易对
        
        Returns:
            ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...]
        """
        try:
            markets = self.exchange.load_markets()
            
            perpetual_symbols = []
            for symbol, market in markets.items():
                # 筛选永续合约
                if market.get('swap') and market.get('active'):
                    perpetual_symbols.append(symbol)
            
            log.info(f"找到 {len(perpetual_symbols)} 个永续合约")
            return sorted(perpetual_symbols)
            
        except Exception as e:
            log.error(f"获取市场列表失败: {e}")
            raise


# ==================== 快捷函数 ====================

def get_btc_price(exchange: str = "bybit") -> float:
    """快速获取 BTC 价格"""
    fetcher = PerpetualDataFetcher(exchange)
    data = fetcher.get_price("BTC/USDT:USDT")
    return data['price']


def get_eth_price(exchange: str = "bybit") -> float:
    """快速获取 ETH 价格"""
    fetcher = PerpetualDataFetcher(exchange)
    data = fetcher.get_price("ETH/USDT:USDT")
    return data['price']


# ==================== 测试 ====================

if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("永续合约数据获取测试")
    print("=" * 50)
    
    # 创建获取器 (尝试 okx)
    fetcher = PerpetualDataFetcher("gate")
    
    # 测试交易对格式 (OKX 使用 BTC/USDT:USDT)
    symbol = "BTC/USDT:USDT"
    
    # 1. 测试获取价格
    print("\n[BTC Price]")
    try:
        btc = fetcher.get_price(symbol)
        print(f"   Price: ${btc['price']:,.2f}")
        print(f"   24h Change: {btc['change_24h']:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. 测试获取资金费率
    print("\n[Funding Rate]")
    try:
        funding = fetcher.get_funding_rate(symbol)
        print(f"   Rate: {funding['funding_rate_percent']:.4f}%")
        print(f"   Sentiment: {funding['sentiment']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. 测试获取 K 线
    print("\n[OHLCV Data - Last 5 Candles]")
    try:
        klines = fetcher.get_klines(symbol, "1h", limit=5)
        print(klines)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n[Test Complete]")

