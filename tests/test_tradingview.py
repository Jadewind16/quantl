# tests/test_tradingview.py
"""
TradingView 数据源测试
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tradingview import TradingViewDataFetcher


class TestSymbolConversion:
    """符号转换测试"""
    
    @pytest.fixture
    def fetcher(self):
        return TradingViewDataFetcher()
    
    # 加密货币测试
    @pytest.mark.parametrize("input_symbol,expected_symbol,expected_exchange", [
        ("BTC", "BTCUSDT.P", "BINANCE"),
        ("ETH", "ETHUSDT.P", "BINANCE"),
        ("SOL", "SOLUSDT.P", "BINANCE"),
        ("btc", "BTCUSDT.P", "BINANCE"),  # 小写
        ("BTC/USDT:USDT", "BTCUSDT.P", "BINANCE"),  # 旧格式
    ])
    def test_crypto_symbols(self, fetcher, input_symbol, expected_symbol, expected_exchange):
        tv_sym, exchange = fetcher._convert_symbol(input_symbol)
        assert tv_sym == expected_symbol
        assert exchange == expected_exchange
    
    # 美股测试
    @pytest.mark.parametrize("input_symbol,expected_symbol,expected_exchange", [
        ("AAPL", "AAPL", "NASDAQ"),
        ("TSLA", "TSLA", "NASDAQ"),
        ("NVDA", "NVDA", "NASDAQ"),
        ("MSFT", "MSFT", "NASDAQ"),
    ])
    def test_stock_symbols(self, fetcher, input_symbol, expected_symbol, expected_exchange):
        tv_sym, exchange = fetcher._convert_symbol(input_symbol)
        assert tv_sym == expected_symbol
        assert exchange == expected_exchange
    
    # ETF 测试
    @pytest.mark.parametrize("input_symbol,expected_symbol,expected_exchange", [
        ("SPY", "SPY", "AMEX"),
        ("QQQ", "QQQ", "AMEX"),
        ("GLD", "GLD", "AMEX"),
    ])
    def test_etf_symbols(self, fetcher, input_symbol, expected_symbol, expected_exchange):
        tv_sym, exchange = fetcher._convert_symbol(input_symbol)
        assert tv_sym == expected_symbol
        assert exchange == expected_exchange
    
    # 外汇测试
    @pytest.mark.parametrize("input_symbol,expected_symbol,expected_exchange", [
        ("EURUSD", "EURUSD", "FX"),
        ("GBPUSD", "GBPUSD", "FX"),
        ("USDJPY", "USDJPY", "FX"),
    ])
    def test_forex_symbols(self, fetcher, input_symbol, expected_symbol, expected_exchange):
        tv_sym, exchange = fetcher._convert_symbol(input_symbol)
        assert tv_sym == expected_symbol
        assert exchange == expected_exchange
    
    # 黄金白银测试
    @pytest.mark.parametrize("input_symbol,expected_symbol,expected_exchange", [
        ("XAUUSD", "XAUUSD", "OANDA"),
        ("GOLD", "XAUUSD", "OANDA"),
        ("XAGUSD", "XAGUSD", "OANDA"),
    ])
    def test_precious_metals(self, fetcher, input_symbol, expected_symbol, expected_exchange):
        tv_sym, exchange = fetcher._convert_symbol(input_symbol)
        assert tv_sym == expected_symbol
        assert exchange == expected_exchange
    
    # 指定交易所格式测试
    @pytest.mark.parametrize("input_symbol,expected_symbol,expected_exchange", [
        ("AAPL:NASDAQ", "AAPL", "NASDAQ"),
        ("BTCUSDT.P:BINANCE", "BTCUSDT.P", "BINANCE"),
        ("SPX:SP", "SPX", "SP"),
    ])
    def test_explicit_exchange(self, fetcher, input_symbol, expected_symbol, expected_exchange):
        tv_sym, exchange = fetcher._convert_symbol(input_symbol)
        assert tv_sym == expected_symbol
        assert exchange == expected_exchange
    
    # 未知符号默认处理
    def test_unknown_symbol_defaults_to_crypto(self, fetcher):
        tv_sym, exchange = fetcher._convert_symbol("RANDOMXYZ")
        assert tv_sym == "RANDOMXYZUSDT.P"
        assert exchange == "BINANCE"


class TestDataFetching:
    """数据获取测试 (需要网络)"""
    
    @pytest.fixture
    def fetcher(self):
        return TradingViewDataFetcher()
    
    @pytest.mark.slow
    def test_fetch_crypto_klines(self, fetcher):
        df = fetcher.get_klines("BTC", "1h", 10)
        assert df is not None
        assert len(df) == 10
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
    
    @pytest.mark.slow
    def test_fetch_stock_klines(self, fetcher):
        df = fetcher.get_klines("AAPL", "1h", 10)
        assert df is not None
        assert len(df) == 10
    
    @pytest.mark.slow
    def test_fetch_forex_klines(self, fetcher):
        df = fetcher.get_klines("EURUSD", "1h", 10)
        assert df is not None
        assert len(df) == 10
    
    @pytest.mark.slow
    def test_get_price(self, fetcher):
        price = fetcher.get_price("BTC")
        assert price is not None
        assert "price" in price
        assert price["price"] > 0

