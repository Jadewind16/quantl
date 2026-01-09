# src/execution/exchange_wrapper.py
"""
交易所 API 封装示例
展示如何将 REST API 封装成 Python 类
"""
import time
import hmac
import hashlib
import requests
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


class BaseExchange(ABC):
    """
    交易所基类 - 定义所有交易所必须实现的方法
    ABC = Abstract Base Class (抽象基类)
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = ""  # 子类设置
        
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """获取当前价格"""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict:
        """获取账户余额"""
        pass
    
    @abstractmethod
    def create_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """下单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        pass


class YubitExchange(BaseExchange):
    """
    YUBIT 交易所封装示例
    注意：这是示例代码，实际 API 端点需要参考 YUBIT 官方文档
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        
        # 设置 API 地址
        if testnet:
            self.base_url = "https://testnet-api.yubit.com"  # 假设的测试网地址
        else:
            self.base_url = "https://api.yubit.com"  # 假设的正式地址
            
        self.session = requests.Session()
    
    # ========== 私有方法：签名和请求 ==========
    
    def _generate_signature(self, timestamp: str, method: str, endpoint: str, body: str = "") -> str:
        """
        生成 API 签名
        大多数交易所使用 HMAC-SHA256 签名
        """
        # 签名字符串 = 时间戳 + 方法 + 端点 + 请求体
        sign_string = timestamp + method + endpoint + body
        
        # 使用 HMAC-SHA256 加密
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        发送 HTTP 请求
        统一处理：签名、错误、重试
        """
        url = self.base_url + endpoint
        timestamp = str(int(time.time() * 1000))
        
        # 生成签名
        body = ""
        if data:
            import json
            body = json.dumps(data)
            
        signature = self._generate_signature(timestamp, method, endpoint, body)
        
        # 设置请求头
        headers = {
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": timestamp,
            "X-SIGNATURE": signature,
            "Content-Type": "application/json"
        }
        
        # 发送请求
        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = self.session.post(url, json=data, headers=headers, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, params=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")
            
            # 检查响应
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 请求失败: {e}")
    
    # ========== 公开方法：交易功能 ==========
    
    def get_price(self, symbol: str) -> float:
        """
        获取当前价格
        
        参数:
            symbol: 交易对，如 "BTC/USDT"
            
        返回:
            当前价格 (float)
        """
        # 转换格式: "BTC/USDT" -> "BTCUSDT"
        formatted_symbol = symbol.replace("/", "")
        
        # 调用 API
        endpoint = f"/v1/ticker/{formatted_symbol}"
        result = self._request("GET", endpoint)
        
        # 解析返回数据
        return float(result.get("last_price", 0))
    
    def get_balance(self) -> Dict[str, float]:
        """
        获取账户余额
        
        返回:
            {"BTC": 0.5, "USDT": 10000.0, ...}
        """
        endpoint = "/v1/account/balance"
        result = self._request("GET", endpoint)
        
        # 解析成简单格式
        balances = {}
        for asset in result.get("assets", []):
            balances[asset["currency"]] = float(asset["available"])
            
        return balances
    
    def create_order(
        self, 
        symbol: str, 
        side: str,  # "buy" 或 "sell"
        amount: float, 
        price: Optional[float] = None,  # None = 市价单
        order_type: str = "limit"
    ) -> Dict:
        """
        下单
        
        参数:
            symbol: 交易对 "BTC/USDT"
            side: "buy" 或 "sell"
            amount: 数量
            price: 价格 (市价单不需要)
            order_type: "limit" 或 "market"
            
        返回:
            订单信息 {"order_id": "xxx", "status": "open", ...}
        """
        endpoint = "/v1/order/create"
        
        data = {
            "symbol": symbol.replace("/", ""),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(amount),
        }
        
        if order_type == "limit" and price:
            data["price"] = str(price)
        
        result = self._request("POST", endpoint, data=data)
        
        return {
            "order_id": result.get("order_id"),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "status": result.get("status", "unknown")
        }
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        取消订单
        
        返回:
            True = 成功, False = 失败
        """
        endpoint = f"/v1/order/cancel"
        
        data = {
            "order_id": order_id,
            "symbol": symbol.replace("/", "")
        }
        
        result = self._request("DELETE", endpoint, params=data)
        
        return result.get("success", False)
    
    def get_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[Dict]:
        """
        获取K线数据
        
        返回:
            [{"time": 1234567890, "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1000}, ...]
        """
        endpoint = "/v1/klines"
        
        params = {
            "symbol": symbol.replace("/", ""),
            "interval": timeframe,
            "limit": limit
        }
        
        result = self._request("GET", endpoint, params=params)
        
        # 转换成标准格式
        klines = []
        for k in result.get("data", []):
            klines.append({
                "time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
            
        return klines


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例用法（需要真实的 API 密钥）
    
    # 1. 创建交易所实例
    exchange = YubitExchange(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True  # 使用测试网
    )
    
    # 2. 获取价格
    # price = exchange.get_price("BTC/USDT")
    # print(f"BTC 价格: {price}")
    
    # 3. 获取余额
    # balance = exchange.get_balance()
    # print(f"账户余额: {balance}")
    
    # 4. 下单
    # order = exchange.create_order(
    #     symbol="BTC/USDT",
    #     side="buy",
    #     amount=0.001,
    #     price=50000,
    #     order_type="limit"
    # )
    # print(f"订单: {order}")
    
    print("交易所封装示例 - 请参考代码学习")

