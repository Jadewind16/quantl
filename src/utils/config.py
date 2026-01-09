# src/utils/config.py
"""
配置管理系统
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        self.config_dir = Path("config")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # 加载所有配置文件
        self.exchanges = self._load_yaml("exchanges.yaml")
        self.strategies = self._load_yaml("strategies.yaml")
        self.risk = self._load_yaml("risk.yaml")
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """获取交易所配置"""
        config = self.exchanges['exchanges'].get(exchange_name)
        
        if not config:
            raise ValueError(f"未找到交易所配置: {exchange_name}")
        
        return config
    
    def get_api_credentials(self, exchange_name: str) -> Optional[Dict[str, str]]:
        """获取 API 凭证（如果有的话）"""
        api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
        api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")
        
        if api_key and api_secret:
            return {
                "apiKey": api_key,
                "secret": api_secret
            }
        return None
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略配置"""
        config = self.strategies['strategies'].get(strategy_name)
        
        if not config:
            raise ValueError(f"未找到策略配置: {strategy_name}")
        
        return config
    
    @property
    def is_production(self) -> bool:
        """是否生产环境"""
        return self.environment == "production"
    
    @property
    def is_testnet(self) -> bool:
        """是否使用测试网"""
        return not self.is_production


# 全局配置实例
config = Config()

