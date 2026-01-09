# src/utils/logger.py
"""
专业的日志系统配置
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    """
    配置日志系统
    - 控制台输出：彩色，简洁格式
    - 文件输出：详细格式，按日期轮转
    """
    # 移除默认处理器
    logger.remove()
    
    # 控制台处理器 - 彩色输出
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # 文件处理器 - 详细日志
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    logger.add(
        log_path / "trading_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # 每天午夜轮转
        retention="30 days",  # 保留30天
        compression="zip",  # 压缩旧日志
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    # 错误日志单独文件
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR"
    )
    
    return logger


# 创建全局 logger 实例
log = setup_logger()

