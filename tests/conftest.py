# tests/conftest.py
"""
Pytest 配置
"""
import pytest


def pytest_configure(config):
    """配置自定义标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests that require network"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

