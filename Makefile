# Makefile for quantl trading bot
# Usage: make [target]

.PHONY: help install test all_tests test_fast test_slow lint format clean run

# Python 环境
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
FLAKE8 := $(VENV)/bin/flake8

# 默认目标
help:
	@echo "Available targets:"
	@echo "  make install      - Install dependencies"
	@echo "  make all_tests    - Run all tests (including slow network tests)"
	@echo "  make test         - Run fast tests only (no network)"
	@echo "  make test_fast    - Same as 'make test'"
	@echo "  make test_slow    - Run only slow network tests"
	@echo "  make lint         - Run linter (flake8)"
	@echo "  make format       - Format code (black)"
	@echo "  make clean        - Remove cache files"
	@echo "  make run          - Run Discord bot"

# 安装依赖
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8

# 运行所有测试 (包括需要网络的慢速测试)
all_tests:
	@echo "Running ALL tests (including network tests)..."
	$(PYTEST) tests/ -v --run-slow --tb=short

# 运行快速测试 (不需要网络)
test:
	@echo "Running fast tests..."
	$(PYTEST) tests/ -v --tb=short

test_fast: test

# 只运行慢速测试
test_slow:
	@echo "Running slow network tests..."
	$(PYTEST) tests/ -v --run-slow -m slow --tb=short

# 运行特定测试文件
test_tradingview:
	$(PYTEST) tests/test_tradingview.py -v --run-slow --tb=short

test_indicators:
	$(PYTEST) tests/test_indicators.py -v --tb=short

test_signals:
	$(PYTEST) tests/test_signals.py -v --tb=short

test_quant:
	$(PYTEST) tests/test_quant_advisor.py -v --tb=short

# 测试覆盖率
coverage:
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

# 代码检查
lint:
	@echo "Running linter..."
	$(FLAKE8) src/ --max-line-length=120 --ignore=E501,W503

# 代码格式化
format:
	@echo "Formatting code..."
	$(BLACK) src/ tests/ --line-length=120

# 清理缓存
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage

# 运行 Discord bot
run:
	@echo "Starting Discord bot..."
	$(VENV)/bin/python src/discord_bot.py

# 运行 bot (开发模式)
dev:
	@echo "Starting Discord bot in development mode..."
	$(VENV)/bin/python src/discord_bot.py

# 验证环境
check:
	@echo "Checking environment..."
	@$(PYTHON) --version
	@$(PIP) --version
	@echo "Dependencies installed:"
	@$(PIP) list | grep -E "(discord|pandas|numpy|ccxt|tvDatafeed)"

