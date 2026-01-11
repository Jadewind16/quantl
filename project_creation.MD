# 创建所有源代码目录
mkdir -p src/{data,strategy,execution,risk,backtest,utils}

# 创建配置、数据、日志等目录
mkdir -p config data/{raw,processed,backtest} logs tests/{test_data,test_strategy,test_execution} notebooks scripts

# 创建 .gitkeep 文件保持空目录
touch data/raw/.gitkeep data/processed/.gitkeep data/backtest/.gitkeep logs/.gitkeep

# 创建所有 __init__.py 文件
touch src/__init__.py
touch src/data/__init__.py
touch src/strategy/__init__.py
touch src/execution/__init__.py
touch src/risk/__init__.py
touch src/backtest/__init__.py
touch src/utils/__init__.py

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 添加别名到 .bashrc
echo 'alias quantl="cd ~/quantl && source venv/bin/activate"' >> ~/.bashrc

# 重新加载 bashrc 使别名生效
source ~/.bashrc

# 先升级 pip
pip install --upgrade pip

# 安装所有依赖
pip install -r requirements.txt