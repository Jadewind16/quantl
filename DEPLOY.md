# Deployment Guide / 部署指南

## Option 1: Railway (Recommended / 推荐)

### Step 1: Create GitHub Repository / 创建 GitHub 仓库

```bash
cd ~/quantl
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/quantl.git
git push -u origin main
```

### Step 2: Deploy on Railway / 在 Railway 部署

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `quantl` repository
5. Add environment variable:
   - `DISCORD_BOT_TOKEN` = your token

### Step 3: Done! / 完成！

Railway will auto-deploy. Bot will run 24/7.
Railway 会自动部署，Bot 将 24/7 运行。

Free tier: $5/month credit (usually enough for a bot)
免费额度：每月 $5（通常足够运行一个 bot）

---

## Option 2: Render

### Step 1: Same as above / 同上

Create GitHub repo and push code.

### Step 2: Deploy on Render / 在 Render 部署

1. Go to [render.com](https://render.com)
2. Create "Background Worker" (not Web Service)
3. Connect GitHub repo
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `python src/discord_bot.py`
6. Add env var: `DISCORD_BOT_TOKEN`

---

## Option 3: PythonAnywhere (Paid / 付费)

**Note: Requires $5/month "Hacker" plan for Always-on tasks**
**注意：需要 $5/月 "Hacker" 计划才能运行常驻任务**

### Step 1: Upload Code / 上传代码

```bash
# On PythonAnywhere console:
git clone https://github.com/YOUR_USERNAME/quantl.git
cd quantl
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Create Always-on Task / 创建常驻任务

1. Go to "Tasks" tab
2. Create "Always-on task" (Hacker plan required)
3. Command: `/home/YOUR_USERNAME/quantl/venv/bin/python /home/YOUR_USERNAME/quantl/src/discord_bot.py`

### Step 3: Set Environment Variable / 设置环境变量

Create `.env` file on PythonAnywhere:
```bash
echo "DISCORD_BOT_TOKEN=your_token_here" > .env
```

---

## Option 4: Oracle Cloud Free Tier (Advanced / 进阶)

Free forever VM with 1GB RAM - perfect for bots.
永久免费 VM，1GB 内存 - 非常适合 bot。

### Step 1: Create Free Account

1. Go to [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Create account (needs credit card for verification, won't charge)

### Step 2: Create VM

1. Create "Compute Instance"
2. Choose "Always Free" shape (ARM Ampere A1)
3. Choose Ubuntu 22.04

### Step 3: Setup

```bash
# SSH into VM
ssh -i your_key ubuntu@your_ip

# Install Python
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone and setup
git clone https://github.com/YOUR_USERNAME/quantl.git
cd quantl
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env
echo "DISCORD_BOT_TOKEN=your_token" > .env

# Run with systemd (auto-restart)
sudo nano /etc/systemd/system/discord-bot.service
```

Add this content:
```ini
[Unit]
Description=Discord Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/quantl
ExecStart=/home/ubuntu/quantl/venv/bin/python /home/ubuntu/quantl/src/discord_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable discord-bot
sudo systemctl start discord-bot
sudo systemctl status discord-bot
```

---

## Quick Comparison / 快速对比

| Platform | Cost | Setup | Reliability |
|----------|------|-------|-------------|
| Railway | Free $5/mo | Easy | High |
| Render | Free | Easy | Medium (cold start) |
| PythonAnywhere | $5/mo | Easy | High |
| Oracle Cloud | Free forever | Hard | Very High |
| Fly.io | Free tier | Medium | High |

**Recommendation / 推荐**: Start with Railway, upgrade to Oracle Cloud if you need more.
从 Railway 开始，如果需要更多资源可以升级到 Oracle Cloud。

