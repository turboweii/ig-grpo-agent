#!/bin/bash
# 快速上传脚本

# 配置服务器信息
SERVER_USER="your_username"
SERVER_HOST="your_server_ip"
SERVER_PATH="/path/to/ig-grpo-agent"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}IG-GRPO 上传脚本${NC}"
echo "======================================"

# 检查是否配置了服务器信息
if [ "$SERVER_USER" = "your_username" ]; then
    echo -e "${YELLOW}请先修改脚本中的服务器信息:${NC}"
    echo "SERVER_USER=\"$SERVER_USER\""
    echo "SERVER_HOST=\"$SERVER_HOST\""
    echo "SERVER_PATH=\"$SERVER_PATH\""
    exit 1
fi

# 显示上传内容
echo -e "${GREEN}准备上传 ig-grpo-agent 目录到:${NC}"
echo "  $SERVER_USER@$SERVER_HOST:$SERVER_PATH"
echo ""
echo "排除内容:"
echo "  - outputs/"
echo "  - __pycache__/"
echo "  - *.pyc"
echo "  - .git/"
echo ""

read -p "确认上传? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消上传"
    exit 1
fi

# 执行上传
echo -e "${GREEN}开始上传...${NC}"

rsync -avz --progress \
    --exclude 'outputs' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude '*.egg-info' \
    ig-grpo-agent/ \
    $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

echo -e "${GREEN}上传完成!${NC}"
echo ""
echo "下一步："
echo "1. SSH 登录服务器: ssh $SERVER_USER@$SERVER_HOST"
echo "2. 进入目录: cd $SERVER_PATH"
echo "3. 查看部署指南: cat SERVER_DEPLOYMENT.md"
