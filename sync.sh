#!/bin/bash
set -e

# 配置项
LOCAL_DIR="/Users/daiyongxuan/workspace/infiniai/verl/"
REMOTE_HOST="target-via-aic"
REMOTE_DIR="/mnt/public/daiyongxuan/verl/"
DIRECTION=$1  # push 或 pull

# 函数定义
function push_to_remote() {
    echo "📤 同步：本地 → 远程..."
    rsync -avz --delete "$LOCAL_DIR" "${REMOTE_HOST}:${REMOTE_DIR}"
    echo "✅ 同步完成（本地 → 远程）"
}

function pull_from_remote() {
    echo "📥 同步：远程 → 本地..."
    rsync -avz --delete "${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR"
    echo "✅ 同步完成（远程 → 本地）"
}

# 执行逻辑
if [[ "$DIRECTION" == "push" ]]; then
    push_to_remote
elif [[ "$DIRECTION" == "pull" ]]; then
    pull_from_remote
else
    echo "❌ 错误：请输入同步方向：push 或 pull"
    echo "用法: ./sync_verl.sh [push|pull]"
    exit 1
fi