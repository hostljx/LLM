#!/bin/bash

#chmod +x stop_vllm.sh stop_vllm.sh

# 查找 vllm 进程
PID=$(ps aux | grep "vllm serve" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "未找到运行中的 vllm 服务"
    exit 1
fi

echo "找到进程 PID: $PID"
echo "正在停止服务..."

# 停止服务
kill $PID

# 等待进程结束
sleep 2

# 检查是否还在运行，如果是则强制杀死
if ps -p $PID > /dev/null 2>&1; then
    echo "强制停止..."
    kill -9 $PID
fi

echo "服务已停止"