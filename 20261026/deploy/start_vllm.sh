#!/bin/bash

#chmod +x stop_vllm.sh start_vllm.sh

# 双卡并发优化配置 - 使用张量并行
# --tensor-parallel-size 2: 将模型分割到2张GPU上
# --max-num-seqs: 最大并发请求数
# --max-num-batched-tokens: 批处理最大token数
# --gpu-memory-utilization: GPU显存利用率 (0.0-1.0)
# --enable-prefix-caching: 启用前缀缓存
# --disable-custom-all-reduce: 禁用自定义all-reduce（避免RTX 2080 Ti兼容性问题）

nohup vllm serve /root/autodl-tmp/train_data/1.5b \
    --api-key 12345 \
    --served-model-name qwen \
    --port 6006 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --max-num-seqs 96 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --disable-custom-all-reduce \
    --disable-log-requests > vllm.log 2>&1 &

echo "服务已启动，PID: $!"
echo "查看日志: tail -f vllm.log"
echo "查看进程: ps aux | grep vllm"