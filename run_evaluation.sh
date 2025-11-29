#!/bin/bash

# 批量运行最短路径评估脚本
# 使用方法: ./run_evaluation.sh

echo "========================================="
echo "开始批量评估 GPT-4o 在最短路径任务上的表现"
echo "模型: gpt-4o | 难度: hard | SC: 0"
echo "测试不同的 Prompt 策略"
echo "========================================="

# 检查 OPENAI_API_KEY 是否设置
if [ -z "$OPENAI_API_KEY" ]; then
    echo "错误: 未设置 OPENAI_API_KEY 环境变量"
    echo "请先运行: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# 激活 conda 环境
echo ""
echo "激活 NLGraph conda 环境..."
eval "$(conda shell.bash hook)"
conda activate NLGraph

# 设置工作目录
cd /Users/junle/Code/LLM4route/NLGraph

# # 任务1: hard mode, CoT, 无自洽性
# echo ""
# echo "========================================="
# echo "任务 1/5: hard mode + CoT + SC=0"
# echo "========================================="
# python evaluation/shortest_path.py --model gpt-4o --mode hard --prompt CoT --SC 0
# echo "✓ 任务 1 完成"

# 任务2: hard mode, none, 无自洽性
echo ""
echo "========================================="
echo "任务 2/5: hard mode + none + SC=0"
echo "========================================="
python evaluation/shortest_path.py --model gpt-4o --mode hard --prompt none --SC 0
echo "✓ 任务 2 完成"

# 任务3: hard mode, 0-CoT, 无自洽性
echo ""
echo "========================================="
echo "任务 3/5: hard mode + 0-CoT + SC=0"
echo "========================================="
python evaluation/shortest_path.py --model gpt-4o --mode hard --prompt 0-CoT --SC 0
echo "✓ 任务 3 完成"

# 任务4: hard mode, Instruct, 无自洽性
echo ""
echo "========================================="
echo "任务 4/5: hard mode + Instruct + SC=0"
echo "========================================="
python evaluation/shortest_path.py --model gpt-4o --mode hard --prompt Instruct --SC 0
echo "✓ 任务 4 完成"

# 任务5: hard mode, PROGRAM, 无自洽性
echo ""
echo "========================================="
echo "任务 5/5: hard mode + PROGRAM + SC=0"
echo "========================================="
python evaluation/shortest_path.py --model gpt-4o --mode hard --prompt PROGRAM --SC 0
echo "✓ 任务 5 完成"





echo ""
echo "========================================="
echo "所有评估任务完成！"
echo "========================================="
echo "结果保存在 log/shortest_path/ 目录下"
echo ""
echo "查看结果:"
echo "  ls -lht log/shortest_path/"
echo "  cat log/shortest_path/gpt-4o-hard-*/prompt.txt"
echo ""
echo "对比不同 Prompt 策略的效果:"
echo "  grep 'Acc:' log/shortest_path/gpt-4o-hard-*/prompt.txt"
echo ""
