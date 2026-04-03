#!/bin/bash

DATA_DIR="/inspire/hdd/project/robot-reasoning/public/RHOS/xianchao/RoboDiag/robotwin/RoboTwin/data"

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data folder $DATA_DIR not exist"
    exit 1
fi

# 开启 nullglob，使无匹配时通配符展开为空数组
shopt -s nullglob

# 获取所有一级子目录
task_dirs=("$DATA_DIR"/*/)

# 关闭 nullglob（可选）
shopt -u nullglob

# 统计目录数量
num_dirs=${#task_dirs[@]}

echo "找到的任务目录数量: $num_dirs"

# 检查是否为期望的数量（这里硬编码为50）
if [ $num_dirs -ne 50 ]; then
    echo "错误：期望 50 个任务目录，实际找到 $num_dirs 个，脚本终止。"
    exit 1
fi

# 遍历 data 目录下的一级子目录
for task_dir in "$DATA_DIR"/*/ ; do
    task_dir="${task_dir%/}"
    task_name="$(basename "$task_dir")"
    echo "processing: $task_name"
    
    # 固定为 demo_clean 和 50
    bash process_data_pi0.sh "$task_name" "demo_clean" "50"
    
    # python scripts/process_data.py "$task_name" "demo_clean" "50"
done

echo "Done!!!"