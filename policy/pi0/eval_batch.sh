export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_config=${1}
train_config_name=${2}
model_name=${3}
seed=${4}
gpu_id=${5}
tasks_file=${6:-./info.txt}   # 默认为 ./info.txt

if [ ! -f "$tasks_file" ]; then
    echo "错误：任务文件 $tasks_file 不存在" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root
count=0
max_tasks=50

# 从文件逐行读取 task_name
while IFS= read -r task_name; do
    # 跳过空行
    [ -z "$task_name" ] && continue
    # 达到50停止
    if [ $count -ge $max_tasks ]; then
        echo "已处理 ${max_tasks} 个任务，停止处理。"
        break
    fi
    echo "processing: $task_name"
    # test on demo_randomized train on demo_clean
    # bash eval_batch.sh demo_randomized pi0_base_aloha_robotwin_full demo_clean 0 0
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 

    count=$((count+1))
done < "$tasks_file"

echo "Done!!!"