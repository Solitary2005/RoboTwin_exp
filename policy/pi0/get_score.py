'''
遍历每个task文件夹下的"pi0/demo_randomized/demo_clean/一串时间/_result.txt"
提取第4行作为模型加载时间
提取第5行作为平均推理时间
提取第6行为成功率
Task | Model Load Time | Average Whole Task Infer Time | Success Rate

'''

import os
import pandas as pd

csv_path = "/inspire/hdd/global_user/liuxiaotong-253108540242/zwj/workspace/RoboTwin/policy/pi0/eval_res_50_tasks.csv"
base_root = "/inspire/hdd/global_user/liuxiaotong-253108540242/zwj/workspace/RoboTwin/eval_result_pi0_full"
all_task_list = sorted(os.listdir(base_root))
assert len(all_task_list) == 50
all_model_time = []
all_infer_time = []
all_sr = []


for task_name in all_task_list:
    res_base_root = os.path.join(base_root, task_name, "pi0/demo_randomized/demo_clean")
    res_base_dir = os.listdir(res_base_root)[0]
    res_path = os.path.join(res_base_root, res_base_dir, "_result.txt")
    with open(res_path, "r") as file:
        all_info = file.readlines()
        all_model_time.append(all_info[4].split()[-1])
        all_infer_time.append(all_info[5].split()[-1])
        all_sr.append(all_info[6])

        
all_res = {
    "Task": all_task_list,
    "Model Load Time": all_model_time,
    "Average Whole Task Infer Time": all_infer_time,
    "Success Rate": all_sr
}       
df = pd.DataFrame(all_res)
df.to_csv(csv_path)        