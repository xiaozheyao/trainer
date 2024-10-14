import json
import os
def eval():
    with open("jobs/eval.jsonl", "r") as f:
        eval_jobs = [json.loads(line) for line in f.readlines()]
    jobs = []
    for ejob in eval_jobs:
        base_model = ejob['ft_config']['model_name']
        
        job = f"python /xyao/code/trainer/eval_lora.py --ckpt-path {ejob['ckpt_path']} --base-model {base_model} --test-set {ejob['ft_config']['test_path']}"
        jobs.append(job)
    for i, job in enumerate(jobs):
        job_id = i % 10
        job = f'sbatch -A a09 --reservation=sai-shared --job-name ft-13b-{job_id} --dependency singleton --environment trainer --output logs/%A.out --wrap="cd /xyao/code/trainer && {job}"'
        if i<5:
            print(job)
        os.system(job)
    

if __name__ =="__main__":
    eval()