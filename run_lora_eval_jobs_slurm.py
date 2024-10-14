import json

def eval():
    with open("jobs/eval.jsonl", "r") as f:
        eval_jobs = [json.loads(line) for line in f.readlines()]
    for ejob in eval_jobs:
        base_model = ejob['ft_config']['model_name']
        print(ejob)
        job = f"python /xyao/code/trainer/eval_lora.py --ckpt-path {ejob['ckpt_path']} --base-model {base_model} --test-set {ejob['ft_config']['test_path']}"
        print(job)
        exit(0)
    

if __name__ =="__main__":
    eval()