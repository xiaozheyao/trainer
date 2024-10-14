import json

def eval(args):
    with open("jobs/eval.jsonl", "r") as f:
        eval_jobs = [json.loads(line) for line in f.readlines()]
    for ejob in eval_jobs:
        base_model = ejob['ft_config']['model_name']
        job = f"python /xyao/code/trainer/eval_lora.py --ckpt-path {ejob['ckpt_path']} --base-model {base_model} --test-set {args.test_set}"
        print(job)
        exit(0)
    

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    eval(args)