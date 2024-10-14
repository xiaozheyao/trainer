import os
import json
import peft
import torch
import transformers
from tqdm import tqdm

def build_prompts(args):
    data = []
    with open(args.test_set, "r") as f:
        prompts = [json.loads(line) for line in f.readlines()]
    for prompt in prompts:
        input_text = prompt["input"].split("<START_A>")[0] + "<START_A>"
        output_text = "<START_A>" + prompt["input"].split("<START_A>")[1]
        data.append({"input": input_text, "output": output_text})
    return data

def eval(args):
    STOP_TOKEN = "<END_A>"
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