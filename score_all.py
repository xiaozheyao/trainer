import os
import json

def process_data(data):
    total = len(data)
    correct = 0
    for datum in data:
        output = datum["output"][0].split("<START_A>")[1].split("<END_A>")[0]
        target = datum["target"].split("<START_A>")[1].split("<END_A>")[0]
        try:
            output = output.split("####")[1].strip()
            target = target.split("####")[1].strip()
            if output == target:
                correct += 1
        except Exception as e:
            pass
    return correct, total

with open('jobs/eval.jsonl', "r") as fp:
    eval_jobs = [json.loads(line) for line in fp.readlines()]
    
total_jobs = len(eval_jobs)

for job in eval_jobs:
    ckpt_path = job['ckpt_path']
    if "eval_results.json" in os.listdir(ckpt_path):
        with open(os.path.join(ckpt_path, "eval_results.json"), "r") as f:
            eval_results = json.load(f)
        print(f"job config: {job}")
        correct, total = process_data(eval_results)
        print(f"Accuracy: {correct/total}, {correct}/{total}")