import os
import json
import peft
import torch
import transformers
from tqdm import tqdm



def main(args):
    jobs = os.listdir(args.lora_ckpts)
    eval_jobs = []
    for job in jobs:
        ft_config_file = os.path.join(args.lora_ckpts, job, "ft_config.json")
        with open(ft_config_file, "r") as f:
            ft_config = json.load(f)
        for sub_dir in os.listdir(os.path.join(args.lora_ckpts, job)):
            if os.path.isdir(os.path.join(args.lora_ckpts, job, sub_dir)):
                for lora_ckpt in os.listdir(os.path.join(args.lora_ckpts, job, sub_dir)):
                    if os.path.isdir(os.path.join(args.lora_ckpts, job, sub_dir, lora_ckpt)):
                        ckpts = os.listdir(os.path.join(args.lora_ckpts, job, sub_dir, lora_ckpt))
                        for ckpt in ckpts:
                            ckpt_path = os.path.join(args.lora_ckpts, job, sub_dir, lora_ckpt, ckpt)
                            if os.path.isdir(ckpt_path):
                                epoch_id = int(ckpt.split("_")[1])
                                eval_jobs.append({
                                    "training_id": job,
                                    "epoch": epoch_id,
                                    "ft_config": ft_config,
                                    "ckpt_path": ckpt_path,
                                })
    STOP_TOKEN = "<END_A>"
    with open("jobs/eval.jsonl", "w") as f:
        for job in eval_jobs:
            json.dump(job, f)
            f.write("\n")
    

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-ckpts",
        type=str,
        help="Path to the LoRA checkpoint.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to the base model checkpoint.",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        help="Path to the test set.",
    )
    args = parser.parse_args()
    main(args)