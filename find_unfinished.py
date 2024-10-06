import os
import json
from typing import List

bs = 16
nd = 2


def cleanup(args):
    print(args)
    output_directory = "outputs"
    results = os.listdir(output_directory)
    existing_configs = []
    for res in results:
        # read ft_config.json
        with open(os.path.join(output_directory, res, "ft_config.json"), "r") as f:
            config = json.load(f)
        existing_configs.append(config)
    print(existing_configs[0])
    print(f"existing: {len(existing_configs)}")
    lrs = [float(lr) for lr in args.lrs.split(",")]
    models = args.base_models.split(",")
    epochs = [int(x) for x in args.epochs.split(",")]
    ranks = [int(rank) for rank in args.lora_ranks.split(",")]
    remaining_jobs = []
    train_path = os.path.abspath(args.train_path)
    test_path = os.path.abspath(args.test_path)
    total_jobs = 0
    for model in models:
        model_size = model.split("/")[-1].split("-")[-2]
        for lr in lrs:
            for rank in ranks:
                for epoch in epochs:
                    job = {
                        "model": model,
                        "lr": lr,
                        "rank": rank,
                        "epoch": epoch,
                    }
                    total_jobs += 1
                    existing_in_results = False
                    for config in existing_configs:
                        if config['model_name'] == model and config['lr'] == lr and config['lora_rank'] == rank and config['num_epochs'] == epoch:
                            existing_in_results = True
                            break
                    if not existing_in_results:
                        remaining_jobs.append(job)
    print(f"total: {total_jobs}")
    print(len(remaining_jobs))
    print(remaining_jobs[0])
    jobs = []
    for job in remaining_jobs:
        job = f"python finetune.py --batch-size-per-device {bs} --num-devices {nd} --model_name {model} --output_dir outputs/ --lr {lr} --num-epochs {epoch} --ds-config deepspeed_configs/zero_3_llama_2_{model_size}.json --train-path {train_path} --special-token-path {args.special_token_path} --test-path {test_path} --lora --lora-rank {rank}"
        jobs.append(job)
    for job in jobs:
        with open("jobs.sh", "a") as f:
            f.write(f"ts -G 2 {job}" + "\n")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-models",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument("--lrs", type=str, default="5e-5,1e-5,5e-6,1e-6")
    parser.add_argument("--epochs", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora-ranks", type=str, default="8,16,32,64,128,256")
    parser.add_argument("--train-path", type=str, default="./data/train.jsonl")
    parser.add_argument("--test-path", type=str, default="./data/test.jsonl")
    parser.add_argument("--special-token-path", type=str, default="./data/tokens.json")
    args = parser.parse_args()
    cleanup(args)