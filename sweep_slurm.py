import os
from typing import List

# global parameters
bs = 16
nd = 4


def finetune(args):
    jobs = []
    lrs = [float(lr) for lr in args.lrs.split(",")]
    models = args.base_models.split(",")
    epochs = args.epochs.split(",")
    ranks = [int(rank) for rank in args.lora_ranks.split(",")]
    train_path = os.path.abspath(args.train_path)
    test_path = os.path.abspath(args.test_path)
    special_token_path = os.path.abspath(args.special_token_path)
    
    for model in models:
        model_size = model.split("/")[-1].split("-")[-2]
        for lr in lrs:
            for rank in ranks:
                for epoch in epochs:
                    job = f"python /xyao/code/trainer/finetune.py --batch-size-per-device {bs} --num-devices {nd} --model_name {model} --output_dir outputs/ --lr {lr} --num-epochs {epoch} --ds-config /xyao/code/trainer/deepspeed_configs/zero_3_llama_2_{model_size}.json --train-path {train_path} --special-token-path {special_token_path} --test-path {test_path} --lora --lora-rank {rank}"
                    jobs.append(job)
    
    for job in jobs[0:5]:
        print(job)
    
    for i, job in enumerate(jobs):
        job_id = i%15
        job = f'sbatch -p debug --job-name ft-13b-{i} --dependency singleton --environment trainer --output logs/%A.out --wrap="cd /xyao/code/trainer && {job}"'
        os.system(job)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", help="batch size", type=int, default=16)
    parser.add_argument(
        "--base-models",
        type=str,
        default="meta-llama/Llama-2-13b-hf",
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
    finetune(args)
