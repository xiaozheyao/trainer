import json
import transformers
import torch
from tqdm import tqdm
from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer, TextStreamer
from deltazip import AutoDeltaZipModelForCausalLM

STOP_TOKEN = "<END_A>"


def build_prompts():
    with open("data/test.jsonl", "r") as f:
        prompts = [json.loads(line) for line in f.readlines()]
    data = []
    for prompt in prompts:
        input_text = prompt["input"].split("<START_A>")[0] + "<START_A>"
        output_text = "<START_A>" + prompt["input"].split("<START_A>")[1]
        data.append({"input": input_text, "output": output_text})
    return data


def build_model(args):
    if args.is_awq:
        model = AutoAWQForCausalLM.from_quantized(args.ckpt_path, fuse_layers=True)
    elif args.is_sparsegpt:
        model = AutoDeltaZipModelForCausalLM.from_compressed(
            args.ckpt_path, strict=True, device="cpu", unpack=True
        )
        model = model.half()
        model = model.to("cuda")
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.ckpt_path, torch_dtype=torch.bfloat16
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.ckpt_path)
    return model, tokenizer


def evaluate(args):
    print(args)
    prompts = build_prompts()
    model, tokenizer = build_model(args)
    model.eval()
    model.to("cuda")
    stop_token_embeding = tokenizer(
        STOP_TOKEN, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to("cuda")

    def custom_stopping_criteria(embeddings, *args, **kwargs) -> bool:
        return stop_token_embeding in embeddings

    stopping_criteria = transformers.StoppingCriteriaList([custom_stopping_criteria])
    results = []

    with torch.no_grad():
        for prompt in tqdm(prompts):
            input_ids = tokenizer(prompt["input"], return_tensors="pt")["input_ids"].to(
                "cuda"
            )
            generation_output = model.generate(
                input_ids,
                output_scores=True,
                max_new_tokens=500,
                stopping_criteria=stopping_criteria,
            )
            decoded = tokenizer.batch_decode(generation_output)
            results.append(
                {
                    "input": prompt["input"],
                    "output": decoded,
                    "target": prompt["output"],
                }
            )
    with open(args.output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to output directory. Defaults to the orginal checkpoint directory.",
        required=True,
    )
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--is-awq", action="store_true", default=False)
    parser.add_argument("--is-sparsegpt", action="store_true", default=False)
    args = parser.parse_args()
    print("Starting model evaluation")
    evaluate(args)
