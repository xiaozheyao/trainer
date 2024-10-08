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
    eval_jobs = eval_jobs[args.start:args.end]
    for job in tqdm(eval_jobs):
        print(f"Evaluating {job['ckpt_path']}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            job['ckpt_path'],
            legacy=True
        )
        stop_token_embeding = tokenizer(
            STOP_TOKEN,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to("cuda")

        def custom_stopping_criteria(embeddings, *args, **kwargs) -> bool:
            return stop_token_embeding in embeddings
        
        stopping_criteria = transformers.StoppingCriteriaList([custom_stopping_criteria])
        results = []
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        model.resize_token_embeddings(len(tokenizer))
        model: peft.PeftModel = peft.PeftModel.from_pretrained(
            model=model,
            model_id=job['ckpt_path'],
        )
        
        model = model.merge_and_unload()
        model.to("cuda")
        prompts = build_prompts(args)
        
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt["input"], return_tensors="pt")["input_ids"].to("cuda")
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
        with open(os.path.join(job['ckpt_path'], "eval_results.json"), "w") as f:
            json.dump(results, f)

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index to start from",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Index to end at",
    )
    args = parser.parse_args()
    eval(args)