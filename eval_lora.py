import os
import json
import peft
import torch
import transformers
from tqdm import tqdm

STOP_TOKEN = "<END_A>"

def custom_stopping_criteria(embeddings, *args, **kwargs) -> bool:
    return stop_token_embeding in embeddings
    
def build_prompts(args):
    data = []
    with open(args.test_set, "r") as f:
        prompts = [json.loads(line) for line in f.readlines()]
    for prompt in prompts:
        input_text = prompt["input"].split("<START_A>")[0] + "<START_A>"
        output_text = "<START_A>" + prompt["input"].split("<START_A>")[1]
        data.append({"input": input_text, "output": output_text})
    return data

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--test-set", type=str, required=True)
    batch_size = 32
    args = parser.parse_args()
    print(f"Evaluating {args.ckpt_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.ckpt_path,
        legacy=True
    )
    stop_token_embeding = tokenizer(
        STOP_TOKEN,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to("cuda")
    
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
        model_id=args.ckpt_path,
    )
    model = model.merge_and_unload()
    model.to("cuda")
    prompts = build_prompts(args)
    with torch.no_grad():
        # batch process prompts
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            encoding = tokenizer([x['input'] for x in batch_prompts], padding=True, return_tensors='pt').to("cuda")
            generation_output = model.generate(
                **encoding,
                output_scores=True,
                max_new_tokens=500,
                stopping_criteria=stopping_criteria,
            )
            decoded = tokenizer.batch_decode(generation_output)
            for j in range(len(batch_prompts)):
                results.append(
                    {
                        "input": batch_prompts[j]["input"],
                        "output": decoded[j].replace("</s>", ""),
                        "target": batch_prompts[j]["output"],
                    }
                )
    with open(os.path.join(args.ckpt_path, "eval_results.json"), "w") as f:
        json.dump(results, f)