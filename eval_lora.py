import os
import json
import peft
import torch
import transformers
from tqdm import tqdm
from transformers import TextGenerationPipeline
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
    prompts = build_prompts(args)[:5]
    stopping_criteria = transformers.StoppingCriteriaList([custom_stopping_criteria])
    generation_kwargs = {
        "max_new_tokens": 500,
        "generation_kwargs": {"stopping_criteria": stopping_criteria}
    }

    pipeline = TextGenerationPipeline(model, tokenizer, device="cuda")
    with torch.no_grad():
        outputs = pipeline([prompt["input"] for prompt in prompts])
    for i in range(len(prompts)):
        prompt = prompts[i]['input']
        target = prompts[i]["output"]
        output = outputs[i][0]['generated_text']
        results.append(
            {
                "input": prompt,
                "output": output,
                "target": target,
            }
        )
    print(results)
    # with open(os.path.join(args.ckpt_path, "eval_results.json"), "w") as f:
    #     json.dump(results, f)