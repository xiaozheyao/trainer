import json
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "hf_ckpts/full_ft"
quant_path = "hf_ckpts/awq_full_ft"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, device_map="cuda", **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_dataset():
    with open("data/train.jsonl", "r") as fp:
        examples = [json.loads(line) for line in fp.readlines()]
    return [x["input"] for x in examples][:256]


# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_dataset())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
