ts -G 1 python eval.py --output-path eval/eval_fmt_7b_awq.jsonl --ckpt-path hf_ckpts/awq_full_ft --is-awq

ts -G 1 python eval.py --output-path eval/eval_fmt_7b_sparsegpt.jsonl --ckpt-path hf_ckpts/compressed_4b0.5s_128_nodelta  --is-sparsegpt