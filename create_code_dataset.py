from datasets import load_dataset
import json
import os

dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")

dataset_splits = {
    "train": load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[:80%]"),
    "test": load_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-110K", split="train[80%:100%]"
    ),
}


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    with open("data/code/tokens.json", "w") as f:
        tokens = {}
        tokens["tokens"] = ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        f.write(json.dumps(tokens))

    for key, ds in dataset_splits.items():
        with open(f"data/code/{key}.jsonl", "w") as f:
            for item in ds:
                newitem = {}
                newitem["input"] = (
                    f"<START_Q>{item['instruction']}<END_Q>"
                    f"<START_A>{item['response']}<END_A>"
                )
                f.write(json.dumps(newitem) + "\n")


if __name__ == "__main__":
    main()
