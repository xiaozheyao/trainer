def finetune(args):
    print(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", help="batch size")
    parser.add_argument(
        "--base-model",
        type=str,
        choices=[
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
    )
    args = parser.parse_args()
    finetune(args)