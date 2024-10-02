if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file")
    args = parser.parse_args()
    with open(args.input) as f:
        data = json.load(f)
    correct = 0
    total = len(data)
    for datum in data:
        output = datum["output"][0].split("<START_A>")[1].split("<END_A>")[0]
        target = datum["target"].split("<START_A>")[1].split("<END_A>")[0]
        try:
            output = output.split("####")[1].strip()
            target = target.split("####")[1].strip()
            if output == target:
                correct += 1
        except Exception as e:
            pass
    print(f"Accuracy: {correct/total}, {correct}/{total}")
