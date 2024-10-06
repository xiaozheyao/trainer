import os

all_results = os.listdir("outputs")
for res in all_results:
    folders = [f for f in os.listdir(f"outputs/{res}") if os.path.isdir(f"outputs/{res}/{f}")]
    for folder in folders:
        for subfolder in [x for x in os.listdir(f"outputs/{res}/{folder}") if os.path.isdir(f"outputs/{res}/{folder}/{x}")]:
            if "error.txt" in os.listdir(f"outputs/{res}/{folder}/{subfolder}"):
                print(f"outputs/{res}/{folder}/{subfolder}")
                os.system(f"rm -rf outputs/{res}")