from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description="Script that uses a dataset path.")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory or file."
)
parser.parse_args()
dataset_path = parser.dataset_path

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", split="train", streaming=False)
ds.save_to_disk(f"{dataset_path}/datasets/fineweb_edu-350b")

ds = load_dataset("nvidia/Nemotron-CC-Math-v1", "4plus")
ds.save_to_disk(f"{dataset_path}/datasets/Nemotron-CC-Math-v1-4plus")

ds = load_dataset("nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-General")
ds.save_to_disk(f"{dataset_path}/datasets/Nemotron-Pretraining-SFT-v1-General")

ds = load_dataset("nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-Code")
ds.save_to_disk(f"{dataset_path}/datasets/Nemotron-Pretraining-SFT-v1-Code")

ds = load_dataset("nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-MATH")
ds.save_to_disk(f"{dataset_path}/datasets/Nemotron-Pretraining-SFT-v1-Math")