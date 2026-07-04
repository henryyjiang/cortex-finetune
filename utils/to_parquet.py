from datasets import load_from_disk
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Script that uses a dataset path.")
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory or file to load."
)
parser.add_argument(
    "--dataset_save_dir",
    type=str,
    required=True,
    help="Path to the dataset directory or file to save."
)
parser.parse_args()
dataset_path = parser.dataset_path

ds = load_from_disk(parser.dataset_path)

num_shards = 516 # 4 for val -- has to be moved manually after
for index in tqdm(range(num_shards), desc="Saving shards"):
    print(index)
    shard = ds.shard(index=index, num_shards=num_shards, contiguous=True)
    shard.to_parquet(f"{parser.dataset_save_dir}/shard-{index:05d}.parquet")
