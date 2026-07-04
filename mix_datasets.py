from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

def get_ds(split_path, inds):
    splits = []
    for split in inds:
        splits.append(load_from_disk(split_path(split)))
    return splits

split_path = lambda x: f"/p/vast1/pretrain/huginn_llama/llama_huginn_preprocessed_data_packed_wrapped/smcleish/Recurrent-TinyLlama-3T-untrained/tinyllama_1_1b_packed_350b_sample_shard_{x}_wrapped_packing/dataset"
fineweb_ds = get_ds(split_path, [0, 20])

split_path = lambda x: f"/p/vast1/pretrain/huginn_llama/llama_huginn_preprocessed_data_packed_wrapped/smcleish/Recurrent-TinyLlama-3T-untrained/tinyllama_1_1b_packed_nemotron_sft_v1_general_shard_{x}_wrapped_packing/dataset"
fineweb_ds += get_ds(split_path, range(5))
print(concatenate_datasets(fineweb_ds))

split_path = lambda x: f"/p/vast1/pretrain/huginn_llama/llama_huginn_preprocessed_data_packed_wrapped/smcleish/Recurrent-TinyLlama-3T-untrained/tinyllama_1_1b_packed_nemotron_sft_v1_math_shard_{x}_wrapped_packing/dataset"
nemotron_ds = [concatenate_datasets(get_ds(split_path, range(2)))]

splits = nemotron_ds + fineweb_ds
print(splits)
ds = concatenate_datasets(splits)
shuffled_dataset = ds.shuffle(seed=42, keep_in_memory=True)
flat_ds = shuffled_dataset.flatten_indices(cache_file_name="cache_path/tmp.arrow", num_proc=96)

num_shards = 516 # 4 for val
for index in tqdm(range(num_shards), desc="Saving shards"):
    print(index)
    shard = flat_ds.shard(index=index, num_shards=num_shards, contiguous=True)
    shard.to_parquet(f"$PROCESSED_DATA_PATH/smcleish/Recurrent-TinyLlama-3T-untrained/fineweb_nemotron_sft_general_math_even_mix/dataset/shard-{index:05d}.parquet")
