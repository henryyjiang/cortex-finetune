from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import List
import os
import json
from jsonargparse import CLI

def get_model_and_tokenizer(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
        torch_dtype=torch.float32,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

@torch.no_grad()
def main(
    model_name: str, 
    ckpts: List[int], 
    batch_size: int = 64, 
    device: str = "cuda",
    eval_file_path: str = "path/eval_dataset/shard-00512.parquet"
):
    dataset = load_dataset(
        "parquet", 
        data_files=eval_file_path,
    )["train"].select(range(1024))
    dataset.set_format("pt")

    for ckpt in ckpts:
        output = {}
        model, tokenizer = get_model_and_tokenizer(f"{model_name}/model_only_chkpt_{ckpt}", device)

        batch = {"input_ids": [], "labels": []}
        for data_idx, inputs in tqdm(enumerate(dataset, start=1)):
            input_ids = inputs["input_ids"][:-1].to(dtype=torch.long, device=device, non_blocking=True)
            mask = ~inputs["attention_mask"].bool()
            labels = torch.where(mask[1:], -100, inputs["input_ids"][1:]).to(
                dtype=torch.long, device=device, non_blocking=True
            )
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

            if (data_idx % batch_size == 0) or (data_idx == len(dataset)):
                input_ids = torch.stack(batch["input_ids"], dim=0)
                labels = torch.stack(batch["labels"], dim=0)
                mask = (labels != -100).float()

                for num_rec in [1,2,4,8,16,32,64]:
                    logits = model(input_ids, num_steps=torch.tensor([num_rec,0], device=model.device)).logits

                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction='none' 
                    )
                    loss = loss.view(logits.size(0), logits.size(1))
                    loss_per_sample = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                    this_list = output.get(num_rec, [])
                    this_list.append(loss_per_sample.tolist())
                    output[num_rec] = this_list
                    del logits

                batch["input_ids"] = []
                batch["labels"] = []

        output_dir = f"{os.getcwd()}/loss_over_rec_eval/{model_name.split('/')[-1]}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/chkpt_{ckpt}.json", "w") as f:
            json.dump(output, f)

if __name__ == "__main__":
    CLI(main)

# HIP_VISIBLE_DEVICES=0 python multi_recurence_eval.py huginn_llama/YOUR_MODEL [1000] --eval_file_path=YOUR_PATH