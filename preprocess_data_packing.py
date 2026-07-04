from train import DEFAULT_SYS_PROMPT, Message
from typing import Any, Optional
from dataclasses import field
from transformers import AutoTokenizer
from jsonargparse import CLI
from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
from trl import pack_dataset
import torch

def format_and_tokenize_examples(examples, tokenizer, q_col, a_col, max_length, take_loss_over_all_tokens, return_type="pt"):
    conversations = []
    for idx in range(len(examples[q_col])):
        if q_col != "text":
            messages = [
                Message(role="system", content=DEFAULT_SYS_PROMPT),
                Message(role="user", content=examples[q_col][idx].strip()),
                Message(role="Huginn", content=examples[a_col][idx].strip()),
            ]
        else:
            messages = examples[q_col][idx].strip() + tokenizer.eos_token
        conversations.append(messages)
    
    if q_col != "text":
        chat_encoding = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            padding="max_length",
            max_length=max_length + 1,
            return_tensors=return_type,
            return_dict=True,
            truncation=True,
        )
        if take_loss_over_all_tokens:
            chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]
    else:
        chat_encoding = tokenizer(
            conversations,
            padding=False,
            # max_length=max_length + 1,
            return_tensors=return_type,
            truncation=False,
            add_special_tokens=True,
        )
        chat_encoding["assistant_masks"] = chat_encoding["attention_mask"]

    return {
        "input_ids": chat_encoding["input_ids"],
        # "mask": chat_encoding["assistant_masks"],
        "attention_mask": chat_encoding["attention_mask"],
    }

def pad_or_truncate(example, tokenizer_pad_id, max_len):
    for key, pad_id in [('input_ids', tokenizer_pad_id), ('attention_mask', 0)]:
        tensor = example[key]

        length = tensor.shape[0]

        if length < max_len:
            pad_length = max_len - length
            tensor = torch.cat([tensor, torch.full((pad_length,), pad_id, dtype=tensor.dtype)])
        elif length > max_len:
            tensor = tensor[:max_len]

        example[key] = tensor

    return example

def process_data(
    tokenizer_name: str = "smcleish/Recurrent-Llama-3.2-untrained",
    out_path: str = "del",
    dataset_location: str = "/p/vast/$USER/datasets/fineweb_edu",
    q_col: str = "text",
    a_col: str = "answer",
    dataset_config: str = "main",
    max_length: int = 1024,
    max_samples: Optional[int] = None,
    take_loss_over_all_tokens: bool = False,
    num_proc: int = 96,
    pack: bool = True,
    batch_size: int = 1024,
    wrapped_packing: bool = True,
    cache_path: str = "/p/lustre/$USER",
    save_path: str = "/p/vast/$USER"
):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if wrapped_packing:
        tokenizer.model_max_length = int(1e30)

    packing_str = '_packed_wrapped' if wrapped_packing else ('_packed' if pack else '')
    if wrapped_packing:
        assert pack, "Can't have wrapped_packing=true without pack=true"
    dataset_save_dir = f"{save_path}/llama_huginn_preprocessed_data{packing_str}/{tokenizer_name}/{out_path}/dataset"

    dataset = load_from_disk(dataset_location, dataset_config)
    
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    tokenized_dataset = dataset.map(
        format_and_tokenize_examples,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        fn_kwargs={"tokenizer": tokenizer, "q_col": q_col, "a_col": a_col, "max_length": max_length, "take_loss_over_all_tokens": take_loss_over_all_tokens, "return_type": None if pack else "pt"},
        cache_file_name=f"{cache_path}/posttrain_huginn/processing_cache/tmp_cache_{out_path}.arrow",
    )

    if pack:
        # https://github.com/huggingface/trl/commit/0353d6766144981040ce47eb16925bb7f5e6ecf7 ffd vs bfd = same code just renamed
        tokenized_dataset = pack_dataset(
            tokenized_dataset, 
            seq_length=max_length+1, 
            strategy="wrapped" if wrapped_packing else "ffd", 
            map_kwargs={
                "num_proc": num_proc,
                "desc": "packing",
                "batch_size": batch_size,
                "writer_batch_size": batch_size,
                "cache_file_name": f"{cache_path}/posttrain_huginn/processing_cache/tmp_cache_packing{'_wrapped' if wrapped_packing else ''}_{out_path}.arrow",
            },
        )

        tokenized_dataset.set_format("pt")
        if "position_ids" in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns(["position_ids"])

        tokenized_dataset = tokenized_dataset.map(
            pad_or_truncate, 
            fn_kwargs={
                "tokenizer_pad_id": tokenizer.pad_token_id, 
                "max_len": max_length + 1
            }, 
            num_proc=num_proc,
            cache_file_name=f"{cache_path}/posttrain_huginn/processing_cache/tmp_cache_padding_{out_path}.arrow",
        )

    tokenized_dataset.save_to_disk(dataset_save_dir, num_proc=num_proc, max_shard_size="2GB")
    tokenized_dataset.cleanup_cache_files()
    dataset.cleanup_cache_files()


if __name__ == "__main__":
    CLI(process_data)