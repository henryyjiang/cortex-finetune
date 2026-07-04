from torch.utils.data import IterableDataset
from pathlib import Path
import pyarrow.parquet as pq
import torch
import random
import hashlib
from torch.utils.data._utils.collate import collate_tensor_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from natsort import natsorted

class ParquetStreamPure(IterableDataset):
    # https://github.com/seal-rg/recurrent-pretraining/blob/59e0b69b2d96a59cbbe79c9d5034d89ecb5ab6f6/recpre/huggingface_dataset.py#L124
    """datasets-free version of (mostly) the same thing - shuffle not across files though
    a bit ironic to keep it in this file"""

    def __init__(
        self,
        dataset_folder_path="",
        seed=12345,
        shuffle=True,
        num_processes=1,
        process_rank=0,
        prefix="",
        verbose=False,
        shuffle_filenames=True,
        broadcast_glob=True,
        stateful=True,  # this dataset is by default (suprisingly) stateful, even outside of iter(dataset)!
        plan_for_later_rank_expansion_to=0,
        num_epochs=1,
    ):
        # Get file list, with distributed broadcast if needed
        if broadcast_glob and torch.distributed.is_initialized():
            if process_rank == 0:
                filenames = natsorted(str(p) for p in Path(dataset_folder_path).glob(f"{prefix}*.parquet"))
            else:
                filenames: list[str] = None  # type: ignore # believe
            obj = [filenames]
            torch.distributed.broadcast_object_list(obj, 0)
            parquet_files = obj[0]
        else:
            parquet_files = natsorted(str(p) for p in Path(dataset_folder_path).glob(f"{prefix}*.parquet"))
        if shuffle_filenames:
            random.Random(seed).shuffle(parquet_files)

        # Shard files for distributed training
        if plan_for_later_rank_expansion_to > 0:
            ranks_for_file_selection = max(plan_for_later_rank_expansion_to, num_processes)
        else:
            ranks_for_file_selection = num_processes
        self.parquet_files = (
            parquet_files[process_rank::ranks_for_file_selection] if ranks_for_file_selection > 1 else parquet_files
        )

        if len(self.parquet_files) < 1:
            raise ValueError(f"Empty dataset on rank {process_rank}")

        if num_epochs > 1:
            self.parquet_files = self.parquet_files * num_epochs

        self._ds_fingerprint = hashlib.shake_128(str(self.parquet_files).encode()).hexdigest(4)

        if verbose:
            print(
                f"Rank {process_rank}/{num_processes} has {len(self.parquet_files)}/{len(parquet_files) * num_epochs} parquet files | identifier={self._ds_fingerprint}"
            )
            print("First 10 parquet files:", ["/".join(fp.split("/")[-2:]) for fp in self.parquet_files[:10]])
            examples = pq.read_table(self.parquet_files[0], columns=["input_ids"]).slice(0, 3).to_pylist()  # Get 3 rows
            for i, example in enumerate(examples):
                print(f"Example {i}: {example['input_ids'][:12]}")  # First 12 tokens of each row
        self.shuffle = shuffle
        self.seed = seed
        self.process_rank = process_rank
        self.stateful = stateful
        # Initialize default state
        self._state_init()

    def _state_init(self):
        self._state = {
            "rng": random.Random(self.seed),
            "rng_state": (-1, [-1], None),
            "buffer": [],
            "file_idx": 0,
            "row_group_idx": 0,
            "fingerprint": self._ds_fingerprint,
        }

    def __iter__(self):
        if not self.stateful:
            self._state_init()

        while self._state["file_idx"] < len(self.parquet_files):
            if not self._state["buffer"]:
                # Refill buffer from current position
                pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
                if self._state["row_group_idx"] >= pf.num_row_groups:
                    print(
                        f"Rank {self.process_rank} | {self._state['file_idx']}-{self._state['row_group_idx']} | "
                        f" New file: {self.parquet_files[self._state['file_idx'] + 1]}"
                    )
                    self._state["file_idx"] += 1
                    self._state["row_group_idx"] = 0
                    continue

                self._read_buffer(pf)
                self._state["row_group_idx"] += 1

            while self._state["buffer"]:
                yield {"input_ids": torch.as_tensor(self._state["buffer"].pop(), dtype=torch.long), "attention_mask": torch.as_tensor(self._state["attn_mask_buffer"].pop(), dtype=torch.long),}

    def _read_buffer(self, parquet_file):
        batch = parquet_file.read_row_group(self._state["row_group_idx"])
        toks = batch.column("input_ids").to_pylist()
        amask = batch.column("attention_mask").to_pylist()

        if self.shuffle:
            # the last used state for a shuffle op
            self._state["rng_state"] = self._state["rng"].getstate()

            idx = list(range(len(toks)))
            self._state["rng"].shuffle(idx)

            self._state["buffer"] = [toks[i] for i in idx]
            self._state["attn_mask_buffer"] = [amask[i] for i in idx]
        else:
            self._state["buffer"] = toks
            self._state["attn_mask_buffer"] = amask

    def state_dict(self):
        # Pack all basic state and RNG state into one tensor for a single gather
        rng_0, rng_1, rng_2 = self._state["rng_state"]
        local_state = torch.tensor(
            [
                self._state["file_idx"],
                self._state["row_group_idx"] - 1,  # -1 because we need to reload the current buffer
                len(self._state["buffer"]),
                int(self._ds_fingerprint, 16),
                rng_0,
                *rng_1,
                rng_2 if rng_2 is not None else -1,
            ],
            device="cuda",
        )

        # Single gather for all state
        gathered_states = [torch.zeros_like(local_state) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_states, local_state)

        result = {
            "file_idx": [s[0].item() for s in gathered_states],
            "row_group_idx": [s[1].item() for s in gathered_states],
            "row_idx": [s[2].item() for s in gathered_states],
            "fingerprint": [hex(s[3].item())[2:] for s in gathered_states],  # type: ignore
            "rng_state": gathered_states,  # Full tensors for unpacking in load
        }

        return result

    def load_state_dict(self, state_dict, offset_ranks=False):
        rank = torch.distributed.get_rank()

        def get_value(key):  # helper for backward compat
            effective_rank = rank % len(state_dict["fingerprint"])
            return state_dict[key][effective_rank]

        if int(get_value("fingerprint"), 16) != int(self._ds_fingerprint, 16):
            print(
                f"WARNING Dataset fingerprint mismatch. Expected {self._ds_fingerprint}, got {get_value('fingerprint')}"
            )
            self._state["file_idx"] = 0
            self._state["row_group_idx"] = 0
            row_idx = 0
        else:
            # Load file IDs only if we can guarantee that this the same data source
            # otherwise these might run out of bounds
            self._state["file_idx"] = max(get_value("file_idx"), 0)
            self._state["row_group_idx"] = max(get_value("row_group_idx"), 0)
            row_idx = max(get_value("row_idx"), 0)
            if offset_ranks:
                row_idx = row_idx + rank % 1000

        if state_dict["rng_state"] is not None:
            # New packed format
            rng_state = state_dict["rng_state"][rank % len(state_dict["rng_state"])]
            # RNG state starts at index 4
            rng_state = (rng_state[4].item(), tuple(x.item() for x in rng_state[5:-1]), rng_state[-1].item())
            if rng_state[2] == -1:
                rng_state = (rng_state[0], rng_state[1], None)

            self._state["rng"] = random.Random()
            self._state["rng"].setstate(rng_state)
            self._state["rng_state"] = rng_state

        pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
        self._read_buffer(pf)
        self._state["buffer"] = self._state["buffer"][:row_idx]
        self._state["attn_mask_buffer"] = self._state["attn_mask_buffer"][:row_idx]
        self._state["row_group_idx"] += 1

def generic_collator(batch):
    to_return = {}

    keys = batch[0].keys()
    for k in keys:
        this_set = []
        for row in batch:
            assert (row[k].shape[0] == 1025) or (row[k].shape[0] == 2049), "row not padded correctly"
            this_set.append(row[k])
        to_return[k] = collate_tensor_fn(this_set)

    return to_return

def get_parquet_dataloader(world_size, global_rank, batch_size, path, testing=False, num_epochs=1):
    dataset = ParquetStreamPure(
        dataset_folder_path=path,
        seed=42,
        shuffle=not testing,
        shuffle_filenames=not testing,
        num_processes=world_size,
        process_rank=global_rank,
        prefix="",
        stateful=True,
        verbose=testing,
        num_epochs=num_epochs,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=generic_collator,
        num_workers=0,
        prefetch_factor=None,
    )
    return loader
