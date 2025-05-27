import time
from pathlib import Path
from typing import Any, Generator, Literal, TypeAlias, TypedDict

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from jaxtyping import Float, Int
from pyarrow import dataset as ds
from pydantic_config import BaseConfig
from torch.utils.data import DataLoader, IterableDataset

from zeroband.training import envs
from zeroband.training.data_prefetch import STABLE_FILE, GCPPrefetcher
from zeroband.utils.logger import get_logger
from zeroband.utils.parquet import pa_schema
from zeroband.utils.world_info import get_world_info


class DataConfig(BaseConfig):
    path: str = "datasets/fineweb-edu"
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 1
    timeout: float = 3600

    local_dir: str = "/dev/shm/zeroband/data"  # only used if path is gcp

    ignore_zero_advantages: bool = False  # don't use in local setup


class DatasetOutput(TypedDict):
    # token level
    input_ids: Int[torch.Tensor, "seq"]
    advantages: Float[torch.Tensor, "seq"]
    loss_mask: Int[torch.Tensor, "seq"]
    logprobs: Float[torch.Tensor, "seq"]

    # sample level
    seq_lens: Int[torch.Tensor, "1"]
    rewards: Float[torch.Tensor, "1"]
    task_rewards: Float[torch.Tensor, "1"]
    length_penalties: Float[torch.Tensor, "1"]
    target_lengths: Int[torch.Tensor, "1"]


class FakeTokenizedDataset(IterableDataset):
    """A dummy dataset that generates random sequences with the full schema including new columns."""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def __iter__(self) -> Generator[DatasetOutput, Any, None]:
        while True:
            world_info = get_world_info()

            # simulate variable sequence length
            current_seq_len = torch.randint(self.seq_len // 2, self.seq_len + 1, (1,)).item()

            prompt_len = torch.randint(0, current_seq_len // 2 + 1, (1,)).item()
            output_len = current_seq_len - prompt_len
            
            input_ids = torch.randint(3, self.vocab_size, (current_seq_len,))
            
            # Simulate per-token advantages, logprobs, and loss_mask
            advantages = torch.randn(current_seq_len)
            # Logprobs for prompt part are often not used or available, set to 0; random for output part
            logprobs_prompt = torch.zeros(prompt_len)
            logprobs_output = torch.randn(output_len)
            logprobs = torch.cat((logprobs_prompt, logprobs_output), dim=0)
            
            loss_mask = torch.zeros(current_seq_len, dtype=torch.long)
            if output_len > 0:
                loss_mask[prompt_len:] = 1 # Loss only on output tokens
            
            # Apply scalar advantage only to output tokens
            scalar_advantage_val = torch.randn(1).item()
            advantages_per_token = torch.zeros_like(advantages)
            if output_len > 0:
                advantages_per_token[prompt_len:] = scalar_advantage_val

            self.step += 1

            yield {
                "input_ids": input_ids,
                "advantages": advantages_per_token, # Use per-token advantages
                "loss_mask": loss_mask,
                "logprobs": logprobs,
                "seq_lens": torch.tensor([current_seq_len], dtype=torch.long),
                "rewards": torch.tensor([torch.randn(1).item()], dtype=torch.float),
                "task_rewards": torch.tensor([torch.randn(1).item()], dtype=torch.float),
                "length_penalties": torch.tensor([torch.randn(1).item()], dtype=torch.float),
                "target_lengths": torch.tensor([self.seq_len], dtype=torch.long),
            }


def validate_schema_pa_file(file: Path):
    """Check if the schema of the parquet file is the same as the schema of the pa_schema"""
    try:
        parquet_schema = pq.read_schema(file)
        # Check if all columns in pa_schema are in parquet_schema (parquet_schema can have more)
        for field in pa_schema:
            if field.name not in parquet_schema.names:
                print(f"Error: Column '{field.name}' from pa_schema not found in file {file}. File columns: {parquet_schema.names}")
                return False
            # TODO: Could add type checking here too if necessary
        return True
    except Exception as e:
        print(f"Error reading schema for file {file}: {e}")
        return False


def _get_dataset_from_files_step(
    step_count: int, path: Path, timeout: float, batch_size: int, ignore_zero_advantages: bool, use_stable_file: bool
) -> ds.Dataset:
    """Get all the files for a given step. Waits until the step is created which is indicated by the stable file."""
    logger = get_logger("TRAIN")
    step_path = path / f"step_{step_count}"

    start_time = time.time()

    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0

    wait_count = 0

    while True:
        files = list(step_path.glob("*.parquet"))
        if envs.TRAINING_ENABLE_ACCEPTED_CHECK:
            accepted_flags = set(i.stem for i in step_path.glob("accepted/*.parquet"))
            files = [i for i in files if i.stem in accepted_flags]

        rows = 0

        if len(files) > 0:
            try:
                # Validate schema for the first file found as a quick check
                if files and not validate_schema_pa_file(files[0]):
                     logger.warn(f"Schema validation failed for {files[0]}. Attempting to load with pa_schema anyway.")
                     # Fallback to trying with explicit schema if validation fails, pyarrow might be robust
                     current_dataset = ds.dataset(files, format="parquet", schema=pa_schema)
                else:
                    current_dataset = ds.dataset(files, format="parquet") # Rely on schema embedding or pa_schema if needed later

                if ignore_zero_advantages:
                    # Ensure 'advantages' column exists before filtering
                    if "advantages" in current_dataset.schema.names:
                        current_dataset = current_dataset.filter(ds.field("advantages") != 0)
                    else:
                        logger.warn("'advantages' column not found for filtering, skipping ignore_zero_advantages.")

                rows = current_dataset.count_rows()
            except Exception as e:
                logger.warn(f"Error loading dataset for step {step_count}: {e}, files: {files}")
                rows = 0 # Reset rows on error

            if rows >= batch_size: # Ensure enough rows globally for the batch
                logger.info(f"Dataset for step {step_count} has enough samples. rows: {rows} and {len(files)} files")

                if use_stable_file:
                    stable_file = step_path / STABLE_FILE
                    if stable_file.exists():
                        logger.info(f"Stable file {stable_file} exists for step {step_count}, returning dataset")
                        return current_dataset
                else: # Not using stable file, proceed if rows are sufficient
                    return current_dataset

        if time.time() - start_time > timeout:
            logger.info("raising timeout")
            raise TimeoutError(f"Timeout waiting for step {step_count} to be created")

        if wait_count % 600 == 0:  # log every 5 minutes
            logger.info(
                f"[data_worker:{worker_id}] Waiting for {step_path} to have enough samples. len(files): {len(files)}, Current rows: {rows}, target: {batch_size}"
            )
            if use_stable_file:
                stable_file = step_path / STABLE_FILE
                if not stable_file.exists():
                    logger.info(f"Stable file {stable_file} does not exist for step {step_count}, waiting for it to be created")

        wait_count += 1
        time.sleep(0.5)


def _should_skip_index(index: int, world_size: int, rank: int, num_workers: int, workers_id: int) -> bool:
    """
    This function is used to skip the index if it is not the responsibility of the current worker.
    It take into account the number of workers as well as rank.

    Its equivalent to checking if index is in samples[rank::world_size][workers_id::num_workers]

    Returns:
        True if the index should be skipped
        False if the index should be processed
    """
    # First, check if the index belongs to this rank (distributed across world_size)
    if (index % world_size) != rank:
        return True

    # Next, compute the position within the rank's subset
    rank_position = index // world_size

    # Check if this position belongs to this worker (distributed across num_workers)
    if (rank_position % num_workers) != workers_id:
        return True

    # If we passed both checks, this index should be processed by this worker
    return False


class ParquetDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        batch_size: int, # Global batch size target for a step
        timeout: float,
        step_count_init: int,
        ignore_zero_advantages: bool,
        pq_read_bs: int = 2048, 
        use_stable_file: bool = False,
    ):
        self._logger = get_logger("TRAIN")
        self._path = path
        self._batch_size = batch_size
        self._pq_read_bs = pq_read_bs
        self._world_info = get_world_info()
        self._step_count = step_count_init - 1
        self._timeout = timeout
        self._ignore_zero_advantages = ignore_zero_advantages
        self._use_stable_file = use_stable_file

    def __iter__(self) -> Generator[DatasetOutput, Any, None]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        global_sample_idx_for_sharding = 0 # Index used for sharding across all files in a step

        while True:
            self._step_count += 1
            self._logger.debug(msg=f"data: Worker {worker_id}/Rank {self._world_info.rank} processing step {self._step_count}")

            dataset = _get_dataset_from_files_step(
                self._step_count, self._path, self._timeout, self._batch_size, 
                self._ignore_zero_advantages, self._use_stable_file
            )

            # Define required columns based on pa_schema which should be the source of truth
            required_columns_from_schema = pa_schema.names
            
            # Ensure all required columns for DatasetOutput are in the schema
            # This is more of a check for development; production assumes schema matches.
            expected_parquet_cols = [
                "input_tokens", "output_tokens", "logprobs", "loss_mask", 
                "advantages", "rewards", "task_rewards", "length_penalties", "target_lengths"
            ]
            for col in expected_parquet_cols:
                if col not in required_columns_from_schema:
                    # This indicates a mismatch between pa_schema and expected structure
                    raise ValueError(f"Critical: Column '{col}' expected for processing is not in pa_schema ({required_columns_from_schema})")

            scanner = dataset.scanner(columns=required_columns_from_schema, batch_size=self._pq_read_bs)
            
            for record_batch in scanner.to_reader():
                for i in range(record_batch.num_rows):
                    if _should_skip_index(global_sample_idx_for_sharding, self._world_info.world_size, self._world_info.rank, num_workers, worker_id):
                        global_sample_idx_for_sharding += 1
                            continue
                    global_sample_idx_for_sharding +=1

                    # Extract data for the current row
                    input_tokens_list = record_batch["input_tokens"][i].as_py()
                    output_tokens_list = record_batch["output_tokens"][i].as_py()
                    
                    full_input_ids_list = input_tokens_list + output_tokens_list
                    full_input_ids = torch.tensor(full_input_ids_list, dtype=torch.long)
                    current_seq_len = len(full_input_ids)

                    # Logprobs from Parquet correspond to output_tokens_list
                    # Pad with 0.0 for prompt tokens, use actual for output tokens
                    raw_logprobs_for_output = record_batch["logprobs"][i].as_py()
                    prompt_len = len(input_tokens_list)
                    # Ensure logprobs list matches output_tokens_list length, pad if necessary (e.g. if logprobs were truncated)
                    padded_raw_logprobs = raw_logprobs_for_output + [0.0] * (len(output_tokens_list) - len(raw_logprobs_for_output))
                    
                    logprobs_list = [0.0] * prompt_len + padded_raw_logprobs[:len(output_tokens_list)]
                    logprobs = torch.tensor(logprobs_list, dtype=torch.float)

                    # Loss mask from Parquet corresponds to full_input_ids_list
                    loss_mask_list = record_batch["loss_mask"][i].as_py()
                    loss_mask = torch.tensor(loss_mask_list, dtype=torch.long)

                    # Advantages: scalar from Parquet, tile over loss_mask == 1 tokens
                    scalar_advantage = record_batch["advantages"][i].as_py()
                    advantages_per_token = torch.zeros(current_seq_len, dtype=torch.float)
                    # Ensure loss_mask has same length as full_input_ids before applying
                    if len(loss_mask) == current_seq_len:
                         advantages_per_token[loss_mask == 1] = scalar_advantage
                    else:
                        self._logger.warn(f"Step {self._step_count}, Sample {global_sample_idx_for_sharding-1}: Length mismatch between loss_mask ({len(loss_mask)}) and full_input_ids ({current_seq_len}). Adv. not applied per token.")
                        # Fallback: if PPO with per-sequence GAE, this might be okay if advantage is scalar per sequence.
                        # For now, let's assume per-token based on loss_mask intention.

                    yield {
                        "input_ids": full_input_ids,
                        "advantages": advantages_per_token,
                                "loss_mask": loss_mask,
                        "logprobs": logprobs,
                        "seq_lens": torch.tensor([current_seq_len], dtype=torch.long),
                        "rewards": torch.tensor([record_batch["rewards"][i].as_py()], dtype=torch.float),
                        "task_rewards": torch.tensor([record_batch["task_rewards"][i].as_py()], dtype=torch.float),
                        "length_penalties": torch.tensor([record_batch["length_penalties"][i].as_py()], dtype=torch.float),
                        "target_lengths": torch.tensor([record_batch["target_lengths"][i].as_py()], dtype=torch.long),
                    }
            global_sample_idx_for_sharding = 0 # Reset for next step/file iteration


def no_collate(batch: list[DatasetOutput]) -> list[DatasetOutput]:
    return batch


def get_dataloader(
    tokenizer,
    local_batch_size: int,
    batch_size: int,
    data_config: DataConfig,
    step_count_init: int,
) -> tuple[DataLoader[list[DatasetOutput]], GCPPrefetcher | None]:
    prefetcher = None
    path = data_config.path
    use_stable_file = False

    if "gs" in data_config.path:
        use_stable_file = True
        if get_world_info().rank == 0:
            prefetcher = GCPPrefetcher(data_config.path, data_config.local_dir)
        if dist.is_initialized():
            dist.barrier() 
        path = data_config.local_dir

    if data_config.fake:
        # vocab_size needed for FakeTokenizedDataset
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 30000 # Default if not found
        train_dataset = FakeTokenizedDataset(data_config.seq_length, vocab_size)
    else:
        train_dataset = ParquetDataset(
            Path(path),
            batch_size,
            data_config.timeout,
            step_count_init,
            data_config.ignore_zero_advantages,
            use_stable_file=use_stable_file,
        )

    loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        num_workers=data_config.num_workers,
        collate_fn=no_collate,
        pin_memory=True, 
        prefetch_factor=data_config.num_workers * 2 if data_config.num_workers > 0 else None 
    )
    return loader, prefetcher


class BatchOutput(TypedDict):
    # token level
    input_ids: Any  # Was: Int[torch.Tensor, "batch seq"]
    advantages: Any  # Was: Float[torch.Tensor, "batch seq"]
    loss_mask: Any  # Was: Int[torch.Tensor, "batch seq"]
    logprobs: Any  # Was: Float[torch.Tensor, "batch seq"]
    position_ids: Any  # Was: Int[torch.Tensor, "batch seq"]

    # sample level (these are now 1D tensors of size `batch` after collation)
    seq_lens: torch.Tensor
    rewards: torch.Tensor
    task_rewards: torch.Tensor
    length_penalties: torch.Tensor
    target_lengths: torch.Tensor


### collate

def collate_fn(samples: list[DatasetOutput], max_seq_len: int, pad_token_id: int) -> BatchOutput:
    """
    Collates a list of DatasetOutput samples into a single BatchOutput.
    If samples are to be packed (concatenated), they form one item in the batch dim [1, max_seq_len].
    If samples are to be padded and batched (e.g. for 'padding' or 'balancing' modes after individual processing),
    this function can be called for each sample to pad it, then results are merged.
    This version is tailored for the 'packing' scenario where `samples` make up ONE final packed sequence.
    """
    if not samples: 
        # Create an empty/dummy BatchOutput if samples list is empty
        # This might be needed by data_parallel_rebalancing
        dummy_shape_token = (1, max_seq_len)
        dummy_shape_sample = (0,) # No samples
        return {
            "input_ids": torch.full(dummy_shape_token, pad_token_id, dtype=torch.long),
            "advantages": torch.zeros(dummy_shape_token, dtype=torch.float),
            "loss_mask": torch.zeros(dummy_shape_token, dtype=torch.long),
            "logprobs": torch.zeros(dummy_shape_token, dtype=torch.float),
            "position_ids": torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0),
            "seq_lens": torch.empty(dummy_shape_sample, dtype=torch.long),
            "rewards": torch.empty(dummy_shape_sample, dtype=torch.float),
            "task_rewards": torch.empty(dummy_shape_sample, dtype=torch.float),
            "length_penalties": torch.empty(dummy_shape_sample, dtype=torch.float),
            "target_lengths": torch.empty(dummy_shape_sample, dtype=torch.long),
        }

    # Token-level data concatenation
    inputs_ids_list = [s["input_ids"] for s in samples]
    advantages_list = [s["advantages"] for s in samples]
    loss_masks_list = [s["loss_mask"] for s in samples]
    logprobs_list = [s["logprobs"] for s in samples]
    position_ids_list = [torch.arange(0, len(s["input_ids"]), dtype=torch.long) for s in samples]

    cat_input_ids = torch.cat(inputs_ids_list, dim=0)
    cat_advantages = torch.cat(advantages_list, dim=0)
    cat_loss_masks = torch.cat(loss_masks_list, dim=0)
    cat_logprobs = torch.cat(logprobs_list, dim=0)
    cat_position_ids = torch.cat(position_ids_list, dim=0)
    
    current_total_len = cat_input_ids.size(0)

    # Pad or truncate the concatenated sequence
    if current_total_len < max_seq_len:
        padding_len = max_seq_len - current_total_len
        pad_ids = torch.full((padding_len,), fill_value=pad_token_id, dtype=cat_input_ids.dtype)
        pad_advantages = torch.zeros(padding_len, dtype=cat_advantages.dtype)
        pad_loss_masks = torch.zeros(padding_len, dtype=cat_loss_masks.dtype)
        pad_logprobs = torch.zeros(padding_len, dtype=cat_logprobs.dtype) 
        pad_position_ids = torch.arange(padding_len, dtype=torch.long) # Simple padding for position_ids

        final_input_ids = torch.cat([cat_input_ids, pad_ids])
        final_advantages = torch.cat([cat_advantages, pad_advantages])
        final_loss_masks = torch.cat([cat_loss_masks, pad_loss_masks])
        final_logprobs = torch.cat([cat_logprobs, pad_logprobs])
        final_position_ids = torch.cat([cat_position_ids, pad_position_ids])
    else:
        final_input_ids = cat_input_ids[:max_seq_len]
        final_advantages = cat_advantages[:max_seq_len]
        final_loss_masks = cat_loss_masks[:max_seq_len]
        final_logprobs = cat_logprobs[:max_seq_len]
        final_position_ids = cat_position_ids[:max_seq_len]

    # Sample-level data: stack them to create batch dimension
    # These should already be 1D tensors of size [1] from DatasetOutput
    seq_lens_tensor = torch.cat([s["seq_lens"] for s in samples])
    rewards_tensor = torch.cat([s["rewards"] for s in samples])
    task_rewards_tensor = torch.cat([s["task_rewards"] for s in samples])
    length_penalties_tensor = torch.cat([s["length_penalties"] for s in samples])
    target_lengths_tensor = torch.cat([s["target_lengths"] for s in samples])

    return {
        "input_ids": final_input_ids.unsqueeze(0),       # [1, max_seq_len]
        "advantages": final_advantages.unsqueeze(0),   # [1, max_seq_len]
        "loss_mask": final_loss_masks.unsqueeze(0),      # [1, max_seq_len]
        "logprobs": final_logprobs.unsqueeze(0),         # [1, max_seq_len]
        "position_ids": final_position_ids.unsqueeze(0), # [1, max_seq_len]
        
        "seq_lens": seq_lens_tensor,                   # [num_samples_in_packed_sequence]
        "rewards": rewards_tensor,                     # [num_samples_in_packed_sequence]
        "task_rewards": task_rewards_tensor,           # [num_samples_in_packed_sequence]
        "length_penalties": length_penalties_tensor,   # [num_samples_in_packed_sequence]
        "target_lengths": target_lengths_tensor,       # [num_samples_in_packed_sequence]
    }


### sequence packing

def pack_datatset_outputs_efficiently(batch_optim: list[DatasetOutput], max_seq_len: int) -> list[list[DatasetOutput]]:
    """
    Packs dataset outputs into bins, where each bin's total sequence length (sum of sample lengths)
    does not exceed max_seq_len. Samples are sorted by length (descending) first.
    Returns a list of bins, where each bin is a list of DatasetOutput samples.
    """
    if not batch_optim: return []

    batch_with_len = [(len(sample["input_ids"]), sample) for sample in batch_optim]
    sorted_batch = sorted(batch_with_len, key=lambda x: x[0], reverse=True)

    bins: list[list[DatasetOutput]] = []
    for seq_len, sample in sorted_batch:
        bin_found = False
        for current_bin in bins:
            current_bin_len = sum(len(s["input_ids"]) for s in current_bin)
            if current_bin_len + seq_len <= max_seq_len:
                current_bin.append(sample)
                bin_found = True
                break
        if not bin_found:
            bins.append([sample])
    return bins


def data_parallel_rebalancing(micro_batches: list[BatchOutput]) -> list[BatchOutput]:
    """
    Ensures all ranks have the same number of micro-batches by padding with duplicates
    of the first micro-batch (or empty ones if the first is also empty).
    """
    num_grad_acc_steps = len(micro_batches)

    max_grad_acc_step = num_grad_acc_steps
    if dist.is_initialized():
        # Ensure tensor is on the correct device if using CUDA
        device = get_world_info().device 
        max_grad_acc_step_tensor = torch.tensor(num_grad_acc_steps, dtype=torch.int32, device=device)
        dist.all_reduce(max_grad_acc_step_tensor, op=dist.ReduceOp.MAX, group=None)
        max_grad_acc_step = int(max_grad_acc_step_tensor.item())
    
    if max_grad_acc_step == 0 and num_grad_acc_steps == 0: # All ranks have no data
        return []
    
    # If this rank has no batches, but others do, it needs to create empty/dummy ones.
    if num_grad_acc_steps == 0 and max_grad_acc_step > 0:
        # Create a dummy BatchOutput. Relies on collate_fn to produce a valid structure.
        # Need max_seq_len and pad_token_id if collate_fn([]) is called.
        # This is a bit problematic as these aren't directly available here.
        # Assuming some defaults or that an empty BatchOutput can be structured manually.
        # For simplicity, let's assume if a rank has 0 micro_batches, it will append pre-defined empty ones.
        # The logic below handles padding if this rank *has* batches but fewer than max.
        # A better approach might be for collate_fn to handle an empty list and return a valid empty BatchOutput.
        # (Updated collate_fn to handle empty list)
        dummy_batch = collate_fn([], max_seq_len=1, pad_token_id=0) # Placeholder max_seq_len and pad_id
        for _ in range(max_grad_acc_step):
            micro_batches.append(dummy_batch) # This dummy_batch might need actual seq_len from config
        return micro_batches

    empty_batch_count = max_grad_acc_step - num_grad_acc_steps
    if empty_batch_count > 0:
        # Use a copy of the first actual micro-batch for padding
        # Ensure micro_batches[0] is valid and not an empty/dummy one itself if this rank had data.
        padding_batch = micro_batches[0] 
    for _ in range(empty_batch_count):
            micro_batches.append(padding_batch) # Appending reference, could deepcopy if modification is a concern

    return micro_batches


def packed_batch_packing(batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int) -> list[BatchOutput]:
    """
    Packs samples into micro-batches. Each micro-batch is a single sequence of `max_seq_len`
    formed by concatenating multiple original samples. `micro_bs` is not directly used here for num samples
    as packing prioritizes filling `max_seq_len`.
    """
    if not batch_optim: return []

    # `pack_datatset_outputs_efficiently` creates bins of samples, each bin's total length <= max_seq_len.
    # Each bin will become one micro-batch.
    packed_sample_bins = pack_datatset_outputs_efficiently(batch_optim, max_seq_len=max_seq_len)

    micro_batches = [collate_fn(bin_of_samples, max_seq_len=max_seq_len, pad_token_id=pad_token_id) 
                     for bin_of_samples in packed_sample_bins if bin_of_samples]

    return data_parallel_rebalancing(micro_batches)


def merge_batches_padding(
    individual_padded_batches: list[BatchOutput], 
    # max_seq_len_for_this_batch: int, # This was from balancing, not general padding
    # pad_token_id: int # Not needed here as already padded
) -> BatchOutput:
    """
    Merges a list of individually processed BatchOutput items (assumed to be [1, seq_len_padded])
    into a single BatchOutput by concatenating along the batch dimension (dim 0).
    Output tensors will have shape [len(individual_padded_batches), seq_len_padded].
    """
    if not individual_padded_batches: 
        # Should not happen if called correctly, but handle defensively.
        # Return a structure consistent with BatchOutput but with batch_size 0.
        # This requires knowing the seq_len they *would* have been padded to.
        # For now, assume this case is avoided by callers.
        raise ValueError("merge_batches_padding received an empty list.")

    # Token level: cat along batch dim (dim 0)
    # All input BatchOutput items should have the same seq_len due to prior padding.
    final_input_ids = torch.cat([b["input_ids"] for b in individual_padded_batches], dim=0)
    final_advantages = torch.cat([b["advantages"] for b in individual_padded_batches], dim=0)
    final_loss_mask = torch.cat([b["loss_mask"] for b in individual_padded_batches], dim=0)
    final_logprobs = torch.cat([b["logprobs"] for b in individual_padded_batches], dim=0)
    final_position_ids = torch.cat([b["position_ids"] for b in individual_padded_batches], dim=0)

    # Sample level: also cat along batch dim (dim 0), as each input BatchOutput had batch_size 1.
    final_rewards = torch.cat([b["rewards"] for b in individual_padded_batches], dim=0)
    final_seq_lens = torch.cat([b["seq_lens"] for b in individual_padded_batches], dim=0)
    final_task_rewards = torch.cat([b["task_rewards"] for b in individual_padded_batches], dim=0)
    final_length_penalties = torch.cat([b["length_penalties"] for b in individual_padded_batches], dim=0)
    final_target_lengths = torch.cat([b["target_lengths"] for b in individual_padded_batches], dim=0)
    
    return {
        "input_ids": final_input_ids,
        "advantages": final_advantages,
        "loss_mask": final_loss_mask,
        "logprobs": final_logprobs,
        "position_ids": final_position_ids,
        "rewards": final_rewards,
        "seq_lens": final_seq_lens,
        "task_rewards": final_task_rewards,
        "length_penalties": final_length_penalties,
        "target_lengths": final_target_lengths,
    }


def packed_batch_padding(batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int) -> list[BatchOutput]:
    """
    Pads each sample to `max_seq_len`, then groups `micro_bs` of these padded samples 
    into a single micro-batch. Output tensors have shape [micro_bs, max_seq_len].
    """
    if not batch_optim: return []
    
    # 1. Collate each sample individually, padding it to max_seq_len.
    # Each `collate_fn` output here will have token tensors of shape [1, max_seq_len]
    # and sample-level tensors of shape [1].
    padded_individual_samples = [collate_fn([sample], max_seq_len, pad_token_id) for sample in batch_optim]

    micro_batches = []
    for i in range(0, len(padded_individual_samples), micro_bs):
        current_micro_batch_padded_samples = padded_individual_samples[i : i + micro_bs]
        if not current_micro_batch_padded_samples: continue

        # 2. Merge these `micro_bs` (or fewer for the last batch) padded samples.
        # `merge_batches_padding` will cat along dim 0.
        merged_micro_batch = merge_batches_padding(current_micro_batch_padded_samples)
        micro_batches.append(merged_micro_batch)
        
    return data_parallel_rebalancing(micro_batches)


### balancing

def pack_datatset_outputs_balancing(
    batch_optim: list[DatasetOutput], 
    max_seq_len_global_config: int, # Global max sequence length constraint
    micro_bs: int # Target number of samples per micro-batch
) -> list[tuple[list[DatasetOutput], int]]: # Returns list of (samples_for_bin, actual_max_len_for_this_bin)
    """
    Packs samples into bins for micro-batches, aiming for `micro_bs` samples per bin.
    The actual max length for each bin is determined by the longest sequence in that bin,
    capped by `max_seq_len_global_config`.
    """
    if not batch_optim: return []

    # Sort by sequence length (ascending) to group similar lengths, potentially improving padding efficiency.
    batch_with_len = [(len(sample["input_ids"]), sample) for sample in batch_optim]
    sorted_batch = sorted(batch_with_len, key=lambda x: x[0])

    bins_and_their_max_len: list[tuple[list[DatasetOutput], int]] = []    
    current_bin_samples: list[DatasetOutput] = []

    for seq_len_of_sample, sample in sorted_batch:
        # Ensure individual sample doesn't exceed global max length
        if seq_len_of_sample > max_seq_len_global_config:
            self._logger.warn(f"Sample with length {seq_len_of_sample} exceeds max_seq_len_global_config {max_seq_len_global_config}. Skipping.")
            continue # Or truncate sample if preferred

        if len(current_bin_samples) < micro_bs:
            current_bin_samples.append(sample)
        else:
            # Current bin is full (reached micro_bs samples), finalize it.
            actual_max_len_for_this_bin = 0
            if current_bin_samples: # Should always be true here
                 actual_max_len_for_this_bin = max(len(s["input_ids"]) for s in current_bin_samples)
            bins_and_their_max_len.append((current_bin_samples, actual_max_len_for_this_bin))
            
            # Start a new bin with the current sample
            current_bin_samples = [sample]
    
    # Add the last partially filled or full bin
    if current_bin_samples:
        actual_max_len_for_this_bin = max(len(s["input_ids"]) for s in current_bin_samples)
        bins_and_their_max_len.append((current_bin_samples, actual_max_len_for_this_bin))

    return bins_and_their_max_len


def packed_batch_balancing(
    batch_optim: list[DatasetOutput], 
    max_seq_len_global_config: int, 
    pad_token_id: int, 
    micro_bs: int
) -> list[BatchOutput]:
    """
    Creates micro-batches by balancing sequence lengths. 
    Each micro-batch will contain up to `micro_bs` samples, padded to the 
    maximum sequence length within that specific micro-batch (but not exceeding `max_seq_len_global_config`).
    """
    if not batch_optim: return []

    # bins_info is a list of (list_of_samples_for_bin, actual_max_len_for_this_bin)
    bins_info = pack_datatset_outputs_balancing(batch_optim, max_seq_len_global_config, micro_bs)

    micro_batches = []
    for samples_for_bin, actual_max_len_for_this_bin in bins_info:
        if not samples_for_bin: continue
        
        # 1. Collate each sample in the current bin individually, padding to `actual_max_len_for_this_bin`.
        padded_samples_in_bin = [
            collate_fn([s], actual_max_len_for_this_bin, pad_token_id) for s in samples_for_bin
        ]
        
        # 2. Merge these individually padded samples to form the micro-batch.
        # Tensors will be [len(samples_for_bin), actual_max_len_for_this_bin].
        if padded_samples_in_bin:
            merged_micro_batch = merge_batches_padding(padded_samples_in_bin)
            micro_batches.append(merged_micro_batch)

    return data_parallel_rebalancing(micro_batches)


###########


CollateMode: TypeAlias = Literal["packing", "padding", "balancing"]


def packed_batch(
    batch_optim: list[DatasetOutput], 
    max_seq_len: int,           # For 'packing' & 'padding', this is THE target seq_len.
                                # For 'balancing', this is a global upper limit.
    pad_token_id: int, 
    micro_bs: int,              # For 'padding' & 'balancing', num samples per micro-batch.
                                # Less direct role in 'packing'.
    collate_mode: CollateMode
) -> list[BatchOutput]:
    """
    Takes a list of DatasetOutput samples (usually a local batch from DataLoader)
    and prepares a list of BatchOutput micro-batches according to the collate_mode.
    """

    match collate_mode:
        case "packing":
            # Output: list of BatchOutput, each with token tensors of shape [1, max_seq_len]
            return packed_batch_packing(batch_optim, max_seq_len, pad_token_id, micro_bs)
        case "padding":
            # Output: list of BatchOutput, each with token tensors of shape [micro_bs, max_seq_len]
            return packed_batch_padding(batch_optim, max_seq_len, pad_token_id, micro_bs)
        case "balancing":
            # Output: list of BatchOutput, each with token tensors of shape [num_samples_in_bin, actual_max_len_for_bin]
            # where num_samples_in_bin <= micro_bs and actual_max_len_for_bin <= max_seq_len.
            return packed_batch_balancing(batch_optim, max_seq_len, pad_token_id, micro_bs)
        case _:
            raise ValueError(f"Invalid collate mode: {collate_mode}")
