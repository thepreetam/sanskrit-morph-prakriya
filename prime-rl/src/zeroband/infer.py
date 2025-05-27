import json
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from zeroband.inference import envs
from zeroband.inference.config import Config
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import setup_pipeline
from zeroband.inference.rewards import compute_rewards
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.utils import fake_chat_template, generate_target_length_prompts, reload_model_weights
from zeroband.training.mp import EnvWrapper
from zeroband.utils.logger import get_logger
from zeroband.utils.metrics import PrimeMetric

# Global logger
logger = get_logger("INFER")


def inference(config: Config):
    # Initialize the logger
    logger.info("Starting inference (DRY RUN MODE)")
    logger.info(f"TP={config.tp}, DP={config.dp}, PP={config.pp.world_size}")

    if hasattr(config, 'task_type') and config.task_type == "sanskrit_morph":
        logger.info("Running Sanskrit Morphological Rendering task (DRY RUN)")

    if config.clean_output_path and config.output_path is not None:
        logger.info(f"Cleaning output path {config.output_path}")
        shutil.rmtree(config.output_path, ignore_errors=True)

    # Initialize prime metrics
    prime_metric = PrimeMetric(disable=config.prime_log_freq is None, period=config.prime_log_freq)

    # Initialize vLLM and get tokenizer
    logger.info("Initializing vLLM (SKIPPED IN DRY RUN)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    except:
        logger.warning(f"Could not load tokenizer for {config.model_name}, using gpt2 as fallback for DRY RUN.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create communication for pipeline
    # if config.pp.world_size > 1: # This depends on llm object
    #     setup_pipeline(
    #         llm=llm, # llm is not defined
    #         rank=config.pp.rank,
    #         world_size=config.pp.world_size,
    #         iroh_seed=config.pp.iroh_seed,
    #         iroh_peer_id=config.pp.iroh_peer_id,
    #     )

    # Load  dataset
    logger.info(f"Loading dataset {config.dataset}")
    # Conditional dataset loading based on task_type for Sanskrit task
    if hasattr(config, 'task_type') and config.task_type == "sanskrit_morph":
        dataset = load_dataset(config.dataset, split="train")
    else:
    dataset = load_dataset(config.dataset, split="train")

    if envs.NODE_ADDRESS is not None:
        assert config.seed is None, "Seed is not supported when NODE_ADDRESS is set"
        assert envs.RANK == 0, "DP is not supported when NODE_ADDRESS is set"
        node_address_int = int(envs.NODE_ADDRESS, 16)
        logger.info(f"Seeding with {node_address_int} ({envs.NODE_ADDRESS})")
    else:
        seed = config.seed + envs.RANK if config.seed is not None else None
        generator = np.random.default_rng(seed)
        # Ensure dataset is shuffled reproducibly if not using NODE_ADDRESS
        if not (hasattr(config, 'task_type') and config.task_type == "sanskrit_morph" and node_address_int is None):
             dataset = dataset.shuffle(generator=generator) # Avoid reshuffling if already handled by node_address logic or if sanskrit + no node address
        node_address_int = None


    if config.difficulty_filtering:
        dataset = dataset.filter(
            lambda x: x[config.difficulty_filtering.solve_rate_field] >= config.difficulty_filtering.min_solve_rate
            and x[config.difficulty_filtering.solve_rate_field] <= config.difficulty_filtering.max_solve_rate
        )

    # Setup TOPLOC (SKIPPED IN DRY RUN as it depends on llm internal)
    # toploc_cache, _ = setup_toploc_cache(
    #     llm, # llm not defined
    #     disable=not config.toploc,
    #     max_seqs=config.batch_size * config.sampling.n,
    #     hidden_size=llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size, # llm not defined
    # )

    # if config.ckpt_start_path is not None: # Depends on llm
    #     path = Path(config.ckpt_start_path)
    #     path_file = path / "model.safetensors"
    #     if not path_file.exists():
    #         raise FileNotFoundError(f"Checkpoint file {path_file} does not exist")
    #     ckpt_step = int(path.name.split("_")[-1])
    #     logger.info(f"Resuming from step {ckpt_step} at {path_file}")
    #     # llm = reload_model_weights(llm, path_file) # llm not defined
    #     real_step = ckpt_step
    # else:
    ckpt_step = 0 # Placeholder
    real_step = 0 # Placeholder


    current_step_batch_counter = 1
    total_problems = 0
    total_tokens = 0
    max_samples = config.max_samples or len(dataset)

    for i in range(0, min(len(dataset), max_samples), config.batch_size):
        if config.step_endpoint is not None:
            try:
                new_real_step = requests.get(config.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_real_step != real_step:
                real_step = new_real_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        # Model reloading logic (SKIPPED IN DRY RUN)
        # logger.info(
        #     f"real_step: {real_step}, ckpt_step: {ckpt_step}, real_step - ckpt_step: {real_step - ckpt_step}, config.async_level: {config.async_level}"
        # )
        # if config.rollout_path is not None and real_step - ckpt_step > config.async_level:
        #     ckpt_step = real_step - config.async_level
        #     attempt_count = 0
        #     while True:
        #         stable_file = Path(config.rollout_path) / f"step_{ckpt_step}/stable"
        #         if stable_file.exists():
        #             logger.info(f"Reloading model weights from {config.rollout_path} ckpt {ckpt_step} (SKIPPED IN DRY RUN)")
        #             # llm = reload_model_weights(llm, Path(config.rollout_path) / f"step_{ckpt_step}/model.safetensors") # llm not defined
        #             total_problems = 0
        #             total_tokens = 0
        #             logger.info(f"Reloaded model weights from {config.rollout_path} ckpt {ckpt_step} (SKIPPED IN DRY RUN)")
        #             break
        #         if attempt_count % 30 == 0:
        #             logger.info(f"No stable file found at {stable_file}, waiting for new checkpoint (SKIPPED IN DRY RUN)")
        #         time.sleep(1)
        #         attempt_count += 1

        if node_address_int is not None:
            generator = np.random.default_rng(node_address_int * current_step_batch_counter + real_step)
            indexes = generator.integers(0, len(dataset), config.batch_size)
            batch = dataset.select(indexes)
        else:
            # Ensure 'i' doesn't exceed dataset bounds after potential shuffling
            current_batch_size = min(config.batch_size, len(dataset) - i)
            if current_batch_size <=0: break # No more data
            batch = dataset.select(range(i, i + current_batch_size))

        if hasattr(config, 'task_type') and config.task_type == "sanskrit_morph":
            prompts_content = [item["input_metadata_str"] for item in batch]
            messages = [[{"role": "user", "content": p}] for p in prompts_content]
            verification_infos = [
                {
                    "metadata": item["input_metadata_str"],
                    "expected_form": item["surface_form_vidyut"],
                    "target_length": config.sampling.get("max_tokens", 10) # Using 10 for dry run
                }
                for item in batch
            ]
            task_types = ["sanskrit_morph"] * len(batch)
            length_prompt_additions, target_lengths = [], [info["target_length"] for info in verification_infos]
        else:
            # Existing logic for other tasks
        messages = [[{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": "<think>\n"}] for item in batch]
        length_prompt_additions, target_lengths = generate_target_length_prompts(config.len_reward, len(batch))
        verification_infos = [json.loads(item["verification_info"]) for item in batch]
        for target_length, verification_info in zip(target_lengths, verification_infos):
            verification_info["target_length"] = target_length
        task_types = [item["task_type"] for item in batch]

        if config.len_reward:
            if config.len_reward.length_prompt_location == "system_prompt":
                messages = [
                    [
                        {"role": "system", "content": length_prompt},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": "<think>\n"},
                    ]
                    for item, length_prompt in zip(batch, length_prompt_additions)
                ]
            else:
                messages = [
                    [{"role": "user", "content": item["prompt"] + length_prompt}, {"role": "assistant", "content": "<think>\n"}]
                for item, length_prompt in zip(batch, length_prompt_additions)
            ]

        if hasattr(config, 'task_type') and config.task_type == "sanskrit_morph":
            prompts = [m[0]["content"] for m in messages]
        elif tokenizer.chat_template: # tokenizer might be a placeholder
            try:
                prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # continue_final_message=True for some models
            except Exception as e:
                logger.warning(f"Tokenizer could not apply chat template: {e}. Using fake_chat_template as fallback for DRY RUN.")
                prompts = fake_chat_template(messages) # Fallback if chat template fails
        else:
            prompts = fake_chat_template(messages)


        logger.info("DRY RUN: SKIPPING ACTUAL MODEL GENERATION")
        start_time = time.time()
        # request_outputs = llm.generate(prompts, sampling_params, use_tqdm=False) # SKIPPED
        
        # Create DUMMY request_outputs for DRY RUN
        dummy_request_outputs = []
        for idx, prompt_text in enumerate(prompts):
            # For Sanskrit task, use expected_form as dummy output, or a fixed string
            # For other tasks, use a generic dummy string.
            dummy_generated_text = "रामः" # Generic dummy for non-Sanskrit or if expected_form is missing
            if hasattr(config, 'task_type') and config.task_type == "sanskrit_morph":
                dummy_generated_text = verification_infos[idx].get("expected_form", "त्रुटिः") # Default to "error" if not found

            # Dummy tokenization (very basic)
            dummy_prompt_tokens = tokenizer.encode(prompt_text)
            dummy_output_tokens = tokenizer.encode(dummy_generated_text)
            
            # Create a structure that mimics vllm.outputs.RequestOutput
            # and vllm.outputs.CompletionOutput
            class DummyCompletionOutput:
                def __init__(self, text, token_ids, logprobs_val=0.0):
                    self.text = text
                    self.token_ids = token_ids
                    # For dry run, logprobs can be simplified or dummied
                    self.logprobs = [[(tid, logprobs_val)] for tid in token_ids] # List of lists of (token_id, logprob)
                    self.cumulative_logprob = logprobs_val * len(token_ids) if token_ids else 0.0
                    self.finish_reason = "length" # or "stop"

            class DummyRequestOutput:
                def __init__(self, request_id, prompt, prompt_token_ids, outputs):
                    self.request_id = request_id
                    self.prompt = prompt
                    self.prompt_token_ids = prompt_token_ids
                    self.outputs = outputs # List of DummyCompletionOutput

            dummy_completion = DummyCompletionOutput(
                text=dummy_generated_text,
                token_ids=dummy_output_tokens,
                logprobs_val=-0.1 # Arbitrary dummy logprob
            )
            dummy_request_outputs.append(
                DummyRequestOutput(
                    request_id=f"dry_run_req_{i}_{idx}",
                    prompt=prompt_text,
                    prompt_token_ids=dummy_prompt_tokens,
                    outputs=[dummy_completion] # n=1 for simplicity
                )
            )
        request_outputs = dummy_request_outputs
        end_time = time.time()

        if not request_outputs: # Should not happen with dummy data unless prompts is empty
             logger.warning("No request_outputs generated/dummied. Skipping batch.")
             continue


        # toploc_cache.maybe_generate_proofs_in_background(force_generate=True) # SKIPPED

        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_total_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_total_tokens
        avg_seq_length = batch_total_tokens / (len(request_outputs) * config.sampling.n) if request_outputs and config.sampling.n > 0 else 0


        elapsed_time = end_time - start_time
        tokens_per_second = batch_total_tokens / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"Batch throughput (DRY RUN): {tokens_per_second:.2f} tok/sec ({batch_total_tokens} tokens in {elapsed_time:.2f}s, avg seq len: {avg_seq_length:.1f})"
        )

        # toploc_cache.wait_for_proofs() # SKIPPED
        # proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])] # SKIPPED
        # toploc_cache.reset_cache() # SKIPPED
        proofs = [b"dummy_proof"] * len(request_outputs) # Dummy proofs

        start = time.time()
        current_len_reward_config = config.len_reward if hasattr(config, 'len_reward') else None
        request_rewards = compute_rewards(request_outputs, verification_infos, task_types, current_len_reward_config)
        logger.info(f"Computed rewards and advantages in {time.time() - start:.2f}s (DRY RUN)")

        table = get_parquet_table(
            request_outputs,
            request_rewards,
            proofs,
            ckpt_step,
            target_lengths,
        )

        step_path = Path(config.output_path) / f"step_{real_step}"
        os.makedirs(step_path, exist_ok=True)
        pq_save_path = f"{step_path}/{uuid.uuid4()}.parquet"
        pq.write_table(table, pq_save_path)
        file_sha = sha256sum(pq_save_path)
        prime_metric.log_prime({"file_sha": file_sha, "file_name": pq_save_path})
        logger.info(f"✨ Saved {len(proofs)} samples to {pq_save_path} with sha {file_sha or 'NA'} (DRY RUN)")

        total_problems += len(prompts)
        metric = {"dashbord-progress/total": total_problems, f"dashbord-progress/{config.dataset}": total_tokens}
        prime_metric.log_prime(metric)

        logger.info(f"Generated {total_problems} problems for step {real_step} (DRY RUN)")
        real_step += 1

        if config.total_step is not None and real_step > config.total_step:
            logger.info(f"Reached total step {config.total_step}, stopping inference (DRY RUN)")
            break

    # Manually destroy vLLM process group to avoid warnings (SKIPPED for DRY RUN)
    # dist.destroy_process_group() # This would error if not initialized


def main(config: Config) -> list[mp.Process]:
    processes = []
    from zeroband.inference import envs as inference_envs

    if config.dp > 1:
        if config.tp == "auto":
            # This logic assumes CUDA, adjust if on CPU for dry run
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            assert num_devices % config.dp == 0, "Number of devices must be divisible by DP"
            config.tp = num_devices // config.dp
        
        gpu_ids = inference_envs.CUDA_VISIBLE_DEVICES if torch.cuda.is_available() else [str(x) for x in range(num_devices)] # type: ignore
        # Ensure gpu_ids is a list of strings if from env, or list of ints if generated for CPU
        if isinstance(gpu_ids, str): gpu_ids = gpu_ids.split(',')
        
        # This part needs careful handling if config.tp is int and gpu_ids are strings from env
        # For dry run on CPU, tp is usually 1, so gpu_ids_per_rank = [[0], [1] ...]
        # If on GPU, CUDA_VISIBLE_DEVICES is a string "0,1,2,3"
        
        effective_tp = config.tp if isinstance(config.tp, int) else 1 # Fallback for "auto" on CPU
        
        # Simplified logic for device distribution for dry run
        # Assuming each DP rank gets one "device" (can be conceptual for CPU)
        num_logical_devices_total = config.dp * effective_tp
        all_device_indices = list(map(str, range(num_logical_devices_total))) # e.g. ["0", "1", "2", "3"] if dp=2, tp=2
        
        gpu_ids_per_rank = [
            all_device_indices[i : i + effective_tp] for i in range(0, len(all_device_indices), effective_tp)
        ]


        for rank, rank_gpu_ids in enumerate(gpu_ids_per_rank):
            current_cuda_visible_devices = ",".join(rank_gpu_ids)
            # For dry run, CUDA_VISIBLE_DEVICES might not be relevant if not using GPUs
            # but EnvWrapper expects it
            envs_for_proc = {"CUDA_VISIBLE_DEVICES": current_cuda_visible_devices, "RANK": str(rank), "LOCAL_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, envs_for_proc), args=(config,))
            processes.append(process)
    else: # Single process run
        if config.tp == "auto": # type: ignore
            config.tp = torch.cuda.device_count() if torch.cuda.is_available() else 1 # type: ignore
        inference(config)


    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) # Added force=True
    # config = Config(**parse_argv())  # type: ignore # Already handled by user

    # Temporarily define a minimal config for dry run if parse_argv is an issue
    # This is a fallback - ideally parse_argv should work
    try:
        from pydantic_config import parse_argv
        config = Config(**parse_argv())
    except ImportError:
        logger.error("pydantic_config not found. Using a fallback minimal config for DRY RUN. Please ensure pydantic_config is installed.")
        class DummySamplingConfig:
            n: int = 1
            max_tokens: int = 10
            def model_dump(self): return {"n": self.n, "max_tokens": self.max_tokens}
        
        class DummyLenRewardConfig: # Add dummy if needed by generate_target_length_prompts
            pass


        class MinimalConfigForDryRun:
            model_name: str = "gpt2" # Fallback model
            tp: int | str = 1
            dp: int = 1
            pp = type('PP', (), {'world_size': 1, 'rank':0, 'iroh_seed':None, 'iroh_peer_id':None})()
            max_model_len: int = 512
            quant: str | None = None
            enforce_eager: bool = False
            download_dir: str | None = None
            dtype: str = "fp32"
            sampling = DummySamplingConfig()
            dataset: str = "preetammukherjee/sanskrit_morph_prakriya" # Default to sanskrit dataset
            task_type: str = "sanskrit_morph"
            output_path: str = "output_dry_run"
            clean_output_path: bool = True
            prime_log_freq: int | None = None
            seed: int | None = 42
            difficulty_filtering = None # type: ignore
            len_reward = None # type: ignore
            toploc: bool = False
            batch_size: int = 4
            ckpt_start_path: str | None = None
            async_level: int = 0 # type: ignore
            rollout_path: str | None = None
            step_endpoint: str | None = None
            total_step: int | None = 1 # Run for 1 step
            max_samples: int | None = 8 # Process 8 samples
            
        config = MinimalConfigForDryRun() # type: ignore
        # Manually set some fields if not in MinimalConfigForDryRun but used directly
        config.sampling.n = 1 # Ensure n is set
        config.sampling.max_tokens = 10 # Ensure max_tokens for sanskrit dry run

    if config.step_endpoint is not None:
        current_step = requests.get(config.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    from zeroband.inference import envs as inference_envs
    if inference_envs.SHARDCAST_SERVERS is not None:
        from zeroband.inference.shardcast_downloader import run_main_bg
        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rollout_path,
            config.async_level + 1,
            max(current_step - config.async_level, 1) if 'current_step' in locals() else 1,
        )
    else:
        shardcast_process = None

    try:
        main(config)
    finally:
        if shardcast_process is not None:
            import os
            import signal
            if shardcast_process.pid is not None: # Check if PID exists
            os.kill(shardcast_process.pid, signal.SIGKILL)
            shardcast_process.join()
