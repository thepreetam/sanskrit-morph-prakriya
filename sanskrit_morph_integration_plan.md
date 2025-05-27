# Engineering Plan: Integrating Sanskrit Morphological Rendering into `prime-rl`

**Project Goal:** To enable training a language model within the `prime-rl` framework to perform Sanskrit morphological rendering (metadata -> surface form), using `Vidyut-Prakriya` for verification and reward calculation.

**Core `prime-rl` Architecture:** The integration will follow `prime-rl`'s existing decoupled architecture:
1.  **Inference Worker (`infer.py` / `infer_sanskrit.py`):** Loads prompts (Sanskrit metadata), generates surface forms using the current model policy, calculates rewards using the Sanskrit verifier, and writes experience data (prompts, generations, rewards, advantages, policy logprobs) to Parquet files in a structured directory (`output_path/step_{n}/`).
2.  **Training Worker (`train.py`):** Loads these Parquet files, re-computes logprobs for the generated actions under its current model, and uses these along with the loaded rewards/advantages to update the model via the GPRO loss.

## Phase 1: Code Implementation & Modification

*(Files are relative to the root of the `prime-rl` repository)*

1.  **Configuration (`.toml` files & Python config classes):**
    *   **Create `configs/inference/sanskrit_morph.toml`:**
        *   Define a new `task_type: "sanskrit_morph"`.
        *   Specify dataset path: `preetammukherjee/sanskrit_morph_prakriya`.
        *   Define how "prompt" (from `input_metadata_str`) and "verification_info" (containing `surface_form_vidyut` and original `input_metadata_str`) are to be extracted from the dataset.
        *   Set `output_path` for Parquet files (e.g., `data/sanskrit_morph_experience/`).
        *   Configure sampling parameters as needed.
    *   **Create `configs/training/sanskrit_morph.toml`:**
        *   Point `data.path` to the `output_path` defined in the inference config.
        *   Configure model (e.g., base model to fine-tune), optimizer, learning rate, batch sizes, etc.
    *   **(Optional) Modify `src/zeroband/inference/config.py` and `src/zeroband/training/config.py`:** If necessary, add new fields to the `Config` classes to support specific needs of the Sanskrit task (e.g., specific fields for loading your dataset).

2.  **Sanskrit Verifier Integration:**
    *   **Add Verifier Code:**
        *   Create a new directory, e.g., `src/zeroband/custom_tasks/sanskrit_morph/`.
        *   Place your `verifier.py` (containing `verify_form(metadata, generated_text)`) and any helper scripts into this directory.
    *   **Dependency Management (`pyproject.toml`):**
        *   Add `vidyut` (the Python package for Vidyut-Prakriya) to the `[project.dependencies]` section in `pyproject.toml`. Ensure it's a version compatible with Linux environments where `prime-rl` will be seriously run.

3.  **Inference Side Modifications (`src/zeroband/inference/`):**
    *   **Modify `rewards.py` (`compute_rewards` function):**
        *   Add a new conditional block: `if task_type == "sanskrit_morph":`.
        *   Inside this block:
            *   For each item in `request_outputs` (model generations) and its corresponding `verification_info`:
                *   Detokenize the generated `token_ids` to get the surface Sanskrit text.
                *   Extract `expected_form` and original `metadata` from `verification_info`.
                *   Call `sanskrit_verifier.verify_form(metadata, generated_sanskrit_text)` (importing from your verifier path).
                *   Store the returned reward.
            *   Structure the output to be compatible with what `get_parquet_table` expects for `request_rewards` (this dictionary should contain the computed rewards and potentially be used to derive/store advantages).
    *   **Adapt `infer.py` (or create `infer_sanskrit.py` if changes are substantial):**
        *   **Dataset Loading:** When `config.task_type == "sanskrit_morph"`, load `preetammukherjee/sanskrit_morph_prakriya`.
        *   **Prompt Preparation:** Use `input_metadata_str` as the "prompt".
        *   **`verification_infos` Construction:** For each prompt, create a `verification_info` dictionary containing `{"expected_form": item["surface_form_vidyut"], "metadata": item["input_metadata_str"]}`. This will be passed to `compute_rewards`.
        *   **`task_types`:** Pass `task_types = ["sanskrit_morph"] * len(batch)` to `compute_rewards`.
    *   **Modify `parquet.py` (`get_parquet_table` function):**
        *   Ensure it correctly processes data when the rewards/info come from the Sanskrit task.
        *   **Key Columns for Parquet Output:**
            *   `input_tokens`: Tokenized `input_metadata_str`.
            *   `output_tokens`: Tokenized generated Sanskrit surface form.
            *   `rewards`: Scalar reward from the verifier.
            *   `task_rewards`: Scalar reward from the verifier (can be same as `rewards`).
            *   `advantages`: Must be computed (e.g., by `compute_rewards` or here, possibly simple reward-to-go if no value function used at inference).
            *   `logprobs`: Log probabilities of `output_tokens` under the inference policy (from `request_outputs[i].outputs[0].logprobs`).
            *   `loss_mask`: Array of 0s for `input_tokens` and 1s for `output_tokens`.
            *   Populate other schema fields (e.g., `length_penalties`, `target_lengths`) with sensible defaults or values derived from the Sanskrit task.

4.  **Training Side Modifications (`src/zeroband/training/`):**
    *   **`data.py` (`ParquetDataset` and `collate_fn`):**
        *   Verify that `ParquetDataset` correctly reads the Parquet files generated by the modified inference side.
        *   Ensure that when it yields `DatasetOutput` dicts:
            *   `input_ids` is the concatenation of `input_tokens` (metadata) and `output_tokens` (generated form) from the Parquet file.
            *   `loss_mask` is correctly loaded/reconstructed to apply only to the `output_tokens` part of `input_ids`.
            *   `rewards`, `advantages`, and `original_logprobs` (which were `logprobs` in the Parquet file) are correctly passed through.
        *   No major changes should be needed here if the Parquet files are correctly formatted by the inference side.

## Phase 2: Local Simulation & Testing (macOS Focus)

*Goal: Test data flow and core logic without running full `vLLM` pipeline.*

1.  **Unit Test Core Components:**
    *   Write Python tests for your modified `compute_rewards` in `rewards.py`. Simulate `request_outputs` and `verification_infos`, and assert that your verifier is called and correct rewards are produced.
    *   Write Python tests for your modified `get_parquet_table` in `parquet.py`. Feed it simulated `request_outputs` and `request_rewards` (from the previous step) and assert that the output Arrow Table has the correct schema, columns, and data (especially `input_tokens`, `output_tokens`, `rewards`, `advantages`, `logprobs`, `loss_mask`).

2.  **Simulate Parquet File Generation:**
    *   Create a Python script that uses your tested `get_parquet_table` (or the relevant parts of your modified `infer.py`) to generate a small number of sample Parquet files.
    *   These files should represent the Sanskrit task, containing a few examples of (metadata tokens, generated Sanskrit tokens, rewards from verifier, advantages, policy logprobs, loss mask).
    *   Place these files in a local directory mimicking the `output_path/step_0/` structure.

3.  **Test `train.py` Data Loading and Processing:**
    *   Ensure all incompatible dependencies (`vllm`, `liger_kernel`, `llmcompressor`, `prime-iroh`, `flash-attn`) are commented out or conditionally skipped in `pyproject.toml` and the Python code for this local testing.
    *   Configure `configs/training/sanskrit_morph.toml` to point `data.path` to your local directory of simulated Parquet files.
    *   Run `uv run python prime-rl/src/zeroband/train.py @ prime-rl/configs/training/sanskrit_morph.toml` (using `python` directly, not `torchrun`, for simpler single-process debugging initially).
    *   **Goal:** Verify that `train.py` can successfully:
        *   Load your Parquet files via `ParquetDataset`.
        *   Correctly process the data into `BatchOutput` (check `input_ids` concatenation, `loss_mask`, rewards, advantages, `original_logprobs`).
        *   Attempt to compute loss (even if the model is tiny or not actually training meaningfully, the data flow is what's being tested).
    *   Use a debugger or print statements to trace the data.

## Phase 3: Preparing the Pull Request for Rohan's Fork

1.  **Create a New Branch:** In your local clone of Rohan's fork (`thepreetam/prime-rl`).
2.  **Commit Changes:** Add all new files and modifications from Phase 1. Include the (commented-out for macOS) `vidyut` dependency in `pyproject.toml`.
3.  **Documentation:**
    *   Add a new Markdown file (e.g., `docs/sanskrit_morphological_rendering.md`) explaining:
        *   The new "sanskrit_morph" task.
        *   How to prepare data (if any specific format is needed for the initial Hugging Face dataset).
        *   How to run the task (example commands for `infer_sanskrit.py` and `train.py` using the new `.toml` configs).
        *   The role of `vidyut` and the verifier.
        *   Any assumptions made.
    *   Update the main `README.md` if necessary to point to this new documentation.
4.  **Testing (Mention Local Simulation):** In your PR description, explain the local simulation steps you took (Phase 2) to verify the data pipeline, given the inability to run the full `vLLM` stack on macOS.
5.  **Submit PR:** Push your branch to your GitHub fork and open a Pull Request against Rohan's `prime-rl` repository. Clearly explain the changes, the goal, and the testing performed. 