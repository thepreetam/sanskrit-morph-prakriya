# Sanskrit Morphological Rendering with Vidyut-Prakriya

This project implements a system for generating Sanskrit morphological data and verifying surface forms using the `vidyut-prakriya` library. It addresses the "surface morphology/paninian rendering" project idea proposed by Rohan Pandey.

The primary goal is to create a dataset suitable for training a language model to render underlying morphological metadata into surface Sanskrit forms. A verifier is also provided to check the correctness of generated forms against `vidyut-prakriya`.

**Hugging Face Dataset:** [preetammukherjee/sanskrit_morph_prakriya](https://huggingface.co/datasets/preetammukherjee/sanskrit_morph_prakriya)

## Project Components

- **Dataset Generator (`generate_dataset.py`):** Creates input-output pairs of (dhātu + morphological metadata) -> (surface Sanskrit form). It uses `vidyut-prakriya` for deriving ground truth forms.
- **Vidyut-Prakriya Verifier (`verifier.py`):** A module that uses `vidyut-prakriya` to confirm the correctness of a generated surface form given the input metadata.
- **Unit Tests (`test_verifier.py`):** Pytest suite for the verifier.
- **Vidyut Data Downloader (`download_vidyut_data.py`):** Script to download the necessary data files for `vidyut-prakriya`.
- **Engineering Plan (`morphological_rendering_rl_plan.md`):** The detailed plan followed for this implementation.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Download Vidyut-Prakriya Data:**
    This data is required by `vidyut-prakriya` for its operations (e.g., Dhatupatha).
    ```bash
    python download_vidyut_data.py
    ```
    This will create a `vidyut_data_root` directory with the necessary files.

2.  **Generate the Dataset:**
    This script will produce the `tinanta_dataset_raw.jsonl` file.
    ```bash
    python generate_dataset.py
    ```
    You can modify `num_dhatus_to_process` within the script to control the dataset size. The current version generates ~30,000 examples from 100 dhātus.

3.  **Run Unit Tests for the Verifier:**
    ```bash
    pytest
    ```
    Or, more specifically:
    ```bash
    python -m pytest test_verifier.py
    ```

4.  **Using the Verifier (Example from `verifier.py`):**
    The `verifier.py` script can be run directly to see test cases:
    ```bash
    python verifier.py 
    ```
    You can import `verify_form` from `verifier` in other scripts:
    ```python
    from verifier import verify_form
    metadata_str = "Dhātu: BU (BvAdi), Lakāra: la~w, Prayoga: kartari, Puruṣa: praTama, Vacana: eka"
    llm_output = "Bavati"
    is_correct = verify_form(metadata_str, llm_output)
    print(f"Is '{llm_output}' correct for '{metadata_str}'? {is_correct}")
    ```

## Next Steps (as per Engineering Plan)

-   Fine-tune a sequence-to-sequence Language Model (e.g., T5-small) on the generated dataset for morphological rendering.
-   Evaluate the trained LLM on rendering accuracy.
-   Explore Reinforcement Learning (RL) for further refinement.
-   Evaluate the impact of morphological rendering training on a downstream English-to-Sanskrit translation task.

This project provides the foundational dataset and verifier needed for these subsequent steps. 