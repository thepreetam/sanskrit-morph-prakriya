from huggingface_hub import HfApi, upload_file
import os

# --- Configuration ---
# !!! IMPORTANT: Please replace these with your actual username and desired dataset name !!!
HF_USERNAME = "preetammukherjee"
DATASET_NAME = "sanskrit_morph_prakriya"
# --- 

LOCAL_DATA_FILE = "tinanta_dataset_raw.jsonl"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

def create_dataset_card_content():
    """Creates the content for the README.md (Dataset Card)."""
    content = f"""\
---
license: mit # Or choose another appropriate license, e.g., cc-by-sa-4.0
language:
  - sa # Sanskrit
tags:
  - sanskrit
  - morphology
  - prakriya
  - vidyut
  - generative-grammar
  - sequence-to-sequence
---

# Vidyut-Prakriya Tinanta Dataset for Morphological Rendering

This dataset contains pairs of Sanskrit morphological metadata and their corresponding surface forms,
_generated_ and _verified_ using the `vidyut-prakriya` library.

## Dataset Structure

The dataset is provided in JSON Lines (`.jsonl`) format. Each line is a JSON object with the following fields:

- `llm_input` (string): A textual representation of the morphological metadata. This serves as the input for a sequence-to-sequence LLM tasked with morphological rendering.
  Example: `"Dhātu: BU (BvAdi), Lakāra: la~w, Prayoga: kartari, Puruṣa: praTama, Vacana: eka"`
- `surface_form_vidyut` (string): The Sanskrit surface form derived by `vidyut-prakriya` for the given metadata.
  Example: `"Bavati"`

## Generation Process

1.  **Dhātu Lexicon**: Dhātus (verb roots) are sourced from the `dhatupatha.tsv` provided with `vidyut-prakriya` (version 0.4.0 data).
2.  **Metadata Combination**: For each dhātu, combinations of the following morphological features are generated:
    - `Lakāra` (tense/mood)
    - `Prayoga` (voice: kartari, karmani, bhave)
    - `Puruṣa` (person: prathama, madhyama, uttama)
    - `Vacana` (number: eka, dvi, bahu)
3.  **Derivation & Verification**: The `vidyut.prakriya.Vyakarana.derive()` method is used to generate the surface form for each metadata combination. Only combinations that yield a valid surface form are included in the dataset.
4.  **LLM Input Format**: The `llm_input` string is formatted to be human-readable and suitable for sequence-to-sequence models. Enum values (Lakāra, Prayoga, etc.) are represented by their SLP1 strings (e.g., `la~w` for `Lakāra.Lat`).

## Intended Use

This dataset is primarily intended for training and evaluating language models on the task of Sanskrit morphological rendering (i.e., generating a surface form from its underlying grammatical specification).

It can also be used for:
- Analyzing the coverage of `vidyut-prakriya`.
- Studies in Sanskrit computational linguistics.

## Project Context

This dataset was generated as part of a project inspired by Rohan Pandey's call for RL projects for Sanskrit. The goal is to use this data to train a model for morphological rendering and subsequently evaluate its impact on English-to-Sanskrit translation quality.

## Citation

If you use this dataset, please consider citing the `vidyut-prakriya` library and/or this repository (once created).

```
"""
    return content

def upload_dataset():
    api = HfApi()

    # Create the repository on the Hub (if it doesn't exist)
    # The `create_repo` function has an `exist_ok` parameter.
    print(f"Creating dataset repository: {REPO_ID} (if it doesn't exist)...")
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
        print(f"Repository {REPO_ID} ensured.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload the data file
    print(f"Uploading {LOCAL_DATA_FILE} to {REPO_ID}...")
    try:
        upload_file(
            path_or_fileobj=LOCAL_DATA_FILE,
            path_in_repo=os.path.basename(LOCAL_DATA_FILE), # Use the same filename in the repo
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add dataset file: {os.path.basename(LOCAL_DATA_FILE)}"
        )
        print(f"Successfully uploaded {LOCAL_DATA_FILE}.")
    except Exception as e:
        print(f"Error uploading data file: {e}")
        # If data file upload fails, we might not want to upload README yet, or handle differently.

    # Create and upload the Dataset Card (README.md)
    readme_content = create_dataset_card_content()
    readme_file_path = "README.md"
    with open(readme_file_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"Uploading {readme_file_path} to {REPO_ID}...")
    try:
        upload_file(
            path_or_fileobj=readme_file_path,
            path_in_repo=readme_file_path, # Path in repo is README.md
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add dataset card (README.md)"
        )
        print(f"Successfully uploaded {readme_file_path}.")
    except Exception as e:
        print(f"Error uploading {readme_file_path}: {e}")
    finally:
        # Clean up local README.md if it was created
        if os.path.exists(readme_file_path):
            os.remove(readme_file_path)

    print(f"\nDataset upload process complete. You can view your dataset at:")
    print(f"https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    if not os.path.exists(LOCAL_DATA_FILE):
        print(f"Error: Local data file '{LOCAL_DATA_FILE}' not found. Please generate it first.")
    else:
        # Before running, double check HF_USERNAME and DATASET_NAME at the top of this script!
        print(f"About to upload dataset to Hugging Face Hub.")
        print(f"Local file: {LOCAL_DATA_FILE}")
        print(f"Target Hub repository: {REPO_ID}")
        confirmation = input("Proceed? (y/n): ")
        if confirmation.lower() == 'y':
            upload_dataset()
        else:
            print("Upload cancelled by user.")
        # print("Please confirm HF_USERNAME and DATASET_NAME in the script, then uncomment the user confirmation and `upload_dataset()` call to proceed.")
        # For now, to avoid accidental uploads with placeholder names, the direct call is commented out.
        # To proceed: 
        # 1. Edit HF_USERNAME and DATASET_NAME in this script.
        # 2. Uncomment the input() and upload_dataset() lines below.
        # upload_dataset() # Make sure to uncomment this line after setting your username and dataset name. 