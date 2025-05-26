# Engineering Plan: Morphological Rendering with Vidyut-Prakriya and LLMs

**Project Goal:** To develop a system that renders underlying morphological metadata into surface Sanskrit forms using a Large Language Model (LLM), verified by `Vidyut-Prakriya`. Subsequently, to evaluate if this training improves the LLM's generalization capabilities for English-to-Sanskrit translation.

**Inspired by:** Rohan Pandey's call for RL projects for Sanskrit, specifically: "Panini's generative grammar renders underlying morphological metadata to a surface Sanskrit form. Generate a dataset of random dhatus & morphological data, then write a verifier using Vidyut-Prakriya. Does the model generalize to improve English->Sanskrit translation ability?"

## 1. Core Components

1.  **Dataset Generation Module:** Creates input-output pairs of (dhātu + morphological metadata) -> (surface Sanskrit form).
2.  **Vidyut-Prakriya Verifier:** A module that uses `Vidyut-Prakriya` to confirm the correctness of a generated surface form given the input metadata.
3.  **Rendering LLM:** A language model trained/fine-tuned to perform the rendering task.
4.  **Evaluation Framework:** Metrics and datasets to assess rendering accuracy and generalization to translation.

## 2. Detailed Phased Plan

### Phase 1: Setup & Foundation

1.  **Environment Setup:**
    *   Create a dedicated Python virtual environment.
    *   Install necessary base libraries: `transformers`, `datasets`, `torch`, `pandas`, `pytest`.
2.  **Vidyut-Prakriya Integration:**
    *   Research `Vidyut-Prakriya`: Identify its installation method, API (likely Rust), and potential Python bindings or CLI interface. The provided image shows Rust usage.
        *   If no direct Python bindings exist, develop a minimal Python wrapper to call `Vidyut-Prakriya` (e.g., using `subprocess` if a CLI is available, or exploring Rust-Python interop libraries like `PyO3` if building from source is feasible).
    *   Test basic `Vidyut-Prakriya` functionality: e.g., derive forms for a few sample dhātus and morphological specifications as shown in the example image (`bhū` -> `bhavati`).
3.  **Lexicon Acquisition:**
    *   Utilize or adapt the `SanskritLexicon` from the `sanskrit_nlp_toolkit` to get a comprehensive list of dhātus, their gaṇas, and other relevant properties.
    *   If `SanskritLexicon` is not yet complete for this purpose, supplement with other dhātu lists (e.g., from existing databases or Paninian grammar resources).

### Phase 2: Dataset Generation 

1.  **Define Morphological Metadata Schema:**
    *   Based on `Vidyut-Prakriya`'s input capabilities (as seen in `Tinanta::builder()` example):
        *   **For Tiṅantas (Verbs):** dhātu (root), gaṇa (class), lakāra (tense/mood), prayoga (voice), puruṣa (person), vacana (number).
        *   **(Future Scope/Optional) For Subantas (Nouns):** prātipadika (stem), liṅga (gender), vibhakti (case), vacana (number). Initially focus on tiṅantas as per example.
    *   Map these fields to the `MorphologicalAnalysis` IR from `sanskrit_nlp_toolkit` where applicable, for consistency.
2.  **Data Generation Script:**
    *   Create a Python script that:
        *   Iterates through the list of dhātus.
        *   For each dhātu, generates random but valid combinations of the defined morphological metadata (e.g., all valid puruṣa/vacana for a given lakāra).
        *   **Input Formatting:** Decide on a clear text format for the LLM input, e.g., "Dhātu: भू (Bhvādi), Lakāra: Lat, Prayoga: Kartari, Puruṣa: Prathama, Vacana: Eka".
3.  **Ground Truth Generation using Vidyut-Prakriya:**
    *   For each generated (dhātu + metadata) combination:
        *   Construct the input for `Vidyut-Prakriya` using the Python wrapper developed in Phase 1.
        *   Invoke `Vidyut-Prakriya` to derive the surface Sanskrit form(s). Handle cases where multiple forms might be valid or no form is generated.
        *   Store the input metadata and the `Vidyut-Prakriya`-generated surface form as an input-output pair.
4.  **Dataset Formatting and Storage:**
    *   Store the dataset in a standard format (e.g., JSON Lines, CSV, or Hugging Face `datasets` library format). Each entry: `{"input_metadata_str": "...", "surface_form_vidyut": "..."}`.
    *   Create training, validation, and test splits. Ensure representation of diverse dhātus and grammatical forms in each split.
5.  **Initial Dataset Size:** Aim for at least 10,000-50,000 unique examples, expandable later.

### Phase 3: Verifier Implementation (Estimated Time: 1 week)

1.  **Python Verifier Function:**
    *   Develop a Python function `verify_form(metadata_input, generated_form)` that:
        *   Takes the original morphological metadata (used to generate the LLM task) and the LLM's `generated_form`.
        *   Uses the `Vidyut-Prakriya` wrapper to get the canonical form(s) for `metadata_input`.
        *   Returns `True` if `generated_form` matches one of the canonical forms (or the primary canonical form), `False` otherwise. Account for potential minor variations if `Vidyut-Prakriya` outputs multiple correct forms.
2.  **Unit Tests for Verifier:**
    *   Write `pytest` tests for the verifier with known correct and incorrect examples.

### Phase 4: Model Selection and Training 

1.  **LLM Selection:**
    *   Choose a suitable pre-trained multilingual or Sanskrit-aware LLM from Hugging Face Hub (e.g., IndicBERT, mBART, a smaller BLOOM variant, or any model showing promise for Sanskrit). Consider models with good sequence-to-sequence capabilities.
2.  **Training Strategy:**
    *   **Option A: Supervised Fine-Tuning (SFT) - Recommended Start**
        *   Task: Sequence-to-sequence. Input is the formatted metadata string, output is the surface Sanskrit form.
        *   Fine-tune the selected LLM on the generated dataset (Phase 2).
        *   Use standard cross-entropy loss.
        *   Metrics: Exact match accuracy against the validation set (using the verifier from Phase 3 for robust checking beyond simple string match), character error rate (CER).
    *   **Option B: Reinforcement Learning (RL) - Advanced, as per Rohan's suggestion**
        *   This would typically follow an initial SFT phase or be used for further refinement.
        *   **Environment:**
            *   State: The input morphological metadata string.
            *   Action: The LLM generates the surface form character by character or token by token.
            *   Reward:
                *   `+1` (or a higher positive value) if the fully generated form is correct according to the `Vidyut-Prakriya` verifier.
                *   `0` or a small negative reward for incorrect forms.
                *   (Optional) Intermediate rewards for partial correctness if feasible.
        *   **Algorithm:** Proximal Policy Optimization (PPO) is a common choice for LLM fine-tuning with RL. Libraries like `trl` (from Hugging Face) or `prime-rl` (mentioned by Rohan) can be used.
        *   The SFT model can serve as the initial policy for RL.
3.  **Training Infrastructure:**
    *   Utilize GPUs (e.g., Google Colab, Kaggle, or cloud VMs).
    *   Employ Hugging Face `Trainer` API for SFT or relevant RL libraries.

### Phase 5: Evaluation

1.  **Rendering Accuracy Evaluation:**
    *   Evaluate the trained LLM on the held-out test set from Phase 2.
    *   **Metrics:**
        *   Exact Match Accuracy (verified by `Vidyut-Prakriya`).
        *   BLEU score (if treating as a "translation" from metadata to form).
        *   Character Error Rate (CER) / Word Error Rate (WER) if applicable.
    *   Error Analysis: Categorize common error types (e.g., incorrect sandhi, wrong affix, issues with specific lakāras).
2.  **Generalization to English-Sanskrit Translation:**
    *   **Baseline Model:**
        *   Take the *same base pre-trained LLM* (used in Phase 4.1) without the morphological rendering training.
        *   Fine-tune it on a standard English-Sanskrit parallel corpus (e.g., FLORES, PMIndia, or other available datasets).
    *   **Morphologically-Aware Model:**
        *   Take the LLM already fine-tuned/RL-trained for morphological rendering.
        *   Further fine-tune this model on the *same* English-Sanskrit parallel corpus using the *same* hyperparameters as the baseline.
    *   **Comparison:**
        *   Evaluate both models (Baseline vs. Morphologically-Aware) on a held-out English-Sanskrit test set.
        *   **Metrics:** BLEU, METEOR, TER.
        *   **Qualitative Analysis:** Crucially, perform human or expert linguistic evaluation focusing on:
            *   Grammatical correctness of the generated Sanskrit (verb conjugations, noun declensions, sandhi).
            *   Fluency and naturalness.
        *   The hypothesis is that the morphologically-aware model will produce more grammatically accurate Sanskrit translations.

### Phase 6: Documentation, Reporting & Contribution 
1.  **Code & Dataset:**
    *   Clean up and document all code (dataset generation scripts, verifier, training scripts).
    *   Write comprehensive unit tests using `pytest`.
    *   Publish the generated dataset to Hugging Face Datasets Hub with a clear data card.
2.  **Report:**
    *   Summarize the methodology, results, and findings regarding rendering accuracy and generalization to translation.
    *   Discuss challenges, learnings, and future work.
3.  **Contribution (as per Rohan's request):**
    *   Prepare a Pull Request to a relevant repository (if specified by Rohan) including:
        *   Key scripts or modules.
        *   Unit tests.
        *   Link to the Hugging Face dataset.

## 3. Tools and Technologies

*   **Programming Language:** Python 3.x
*   **Core Libraries:**
    *   Hugging Face: `transformers`, `datasets`, `accelerate`, `trl` (for RL)
    *   `torch` (or TensorFlow/JAX depending on LLM choice)
    *   `pandas` (for data manipulation)
    *   `pytest` (for testing)
*   **Sanskrit Processing:** `Vidyut-Prakriya` (Rust tool, via wrapper)
*   **(Optional) RL Framework:** `prime-rl` or similar.
*   **Version Control:** Git & GitHub/GitLab.

## 4. Potential Challenges & Mitigation

*   **Vidyut-Prakriya Integration:**
    *   Challenge: Difficulty in calling Rust code from Python, or limited API.
    *   Mitigation: Prioritize finding/building a stable CLI or Python binding. Start with a subset of `Vidyut-Prakriya` features if full integration is too complex initially.
*   **Scope of Morphological Data:**
    *   Challenge: `Vidyut-Prakriya` might not cover all desired dhātus or grammatical forms, or its input requirements might be complex.
    *   Mitigation: Start with a well-supported core set (e.g., common dhātus, basic tiṅanta forms) and expand iteratively. Document any limitations.
*   **Dataset Quality:**
    *   Challenge: Ensuring generated metadata combinations are linguistically valid and diverse.
    *   Mitigation: Cross-reference with Paninian grammar texts or expert input for validation rules during generation.
*   **LLM Training Resources:**
    *   Challenge: Access to sufficient GPU resources for training large models.
    *   Mitigation: Start with smaller model variants. Utilize free resources like Colab/Kaggle. Explore parameter-efficient fine-tuning (PEFT) techniques like LoRA.
*   **Evaluating Generalization:**
    *   Challenge: Isolating the impact of morphological training on translation; confounding factors.
    *   Mitigation: Rigorous experimental setup with controlled comparisons. Emphasize qualitative linguistic evaluation in addition to automated metrics.

## 5. Deliverables

1.  A Hugging Face Dataset of (dhātu + morphological metadata, surface Sanskrit form) pairs, verified by `Vidyut-Prakriya`.
2.  Python scripts for dataset generation and `Vidyut-Prakriya` verifier.
3.  Trained LLM(s) for morphological rendering (SFT and optionally RL versions).
4.  Evaluation results comparing the morphologically-aware model with a baseline on English-Sanskrit translation.
5.  A final report detailing the project, methodology, findings, and contributions.
6.  Unit tests for key components.

This plan provides a structured approach to tackling the proposed project. Flexibility will be needed to adapt to challenges and new insights as the project progresses. 