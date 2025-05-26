import vidyut
from vidyut.prakriya import (
    Data,
    DhatupathaEntry,
    Dhatu,
    Gana,
    Lakara,
    Prayoga,
    Purusha,
    Vacana,
    Pada
)
from vidyut.prakriya import Vyakarana
import os
import random
import json

# --- Setup Data Loading ---
local_download_root = "vidyut_data_root"
prakriya_data_path = os.path.join(local_download_root, "prakriya")

def get_dhatu_entries():
    if not os.path.exists(os.path.join(prakriya_data_path, "dhatupatha.tsv")):
        print(f"dhatupatha.tsv not found in {prakriya_data_path}. Run download_vidyut_data.py first.")
        return []
    data_provider = Data(path=prakriya_data_path)
    return data_provider.load_dhatu_entries()

# --- Define Metadata Generation Logic ---

def get_all_lakaras():
    return Lakara.choices()

def get_all_prayogas():
    return Prayoga.choices()

def get_all_purushas():
    return Purusha.choices()

def get_all_vacanas():
    return Vacana.choices()

# --- Main Dataset Generation ---
def main():
    dhatu_entries = get_dhatu_entries()
    if not dhatu_entries:
        print("Could not load dhatu entries. Exiting.")
        return

    print(f"Loaded {len(dhatu_entries)} dhatu entries.")

    v = Vyakarana()

    # For demonstration, let's process the first few dhatus
    # and generate some combinations.
    num_dhatus_to_process = 100 # Increased for a larger dataset
    dataset_examples = []

    print(f"Targeting {num_dhatus_to_process} dhātus for processing...")

    for i, entry in enumerate(dhatu_entries):
        if i >= num_dhatus_to_process:
            break

        dhatu_obj = entry.dhatu
        print(f"\nProcessing Dhatu: {dhatu_obj.aupadeshika} (Gana: {dhatu_obj.gana}, Code: {entry.code})")

        # Generate combinations of morphological features
        # For simplicity, let's pick one of each for now
        # Later, we'll iterate through all valid combinations or random selections.

        lakaras = get_all_lakaras()
        prayogas = get_all_prayogas()
        purushas = get_all_purushas()
        vacanas = get_all_vacanas()

        # Iterate through all combinations for this dhatu
        for lakara_enum in lakaras:
            for prayoga_enum in prayogas:
                for purusha_enum in purushas:
                    for vacana_enum in vacanas:
                        # Input for Vidyut-Prakriya
                        tinanta_spec = Pada.Tinanta(
                            dhatu=dhatu_obj,
                            lakara=lakara_enum,
                            prayoga=prayoga_enum,
                            purusha=purusha_enum,
                            vacana=vacana_enum
                        )
                        
                        llm_input_str = (
                            f"Dhātu: {dhatu_obj.aupadeshika} ({dhatu_obj.gana}), "
                            f"Lakāra: {lakara_enum.value}, Prayoga: {prayoga_enum.value}, "
                            f"Puruṣa: {purusha_enum.value}, Vacana: {vacana_enum.value}"
                        )
                        
                        current_example = {
                            "llm_input": llm_input_str,
                            "vidyut_input_spec": tinanta_spec, # Storing spec temporarily for potential debugging
                            "surface_form_vidyut": None
                        }

                        try:
                            prakriyas = v.derive(tinanta_spec)
                            if prakriyas:
                                surface_form = prakriyas[0].text
                                current_example["surface_form_vidyut"] = surface_form
                                # print(f"  SUCCESS: {llm_input_str} -> {surface_form}") # Optional: for verbose logging
                            else:
                                # print(f"  INFO: No form for {llm_input_str}") # Optional: for verbose logging
                                pass # Keep surface_form_vidyut as None
                        except Exception as e:
                            # print(f"  ERROR deriving for {llm_input_str}: {e}") # Optional: for verbose logging
                            pass # Keep surface_form_vidyut as None
                        
                        dataset_examples.append(current_example)

        print(f"  Processed Dhatu {dhatu_obj.aupadeshika}, generated {len(lakaras)*len(prayogas)*len(purushas)*len(vacanas)} specifications.")

    print(f"\nGenerated {len(dataset_examples)} total example specifications.")
    # Next steps: 
    # 1. Generate ground truth using Vidyut-Prakriya for each spec.
    # 2. Store in a structured format (JSONL, CSV).
    # 3. Implement more comprehensive combination generation (not just one random).

    # Store dataset_examples to a JSONL file
    output_filename = "tinanta_dataset_raw.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for example in dataset_examples:
            # The vidyut_input_spec is a Pada.Tinanta object, which is not directly JSON serializable.
            # We should store its components if needed, or just the LLM input and Vidyut output.
            # For now, let's store the LLM input and the derived surface form.
            # We also need to remove the actual tinanta_spec object before saving to JSON.
            record_to_store = {
                "llm_input": example["llm_input"],
                "surface_form_vidyut": example["surface_form_vidyut"]
            }
            if example["surface_form_vidyut"] is not None: # Only write if a form was generated
                 f.write(json.dumps(record_to_store, ensure_ascii=False) + '\n')
    
    final_written_count = sum(1 for ex in dataset_examples if ex["surface_form_vidyut"] is not None)
    print(f"\nSuccessfully wrote {final_written_count} examples (where Vidyut returned a form) to {output_filename}")

if __name__ == "__main__":
    main() 