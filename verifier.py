from vidyut.prakriya import (
    Vyakarana, Dhatu, Gana, Lakara, Prayoga, Purusha, Vacana, Pada, Data, DhatupathaEntry
)
import re
import os

# --- Data Loading (needed to reconstruct Dhatu object from string) ---
local_download_root = "vidyut_data_root"
prakriya_data_path = os.path.join(local_download_root, "prakriya")

# Cache for Dhatu objects to avoid repeated loading and searching
# Key: aupadeshika string, Value: Dhatu object
DHATU_CACHE = {}
VYAKARANA_INSTANCE = Vyakarana()

def get_dhatu_by_aupadeshika(aupadeshika_str):
    if aupadeshika_str in DHATU_CACHE:
        return DHATU_CACHE[aupadeshika_str]

    # Load all entries if cache is empty (first call)
    # This is inefficient if called multiple times for different dhatus not yet cached
    # but for a typical verifier use case (verifying many forms for a pre-set list of inputs),
    # this might be okay. A better approach might be to pre-populate cache.
    if not DHATU_CACHE: # Simplified: assumes if any are cached, all loadable ones are.
        print("Populating Dhatu cache...") # Should only print once
        data_provider = Data(path=prakriya_data_path)
        all_entries = data_provider.load_dhatu_entries()
        for entry in all_entries:
            DHATU_CACHE[entry.dhatu.aupadeshika] = entry.dhatu
        print(f"Cached {len(DHATU_CACHE)} Dhatu objects.")

    return DHATU_CACHE.get(aupadeshika_str)

# --- Enum String to Enum Member Mapping ---
# Vidyut enums have a from_string class method

# --- Verifier Function ---
def parse_metadata_string(metadata_str):
    """Parses the LLM input string back into components needed for Vidyut."""
    # Example: "Dhātu: BU (BvAdi), Lakāra: Lat, Prayoga: Kartari, Puruṣa: Prathama, Vacana: Eka"
    pattern = re.compile(
        r"Dhātu: (?P<dhatu_name>[^\s(]+) \((?P<gana_name>[^)]+)\), "
        r"Lakāra: (?P<lakara_name>[^,]+), "
        r"Prayoga: (?P<prayoga_name>[^,]+), "
        r"Puruṣa: (?P<purusha_name>[^,]+), "
        r"Vacana: (?P<vacana_name>[^,]+)"
    )
    match = pattern.match(metadata_str)
    if not match:
        raise ValueError(f"Could not parse metadata string: {metadata_str}")
    
    data = match.groupdict()
    
    dhatu_obj = get_dhatu_by_aupadeshika(data['dhatu_name'])
    if not dhatu_obj:
        raise ValueError(f"Dhatu '{data['dhatu_name']}' not found in cached Dhatupatha.")

    # Validate Gana if needed, though it's part of the Dhatu object
    # For now, assume the string Gana matches the object Gana, or that it's not critical for re-derivation.

    try:
        lakara_enum = Lakara.from_string(data['lakara_name'])
        prayoga_enum = Prayoga.from_string(data['prayoga_name'])
        purusha_enum = Purusha.from_string(data['purusha_name'])
        vacana_enum = Vacana.from_string(data['vacana_name'])
    except ValueError as e: # from_string raises ValueError for invalid names
        raise ValueError(f"Error converting string to enum: {e}. Original string: {metadata_str}")

    return Pada.Tinanta(
        dhatu=dhatu_obj,
        lakara=lakara_enum,
        prayoga=prayoga_enum,
        purusha=purusha_enum,
        vacana=vacana_enum
    )

def verify_form(metadata_input_str: str, llm_generated_form: str) -> bool:
    """
    Verifies if the llm_generated_form is a valid derivation for the given metadata_input_str.
    
    Args:
        metadata_input_str: The LLM input string (e.g., "Dhātu: BU (BvAdi), ...").
        llm_generated_form: The surface form produced by the LLM.
        
    Returns:
        True if llm_generated_form is one of the Vidyut-derived forms, False otherwise.
    """
    if not llm_generated_form: # An empty form is definitely not correct
        return False
        
    try:
        tinanta_spec = parse_metadata_string(metadata_input_str)
    except ValueError as e:
        print(f"Error parsing metadata for verifier: {e}")
        return False # Cannot verify if metadata is unparsable

    try:
        prakriyas = VYAKARANA_INSTANCE.derive(tinanta_spec)
        if prakriyas:
            valid_forms = [p.text for p in prakriyas]
            return llm_generated_form in valid_forms
        else:
            # If Vidyut generates no forms, then any LLM form is technically incorrect *unless* the LLM
            # is expected to produce forms even when Vidyut doesn't (e.g. handling archaic/rare forms not in Vidyut).
            # For this project, assume Vidyut is the ground truth: if it has no forms, LLM should also have none.
            # However, an LLM producing *something* when Vidyut produces *nothing* is a mismatch.
            return False 
    except Exception as e:
        print(f"Error during Vidyut derivation in verifier: {e}")
        return False


if __name__ == '__main__':
    # Test cases for the verifier
    print("Running verifier test cases...")

    # Case 1: Correct form (using enum.value for SLP1 strings)
    meta1 = f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form1_correct = "Bavati"
    form1_incorrect = "Bavami"
    print(f"Test 1 (Correct): {meta1} -> {form1_correct} : {verify_form(meta1, form1_correct)}")
    print(f"Test 1 (Incorrect): {meta1} -> {form1_incorrect} : {verify_form(meta1, form1_incorrect)}")

    # Case 2: Another Dhatu, different form
    # Ensure kf (TanAdi) uses .value for enums too
    meta2 = f"Dhātu: kf (TanAdi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form2_correct = "karoti"
    form2_incorrect_gana = "kRNoti" 
    if get_dhatu_by_aupadeshika("kf") and get_dhatu_by_aupadeshika("kf").gana.name == "TanAdi":
        print(f"Test 2 (Correct): {meta2} -> {form2_correct} : {verify_form(meta2, form2_correct)}")
        print(f"Test 2 (Incorrect Gana/Form): {meta2} -> {form2_incorrect_gana} : {verify_form(meta2, form2_incorrect_gana)}")
    else:
        print(f"Skipping Test 2 for kf (TanAdi) as Dhatu not found or Gana mismatch in cache.")

    # Case 3: Vidyut produces multiple forms (if we can find one)
    # Example: tud (TudAdi) + Lat + Kartari + Pra + Eka -> tudati, todaTi (hypothetical if an option exists)
    # For now, we check if LLM form is *in* the list of Vidyut forms.

    # Case 4: LLM produces empty form (meta1 uses .value)
    print(f"Test 4 (Empty LLM form): {meta1} -> '' : {verify_form(meta1, '')}")

    # Case 5: Invalid metadata string (this one is intentionally malformed, so it's fine)
    meta5_invalid = "Dhātu: XYZ, Lakāra: Foo, Prayoga: Bar, Puruṣa: Baz, Vacana: Qux"
    print(f"Test 5 (Invalid Metadata): {meta5_invalid} -> {form1_correct} : {verify_form(meta5_invalid, form1_correct)}")

    # Case 6: Valid metadata, but Vidyut produces no forms (use .value for enums)
    meta6_no_form_attempt = f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Let.value}, Prayoga: {Prayoga.Karmani.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form6 = "someform"
    print(f"Test 6 (Vidyut no form expected): {meta6_no_form_attempt} -> {form6} : {verify_form(meta6_no_form_attempt, form6)}")

    # Inspect cache for specific dhatus
    print("\nCache inspection:")
    if DHATU_CACHE:
        for k, v_dhatu in DHATU_CACHE.items():
            if "kf" in k or "tan" in k or "BU" in k: # Look for variations
                 print(f"  Found in cache: Key='{k}', Aupadeshika='{v_dhatu.aupadeshika}', Gana='{v_dhatu.gana.name}', Original Gana Enum: {v_dhatu.gana}")
    else:
        print("  DHATU_CACHE is empty or not populated at this point of inspection.") 