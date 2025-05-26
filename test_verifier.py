import pytest
from verifier import verify_form, get_dhatu_by_aupadeshika # Assuming verifier.py is in the same directory or PYTHONPATH
from vidyut.prakriya import Lakara, Prayoga, Purusha, Vacana # For constructing test metadata

# Test cases for the verifier

@pytest.fixture(scope="module")
def populated_dhatu_cache():
    # Ensure the cache is populated once for all tests in this module
    # by calling a function that triggers cache population if it's not already done.
    # The first call to verify_form or get_dhatu_by_aupadeshika will do this.
    # For example, get a known dhatu:
    if not get_dhatu_by_aupadeshika("BU"):
        # This state should ideally not be reached if tests run after verifier.py main block,
        # but it's a safeguard.
        verify_form(f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}", "Bavati")
    return True

def test_correct_form(populated_dhatu_cache):
    meta = f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_correct = "Bavati"
    assert verify_form(meta, form_correct) == True

def test_incorrect_form(populated_dhatu_cache):
    meta = f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_incorrect = "Bavami"
    assert verify_form(meta, form_incorrect) == False

def test_empty_llm_form(populated_dhatu_cache):
    meta = f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    assert verify_form(meta, "") == False

def test_invalid_metadata_string(populated_dhatu_cache):
    meta_invalid = "Dhātu: XYZ, Lakāra: Foo, Prayoga: Bar, Puruṣa: Baz, Vacana: Qux"
    form_correct = "Bavati" # A valid form, but metadata is bad
    assert verify_form(meta_invalid, form_correct) == False

def test_vidyut_produces_no_form(populated_dhatu_cache):
    # BU + Let + Karmani + Prathama + Eka -> Vidyut produces no form for this specific combination
    meta_no_form = f"Dhātu: BU (BvAdi), Lakāra: {Lakara.Let.value}, Prayoga: {Prayoga.Karmani.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_some_output = "someform" # LLM produces something
    assert verify_form(meta_no_form, form_some_output) == False

# --- Tests for kf (SvAdi) ---
def test_kf_svadi_correct(populated_dhatu_cache):
    dhatu_key = "kf\\Y" 
    dhatu_kf_svadi = get_dhatu_by_aupadeshika(dhatu_key)

    if not (dhatu_kf_svadi and dhatu_kf_svadi.gana.name == "Svadi"):
        pytest.skip(f"{dhatu_key} (Svadi) not found or Gana is not Svadi. Actual Gana: {dhatu_kf_svadi.gana.name if dhatu_kf_svadi else 'N/A'}. Skipping test.")
    
    meta = f"Dhātu: {dhatu_key} (Svadi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_correct = "kfRoti"
    assert verify_form(meta, form_correct) == True

def test_kf_svadi_incorrect_form(populated_dhatu_cache):
    dhatu_key = "kf\\Y"
    dhatu_kf_svadi = get_dhatu_by_aupadeshika(dhatu_key)
    if not (dhatu_kf_svadi and dhatu_kf_svadi.gana.name == "Svadi"):
        pytest.skip(f"{dhatu_key} (Svadi) not found or Gana is not Svadi. Skipping test.")

    meta = f"Dhātu: {dhatu_key} (Svadi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_incorrect = "karoti" 
    assert verify_form(meta, form_incorrect) == False

# --- Tests for qukfY (TanAdi) ---
def test_qukfY_tanadi_correct(populated_dhatu_cache):
    dhatu_key = "qukf\\Y"
    dhatu_qukf_tanadi = get_dhatu_by_aupadeshika(dhatu_key)
    if not (dhatu_qukf_tanadi and dhatu_qukf_tanadi.gana.name == "Tanadi"):
        pytest.skip(f"{dhatu_key} (Tanadi) not found or Gana is not Tanadi. Skipping test.")

    meta = f"Dhātu: {dhatu_key} (Tanadi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_correct = "karoti"
    assert verify_form(meta, form_correct) == True

def test_qukfY_tanadi_incorrect_form(populated_dhatu_cache):
    dhatu_key = "qukf\\Y"
    dhatu_qukf_tanadi = get_dhatu_by_aupadeshika(dhatu_key)
    if not (dhatu_qukf_tanadi and dhatu_qukf_tanadi.gana.name == "Tanadi"):
        pytest.skip(f"{dhatu_key} (Tanadi) not found or Gana is not Tanadi. Skipping test.")

    meta = f"Dhātu: {dhatu_key} (Tanadi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_incorrect = "kRNoti" 
    assert verify_form(meta, form_incorrect) == False


# --- Tests for tanu~^ (TanAdi) ---
def test_tanu_tanadi_correct(populated_dhatu_cache):
    dhatu_key = "tanu~^"
    dhatu_tan = get_dhatu_by_aupadeshika(dhatu_key)
    if not (dhatu_tan and dhatu_tan.gana.name == "Tanadi"):
        pytest.skip(f"{dhatu_key} (Tanadi) not found or Gana is not Tanadi. Skipping test.")

    meta = f"Dhātu: {dhatu_key} (Tanadi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_correct = "tanoti"
    assert verify_form(meta, form_correct) == True

def test_tanu_tanadi_incorrect_form(populated_dhatu_cache):
    dhatu_key = "tanu~^"
    dhatu_tan = get_dhatu_by_aupadeshika(dhatu_key)
    if not (dhatu_tan and dhatu_tan.gana.name == "Tanadi"):
        pytest.skip(f"{dhatu_key} (Tanadi) not found or Gana is not Tanadi. Skipping test.")

    meta = f"Dhātu: {dhatu_key} (Tanadi), Lakāra: {Lakara.Lat.value}, Prayoga: {Prayoga.Kartari.value}, Puruṣa: {Purusha.Prathama.value}, Vacana: {Vacana.Eka.value}"
    form_incorrect = "tanati" 
    assert verify_form(meta, form_incorrect) == False

# Example of a form where Vidyut might produce multiple outputs (hypothetical)
# If we find such a case, we can add a test:
# meta_multi = "..."
# form_variant1 = "..."
# form_variant2 = "..."
# assert verify_form(meta_multi, form_variant1) == True
# assert verify_form(meta_multi, form_variant2) == True 