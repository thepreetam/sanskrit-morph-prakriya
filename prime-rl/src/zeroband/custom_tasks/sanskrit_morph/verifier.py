from vidyut.prakriya import Prakriya
import traceback

# Mappings from English terms to Sanskrit terms used by vidyut
# These should be verified against the specific vidyut version and expectations.
GENDER_MAP = {"masculine": "पुंल्लिङ्ग", "feminine": "स्त्रीलिङ्ग", "neuter": "नपुंसकलिङ्ग"}
CASE_MAP = {
    "nominative": "प्रथमा", "accusative": "द्वितीया", "instrumental": "तृतीया",
    "dative": "चतुर्थी", "ablative": "पञ्चमी", "genitive": "षष्ठी",
    "locative": "सप्तमी", "vocative": "सम्बोधन प्रथमा" # Or sometimes just "सम्बोधनम्"
}
NUMBER_MAP = {"singular": "एकवचन", "dual": "द्विवचन", "plural": "बहुवचन"}

def verify_sanskrit_form(metadata_str: str, generated_form: str) -> bool:
    """
    Verifies if the generated Sanskrit form matches the given metadata
    using vidyut-prakriya.
    Example metadata_str: "madhupa#noun#masculine#instrumental#singular"
    """
    # print(f"Attempting to verify: Metadata='{metadata_str}', Generated='{generated_form}'")
    try:
        parts = metadata_str.split('#')
        if len(parts) != 5:
            print(f"Error: Invalid metadata format: '{metadata_str}'. Expected 5 parts, got {len(parts)}.")
            return False

        lemma, pos_str, gender_str_en, case_str_en, number_str_en = parts

        # Currently, this verifier is tailored for nominals ('noun') based on dataset structure.
        if pos_str.lower() != "noun":
            print(f"Warning: Verifier currently only supports 'noun' POS. Got '{pos_str}'. Treating as incorrect.")
            return False

        # Validate and map English terms to Sanskrit terms
        if gender_str_en not in GENDER_MAP:
            print(f"Error: Unmapped gender term: '{gender_str_en}' in metadata '{metadata_str}'")
            return False
        if case_str_en not in CASE_MAP:
            print(f"Error: Unmapped case term: '{case_str_en}' in metadata '{metadata_str}'")
            return False
        if number_str_en not in NUMBER_MAP:
            print(f"Error: Unmapped number term: '{number_str_en}' in metadata '{metadata_str}'")
            return False

        gender_skt = GENDER_MAP[gender_str_en]
        case_skt = CASE_MAP[case_str_en]
        number_skt = NUMBER_MAP[number_str_en]

        # Build the prakriya object
        p_builder = Prakriya.new_builder().lemma(lemma)
        p_builder = p_builder.linga(gender_skt)
        p_builder = p_builder.vibhakti(case_skt)
        p_builder = p_builder.vacana(number_skt)
        
        # Assuming 'build_sup()' for nominal (subanta) forms
        prakriya_obj = p_builder.build_sup() 
            
        valid_forms = [form.text for form in prakriya_obj.forms()]

        if not valid_forms:
            if not generated_form:
                # No forms expected by vidyut, and none generated. This might be for impossible metadata.
                # print(f"Info: No valid forms from vidyut and empty generated form for '{metadata_str}'. Considered correct.")
                return True # Or False, if empty generation for impossible metadata is penalized
            else:
                # No forms expected by vidyut, but something was generated.
                # print(f"Verification failed: No forms from vidyut for '{metadata_str}', but got '{generated_form}'. Valid: {valid_forms}")
                return False
        
        is_correct = generated_form in valid_forms
        # if is_correct:
        #     print(f"Verification success: '{generated_form}' is valid for '{metadata_str}'. Valid: {valid_forms}")
        # else:
        #     print(f"Verification failed: '{generated_form}' not in {valid_forms} for '{metadata_str}'")
        return is_correct

    except Exception:
        print(f"Error during Sanskrit verification for metadata '{metadata_str}', generated_form '{generated_form}':\n{traceback.format_exc()}")
        return False 