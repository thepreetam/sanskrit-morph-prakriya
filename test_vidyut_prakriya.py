from vidyut.prakriya import Vyakarana, Dhatu, Gana, Lakara, Prayoga, Purusha, Vacana, Pada, Data
import traceback
import os

# Setup to get Dhatu object by aupadeshika string
DHATU_CACHE = {}
prakriya_data_path = os.path.join("vidyut_data_root", "prakriya")

def get_dhatu_by_aupadeshika_for_test(aupadeshika_str):
    if not DHATU_CACHE:
        data_provider = Data(path=prakriya_data_path)
        all_entries = data_provider.load_dhatu_entries()
        for entry in all_entries:
            DHATU_CACHE[entry.dhatu.aupadeshika] = entry.dhatu
    return DHATU_CACHE.get(aupadeshika_str)

def run_test():
    # Initialize Vyakarana
    v = Vyakarana()

    # Test case: kf\Y (SvAdi) + Lat + Kartari + Prathama + Eka
    print("Testing: kf\\Y (SvAdi) + Lat + Kartari + Prathama + Eka")
    dhatu_key_kf_svadi = "kf\\Y"
    dhatu_obj_kf_svadi = get_dhatu_by_aupadeshika_for_test(dhatu_key_kf_svadi)

    if dhatu_obj_kf_svadi and dhatu_obj_kf_svadi.gana.name == "Svadi":
        print(f"Found Dhatu: {dhatu_obj_kf_svadi.aupadeshika}, Gana: {dhatu_obj_kf_svadi.gana.name}")
        tinanta_args_kf_svadi = Pada.Tinanta(
            dhatu=dhatu_obj_kf_svadi,
            lakara=Lakara.Lat,
            prayoga=Prayoga.Kartari,
            purusha=Purusha.Prathama,
            vacana=Vacana.Eka
        )
        prakriyas_kf_svadi = v.derive(tinanta_args_kf_svadi)

        if prakriyas_kf_svadi:
            print(f"Derived forms for {dhatu_key_kf_svadi} (Svadi) Lat Pra Eka:")
            for p in prakriyas_kf_svadi:
                print(f"  Form: {p.text}")
            # print("  Steps:")
            # for step in p.history:
            #     # Ensure attributes are accessed correctly for Step object as figured out previously
            #     results_str = ", ".join(step.result)
            #     print(f"    {step.code}: [{results_str}]")
        else:
            print(f"No forms derived for {dhatu_key_kf_svadi} (Svadi) Lat Pra Eka.")
    else:
        print(f"Could not find Dhatu {dhatu_key_kf_svadi} with Gana Svadi for testing.")

    # --- Test BU -> bhavati ---
    print("\nTesting: BU -> bhavati")
    try:
        # Access enum members
        gana_bhvadi = Gana.Bhvadi 
        lakara_lat = Lakara.Lat
        prayoga_kartari = Prayoga.Kartari
        purusha_prathama = Purusha.Prathama
        vacana_eka = Vacana.Eka

        # Create Dhatu object
        dhatu_bu = Dhatu.mula("BU", gana_bhvadi)

        # Create Tinanta arguments object using Pada.Tinanta
        tinanta_args_bhu = Pada.Tinanta(
            dhatu=dhatu_bu,
            lakara=lakara_lat,
            prayoga=prayoga_kartari,
            purusha=purusha_prathama,
            vacana=vacana_eka
        )

        # Call v.derive() with the Tinanta arguments object
        prakriyas_bhu = v.derive(tinanta_args_bhu)

        if prakriyas_bhu:
            print(f"Successfully derived forms for BU!")
            for p in prakriyas_bhu:
                # Displaying the enum values themselves for clarity in output
                print(f"Input: Dhatu: BU (Gana: {gana_bhvadi}), Lakara: {lakara_lat}, Prayoga: {prayoga_kartari}, Purusha: {purusha_prathama}, Vacana: {vacana_eka}")
                print(f"Output: {p.text}")
                history_output = []
                for step in p.history:
                    result_terms = [t for t in step.result if t] # t is already a string
                    result_str = " + ".join(result_terms)
                    history_output.append(f"{step.code:<10} | {result_str}")
                print("\n".join(history_output))
                print("---")
        else:
            print("No forms derived for BU (call succeeded but returned empty list).")

    except AttributeError as ae:
        print(f"AttributeError: {ae}. This might indicate an issue with enum access (e.g., Gana.Bhvadi) or method names.")
        traceback.print_exc()
    except ValueError as ve:
        print(f"ValueError: {ve}. This might be an SLP1 string issue for Dhatu.mula or invalid enum value.")
        traceback.print_exc()
    except TypeError as te:
        print(f"TypeError: {te}. This likely means the arguments to Pada.Tinanta() or v.derive() are incorrect.")
        traceback.print_exc()
    except Exception as e:
        print(f"Generic error deriving for BU: {e}")
        traceback.print_exc()

    # --- Test gam -> gacchati ---
    print("\nTesting: gam -> gacchati")
    try:
        # Reusing enums from above (gana_bhvadi, lakara_lat, etc.)
        # Assuming gam (SLP1) is also Bhvadi for this test case as per plan example context
        dhatu_gam = Dhatu.mula("gam", gana_bhvadi)

        tinanta_args_gam = Pada.Tinanta(
            dhatu=dhatu_gam,
            lakara=lakara_lat,
            prayoga=prayoga_kartari,
            purusha=purusha_prathama,
            vacana=vacana_eka
        )

        prakriyas_gam = v.derive(tinanta_args_gam)

        if prakriyas_gam:
            print(f"Successfully derived forms for gam!")
            for p in prakriyas_gam:
                print(f"Input: Dhatu: gam (Gana: {gana_bhvadi}), Lakara: {lakara_lat}, Prayoga: {prayoga_kartari}, Purusha: {purusha_prathama}, Vacana: {vacana_eka}")
                print(f"Output: {p.text}")
                history_output = []
                for step in p.history:
                    result_terms = [t for t in step.result if t] # t is already a string
                    result_str = " + ".join(result_terms)
                    history_output.append(f"{step.code:<10} | {result_str}")
                print("\n".join(history_output))
                print("---")
        else:
            print("No forms derived for gam (call succeeded but returned empty list).")
            
    except AttributeError as ae:
        print(f"AttributeError: {ae}.")
        traceback.print_exc()
    except ValueError as ve:
        print(f"ValueError: {ve}.")
        traceback.print_exc()
    except TypeError as te:
        print(f"TypeError: {te}.")
        traceback.print_exc()
    except Exception as e:
        print(f"Generic error deriving for gam: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test() 