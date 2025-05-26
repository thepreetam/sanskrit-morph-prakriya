from vidyut.prakriya import Pada, Dhatu, Gana, Lakara, Prayoga, Purusha, Vacana
import traceback

print("Inspecting Pada.Tinanta...")
if hasattr(Pada, 'Tinanta'):
    print(f"type(Pada.Tinanta): {type(Pada.Tinanta)}")
    print(f"dir(Pada.Tinanta):")
    print(dir(Pada.Tinanta))

    # Try to instantiate it, assuming it's a class and takes keyword args
    # This part is for inspection; actual use will be in test_vidyut_prakriya.py
    print("\nAttempting to call/instantiate Pada.Tinanta (expect TypeError if args are wrong)...")
    try:
        gana_bhvadi = Gana.Bhvadi 
        lakara_lat = Lakara.Lat
        prayoga_kartari = Prayoga.Kartari
        purusha_prathama = Purusha.Prathama
        vacana_eka = Vacana.Eka
        dhatu_bu = Dhatu.mula("BU", gana_bhvadi)

        tinanta_spec = Pada.Tinanta(
            dhatu=dhatu_bu,
            lakara=lakara_lat,
            prayoga=prayoga_kartari,
            purusha=purusha_prathama,
            vacana=vacana_eka
        )
        print(f"Successfully created Pada.Tinanta object: {tinanta_spec}")
        print(f"Type of object: {type(tinanta_spec)}")
        # If this tinanta_spec is a 'Pada' instance, it should be accepted by v.derive()

    except TypeError as te:
        print(f"TypeError when calling Pada.Tinanta: {te}")
        print("This indicates the arguments or way of calling is incorrect.")
    except AttributeError as ae:
        if "Gana has no attribute" in str(ae) or \
           "Lakara has no attribute" in str(ae):
            print(f"Error accessing Gana/Lakara enum: {ae}")
        else:
            print(f"AttributeError during Pada.Tinanta attempt: {ae}")
        traceback.print_exc()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
else:
    print("Pada.Tinanta not found.") 