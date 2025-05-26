from vidyut.prakriya import Vyakarana, Dhatu, Gana, Lakara, Prayoga, Purusha, Vacana, Pada, Step
import traceback

print("Inspecting Step class and an instance if possible...")
print("dir(Step):")
print(dir(Step))

# To inspect an instance, we need to successfully get one from a derivation.
# We'll try a minimal derivation and then print dir(step_instance)

v = Vyakarana()
try:
    gana_bhvadi = Gana.Bhvadi 
    lakara_lat = Lakara.Lat
    prayoga_kartari = Prayoga.Kartari
    purusha_prathama = Purusha.Prathama
    vacana_eka = Vacana.Eka
    dhatu_bu = Dhatu.mula("BU", gana_bhvadi)
    tinanta_args_bhu = Pada.Tinanta(
        dhatu=dhatu_bu,
        lakara=lakara_lat,
        prayoga=prayoga_kartari,
        purusha=purusha_prathama,
        vacana=vacana_eka
    )
    prakriyas_bhu = v.derive(tinanta_args_bhu)
    if prakriyas_bhu and prakriyas_bhu[0].history:
        step_instance = prakriyas_bhu[0].history[0]
        print("\ndir(step_instance):")
        print(dir(step_instance))

        # Try to access rule and code based on Rust example step.rule().code()
        if hasattr(step_instance, 'rule'):
            rule_obj = step_instance.rule()
            print("\ndir(rule_obj) from step_instance.rule():")
            print(dir(rule_obj))
            if hasattr(rule_obj, 'code'):
                if callable(rule_obj.code):
                    print(f"rule_obj.code() exists: {rule_obj.code()}")
                else:
                    print(f"rule_obj.code attribute exists: {rule_obj.code}")
            else:
                print("rule_obj does not have .code")
        else:
            print("step_instance does not have .rule() method or .rule attribute")

    else:
        print("\nCould not get a step instance for inspection.")

except Exception as e:
    print(f"\nError during Step inspection setup: {e}")
    traceback.print_exc() 