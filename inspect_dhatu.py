from vidyut.prakriya import Dhatu, Gana

# Assuming Gana.Bhvadi works based on previous non-failure
try:
    gana_bhvadi = Gana.Bhvadi
    dhatu_instance = Dhatu.mula("BU", gana_bhvadi)
    print(f"Methods and attributes of a Dhatu instance (from Dhatu.mula):")
    print(dir(dhatu_instance))
except AttributeError as e:
    print(f"Failed to create Dhatu instance or Gana.Bhvadi not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print("\nAttributes of the Dhatu class itself:")
print(dir(Dhatu)) 