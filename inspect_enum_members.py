from vidyut.prakriya import Lakara

print("dir(Lakara):")
print(dir(Lakara))

print("\nLakara members (if __members__ exists):")
if hasattr(Lakara, '__members__'):
    print(Lakara.__members__)
else:
    print("Lakara does not have __members__ attribute.")

# Also, check if they are simply attributes
print("\nPotential enum members as attributes:")
for attr_name in dir(Lakara):
    if not attr_name.startswith('_'):
        attr_value = getattr(Lakara, attr_name)
        # Check if it's an instance of Lakara itself (or its type)
        if isinstance(attr_value, type(Lakara.Lat)) or type(attr_value) == type(Lakara):
             print(f"  {attr_name}: {attr_value}")

print("\nInspecting Lakara.choices:")
if hasattr(Lakara, 'choices') and callable(Lakara.choices):
    choices_list = Lakara.choices()
    print(f"Lakara.choices(): {choices_list}")
    print(f"type(Lakara.choices()): {type(choices_list)}")
    if choices_list:
        print(f"type of first element in Lakara.choices(): {type(choices_list[0])}")
        print(f"First element: {choices_list[0]}")

lat_enum = Lakara.Lat

print(f"Lakara.Lat: {lat_enum}")
print(f"Lakara.Lat.name: {lat_enum.name}")
print(f"Lakara.Lat.value: {lat_enum.value}") # _value_ was in dir() output

# Test from_string with various inputs
test_strings = ["Lat", "lat", "la~w", lat_enum.name, str(lat_enum.value)]
print("\nTesting Lakara.from_string():")
for s in test_strings:
    try:
        parsed_enum = Lakara.from_string(s)
        print(f"  Lakara.from_string('{s}') -> {parsed_enum} (Success)")
    except ValueError as e:
        print(f"  Lakara.from_string('{s}') -> Error: {e}") 