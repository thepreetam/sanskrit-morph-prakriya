from vidyut.prakriya import DhatuEntry

print("Methods and attributes of the DhatuEntry class:")
print(dir(DhatuEntry))

# Try to see if it can be instantiated and what its instance looks like
# This will likely fail if we don't know the constructor args
try:
    entry_instance = DhatuEntry()
    print("\nMethods and attributes of a DhatuEntry instance (default init):")
    print(dir(entry_instance))
except TypeError as te:
    print(f"\nError instantiating DhatuEntry with no args: {te}")
except Exception as e:
    print(f"\nAn error occurred with DhatuEntry: {e}") 