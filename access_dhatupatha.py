from vidyut.prakriya import Data, DhatupathaEntry, Dhatu, Gana # Import necessary classes
import traceback
import os

# Deduce the data path based on typical Rust ProjectDirs location for macOS
# For vidyut, this is likely org.ambuda.vidyut
data_path_str = os.path.expanduser("~/Library/Application Support/org.ambuda.vidyut")
print(f"Deduced data path: {data_path_str}")

print("Inspecting vidyut.prakriya.Data...")
print("dir(Data class):")
print(dir(Data))

try:
    # How is Data used? Is it instantiated? Does it have static methods?
    # data_instance = Data() # Guessing it might not be instantiated directly
    # print("\ndir(Data instance):")
    # print(dir(data_instance))

    # More likely, Data provides access to the Dhatupatha as a property or method.
    # For example, Data.dhatupatha() or just Data.DATUPATHA if it's a loaded object.
    
    # Let's assume Vyakarana object might hold the data or provide access
    from vidyut.prakriya import Vyakarana
    v = Vyakarana()
    print("\ndir(Vyakarana instance) again to check for data accessors:")
    print(dir(v))
    # Look for attributes like 'data', 'dhatupatha', etc.

    # If Data is a namespace or has static methods:
    if hasattr(Data, 'dhatupatha'):
        print("\nData class has a 'dhatupatha' attribute/method.")
        # dhatupatha_obj = Data.dhatupatha() # if it's a method
        # print(dir(dhatupatha_obj))
    elif hasattr(Data, 'create_dhatupatha'): # From one of the Rust examples
        print("\nData class has 'create_dhatupatha'.")
        # dp = Data.create_dhatupatha() 
        # print(dp) -> this would be a list of DhatupathaEntry

    # The Rust `Dhatupatha` struct might be exposed directly in Python if we are lucky
    try:
        from vidyut.prakriya import Dhatupatha
        print("\nSuccessfully imported Dhatupatha from vidyut.prakriya")
        print("dir(Dhatupatha class):")
        print(dir(Dhatupatha))
        # If it has a constructor or a static from_path or get_default method:
        # dp = Dhatupatha() or Dhatupatha.default() or similar
        # print(dir(dp))
        # for entry in dp: print(entry) 

    except ImportError:
        print("\nCould not import Dhatupatha directly from vidyut.prakriya.")


    print("\nExploring how to get Dhatupatha entries...")
    # The Rust examples load a Dhatupatha object and iterate it.
    # Each item in iteration is a DhatupathaEntry.
    # We need to find how to get this iterable Dhatupatha in Python.

    # What if Vyakarana instance itself has the dhatupatha?
    if hasattr(v, 'dhatupatha'):
        dp = v.dhatupatha()
        print(f"Found v.dhatupatha(). Type: {type(dp)}")
        # if it's a list already:
        # for entry in dp:
        #    print(f"Dhatu: {entry.dhatu().aupadeshika()}, Gana: {entry.dhatu().gana()}")
        #    break
    else:
        # The Rust tests sometimes use `Vyakarana::new_empty().dhatupatha()`. 
        # This hints that the Vyakarana object is the source.
        # `create_dhatupatha` was in a test file, not core code. Might be test utility.
        pass


    # Let's try to iterate over something from Vyakarana or Data that might be the Dhatupatha
    # The `snapshot_tests.rs` used `Dhatupatha::from_path("data/dhatupatha.tsv")?` which returns `Dhatupatha`
    # Then it does `dhatupatha.iter().collect()`
    # The `vidyut.prakriya.DhatupathaEntry` is what we expect to get for each item.

    # The file `create_all_tinantas.rs` has:
    # fn run(dhatupatha: Dhatupatha, args: Args) -> Result<(), Box<dyn Error>> {
    # ...
    # for entry in &dhatupatha {
    #    let dhatu = entry.dhatu().clone().with_sanadi(&sanadis);
    # ...
    # }
    # fn main() {
    #    let dhatus = match Dhatupatha::from_path("data/dhatupatha.tsv") { ... }
    # ...
    # }
    # This suggests `Dhatupatha` is the main iterable. If not directly importable, perhaps v.data().dhatupatha()

    # If `Data` class itself is a container for the loaded data:
    # data_container = Data() ? or Data is static access?
    # dhatupatha_iterable = data_container.dhatupatha_entries() ?

    # The `vidyut.prakriya.Data` class might be the key if `Dhatupatha` class is not exposed.
    # `dir(Data)` was `['__class__', ..., 'config', 'dhatupatha', 'ganapatha', 'krt_pratyayas', 'links', 'nipata', 'parsed_dhatupatha', 'pratipadikas', 'purva_prayoga', 'sandhi', 'sup_pratyayas', 'sutras', 'unadis', 'upasargas', 'vrddhi_map']`
    # from a previous exploration (not in this session's output yet but I recall it or similar from PyPI examples).
    # If that `dir(Data)` is correct, then `Data.dhatupatha` or `Data.parsed_dhatupatha` could be it.
    # It seems `Data` is not meant to be instantiated but provides static access.

    if hasattr(Data, 'parsed_dhatupatha'):
        print("Found Data.parsed_dhatupatha")
        dpatha = Data.parsed_dhatupatha()
        print(f"Type of Data.parsed_dhatupatha(): {type(dpatha)}")
        if dpatha:
            print(f"Length: {len(dpatha)}")
            entry = dpatha[0]
            print(f"First entry type: {type(entry)}")
            if hasattr(entry, 'dhatu') and callable(entry.dhatu):
                 d = entry.dhatu()
                 if hasattr(d, 'aupadeshika') and callable(d.aupadeshika):
                     print(f"Entry 0: Dhatu: {d.aupadeshika()}, Gana: {d.gana()}")
                 else:
                     print("Dhatu object from entry doesn't have aupadeshika() or gana() as expected.")
            else:
                print("Entry object doesn't have dhatu() method.")
        else:
            print("Data.parsed_dhatupatha() returned empty or None.")

    print("\nAttempting to use Data(path=data_path_str) and its load_dhatu_entries() method...")

    if not os.path.exists(data_path_str):
        print(f"WARNING: Deduced data path does not exist: {data_path_str}")
        print("Vidyut data might not be downloaded or is in a different location.")
        print("Try running vidyut.download_data() if you haven't.")
        # Attempting to proceed, Data() might handle it or fail more clearly

    data_provider = Data(path=data_path_str) # Instantiate Data with the path
    print(f"Successfully instantiated Data with path. Type: {type(data_provider)}")
    
    if hasattr(data_provider, 'load_dhatu_entries') and callable(data_provider.load_dhatu_entries):
        dhatu_entries = data_provider.load_dhatu_entries()
        
        if dhatu_entries:
            print(f"\nSuccessfully loaded {len(dhatu_entries)} dhatu entries.")
            print(f"Type of the returned collection: {type(dhatu_entries)}")
            
            first_entry = dhatu_entries[0]
            print(f"Type of the first entry: {type(first_entry)}")
            
            if isinstance(first_entry, DhatupathaEntry):
                print("First entry is an instance of DhatupathaEntry.")
                print("\ndir(first_entry):")
                print(dir(first_entry))
                
                if hasattr(first_entry, 'dhatu') and callable(first_entry.dhatu):
                    dhatu_obj = first_entry.dhatu()
                    if isinstance(dhatu_obj, Dhatu):
                        if hasattr(dhatu_obj, 'aupadeshika') and callable(dhatu_obj.aupadeshika) and \
                           hasattr(dhatu_obj, 'gana') and callable(dhatu_obj.gana):
                            print(f"Entry 0: Dhatu: {dhatu_obj.aupadeshika()}, Gana: {dhatu_obj.gana()}")
                        else:
                            print("Dhatu obj lacks aupadeshika/gana methods.")
                    else:
                        print("entry.dhatu() did not return Dhatu object.")
                else:
                    print("Entry lacks .dhatu() method.")

                if hasattr(first_entry, 'number') and callable(first_entry.number):
                     print(f"Entry number: {first_entry.number()}")
                else:
                    print("Entry lacks .number() method.")
            else:
                print("First entry is NOT DhatupathaEntry.")
        else:
            print("\ndata_provider.load_dhatu_entries() returned None or an empty list.")
    else:
        print("data_provider object does not have load_dhatu_entries method.")

except FileNotFoundError as fnfe:
    print(f"\nFileNotFoundError: {fnfe}. The path {data_path_str} or files within it were not found.")
    traceback.print_exc()
except TypeError as te:
    # Catch if Data(path=...) has wrong signature or load_dhatu_entries issues
    print(f"\nTypeError: {te}")
    traceback.print_exc()
except Exception as e:
    print(f"\nAn other error occurred: {e}")
    traceback.print_exc() 