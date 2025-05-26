import vidyut
from vidyut.prakriya import Data, DhatupathaEntry, Dhatu, Gana # For testing Data instantiation
import traceback
import os
import shutil

# Define a local path for Vidyut data download root
local_download_root = "vidyut_data_root"
# The actual data for prakriya (including dhatupatha) will be in a subdirectory
prakriya_data_path = os.path.join(local_download_root, "prakriya")

print(f"Using local download root: {os.path.abspath(local_download_root)}")
print(f"Expecting prakriya data at: {os.path.abspath(prakriya_data_path)}")

# Ensure the download root directory exists
if not os.path.exists(local_download_root):
    print(f"Creating directory: {local_download_root}")
    os.makedirs(local_download_root)
else:
    print(f"Directory {local_download_root} already exists.")

print("\nAttempting to download Vidyut data to the specified local root directory...")

try:
    # download_data expects the root path where it will create subdirs like 'prakriya'
    print(f"Calling vidyut.download_data(path='{local_download_root}')...")
    vidyut.download_data(path=local_download_root)
    print(f"vidyut.download_data(path='{local_download_root}') completed.")
    
    # Verify that prakriya_data_path now exists and contains dhatupatha.tsv
    if os.path.exists(prakriya_data_path) and os.path.exists(os.path.join(prakriya_data_path, "dhatupatha.tsv")):
        print(f"Verified that '{os.path.join(prakriya_data_path, "dhatupatha.tsv")}' exists.")
    else:
        print(f"ERROR: '{os.path.join(prakriya_data_path, "dhatupatha.tsv")}' not found after download!")
        # List contents for debugging
        if os.path.exists(local_download_root):
            print(f"Contents of '{local_download_root}': {os.listdir(local_download_root)}")
        if os.path.exists(prakriya_data_path):
             print(f"Contents of '{prakriya_data_path}': {os.listdir(prakriya_data_path)}")
        raise FileNotFoundError(f"dhatupatha.tsv not found in expected location.")

    # Instantiate Data with the path to the directory containing dhatupatha.tsv (i.e., prakriya_data_path)
    print(f"\nAttempting to instantiate Data with path: {prakriya_data_path}")
    data_provider = Data(path=prakriya_data_path)
    print("Successfully instantiated Data provider.")
    
    dhatu_entries = data_provider.load_dhatu_entries()
    if dhatu_entries:
        print(f"Successfully loaded {len(dhatu_entries)} dhatu entries from '{prakriya_data_path}'.")
        first_entry = dhatu_entries[0]
        if isinstance(first_entry, DhatupathaEntry):
            dhatu_obj = first_entry.dhatu
            print(f"First dhatu entry: Code: '{first_entry.code}', Aupadeshika: '{dhatu_obj.aupadeshika}', Gana: {dhatu_obj.gana}")
            # We can now iterate through dhatu_entries for the dataset generation
            print("\nLexicon acquisition successful. Dhatus can be extracted.")
        else:
            print("First entry was not DhatupathaEntry type.")
    else:
        print("Loaded dhatu entries, but the list is empty.")

except FileNotFoundError as fnfe:
    print(f"\nFileNotFoundError: {fnfe}.")
    traceback.print_exc()
except TypeError as te:
    # This could be from Data(path=...) or download_data(path=...) if signature is wrong
    print(f"\nTypeError: {te}") 
    traceback.print_exc()
except Exception as e:
    print(f"\nAn error occurred: {e}")
    traceback.print_exc()

# Note: For Vyakarana() to use this local data, it might need to be told.
# If Vyakarana() keeps working off its default path, the Data(path=local_data_dir)
# ensures our Dhatupatha list comes from a known, consistent location.
# The plan is to generate (dhatu+metadata) -> surface_form pairs.
# We need the (dhatu+metadata) from our controlled Dhatupatha.
# Then, for ground truth, we use Vyakarana().derive() with these.
# So, Vyakarana() using its default path is fine, as long as we can iterate our own Dhatupatha. 