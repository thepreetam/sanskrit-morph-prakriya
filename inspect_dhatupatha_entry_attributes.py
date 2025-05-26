import vidyut
from vidyut.prakriya import Data, DhatupathaEntry
import os

local_download_root = "vidyut_data_root"
prakriya_data_path = os.path.join(local_download_root, "prakriya")

if not os.path.exists(os.path.join(prakriya_data_path, "dhatupatha.tsv")):
    print(f"dhatupatha.tsv not found in {prakriya_data_path}. Please run download_vidyut_data.py first.")
else:
    data_provider = Data(path=prakriya_data_path)
    dhatu_entries = data_provider.load_dhatu_entries()
    if dhatu_entries:
        first_entry = dhatu_entries[0]
        print(f"Type of first_entry: {type(first_entry)}")
        print("dir(first_entry):")
        print(dir(first_entry))
        # Try to access common attributes or methods that might store an ID or number
        for attr in ['id', 'number', 'num', 'code', 'value', 'patha_id', 'entry_id']:
            if hasattr(first_entry, attr):
                print(f"Found attribute '{attr}': {getattr(first_entry, attr)}")
    else:
        print("No dhatu entries found.") 