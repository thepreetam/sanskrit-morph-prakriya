import vidyut
from vidyut.prakriya import Vyakarana, Data
import traceback

print("Inspecting vidyut module for data path clues...")
print("dir(vidyut):")
print(dir(vidyut))

# Common names for data path functions or variables
path_vars = ['get_data_path', 'data_path', 'DATA_PATH', 'default_data_path', 'get_default_data_path']
for pv in path_vars:
    if hasattr(vidyut, pv):
        print(f"Found vidyut.{pv}")
        # try:
        #     path_val = getattr(vidyut, pv)
        #     if callable(path_val):
        #         print(f"vidyut.{pv}() result: {path_val()}")
        #     else:
        #         print(f"vidyut.{pv} value: {path_val}")
        # except Exception as e:
        #     print(f"Error accessing or calling vidyut.{pv}: {e}")

print("\nInspecting Vyakarana instance for data path...")
try:
    v = Vyakarana()
    print("dir(v - Vyakarana instance):")
    print(dir(v))
    # Look for attributes like _data_path, config.data_path etc.
    # For example, if v has a _data_object that is an instance of Data, and Data stores path.
    if hasattr(v, 'data'): # if v has a general data accessor
        print("v.data exists")
        print(dir(v.data))
        if hasattr(v.data, 'path'):
             print(f"v.data.path = {v.data.path}")

except Exception as e:
    print(f"Error instantiating Vyakarana or accessing its members: {e}")
    traceback.print_exc()

print("\nTrying to find default path for Data(path=...) if Vyakarana uses one implicitly")
# This is more complex; Vyakarana likely uses an internal mechanism or a known default location.
# PyO3/Rust bindings often use `platform_dirs` or similar to get user data directories.
# If vidyut.download_data() was called, it placed data somewhere. That's the path we need.

# An alternative from the Rust side is `Config::from_env()` or `Config::default()`
# which then has `Config::data_path()`. Let's see if `vidyut.config` exists.
if hasattr(vidyut, 'config'):
    print("Found vidyut.config module/object")
    print(dir(vidyut.config))
    # if vidyut.config.Config exists:
    #    cfg = vidyut.config.Config.from_env() or vidyut.config.Config.default()
    #    data_p = cfg.data_path()

# The original C++ version of related tools often used environment variables like VIDYUT_DATA_DIR
# or XDG_DATA_HOME.

# For now, the most direct path is to find how Vyakarana() gets its data implicitly.
# The Rust code for Vyakarana::new() likely calls something like Config::from_env().unwrap().data().
# The `Data` constructor in Rust `pub fn new(path: &Path) -> Result<Self>`
# So, Vyakarana::new() must determine this `path`.

# One of the Rust files (create_all_tinantas.rs) had:
# `let dhatus = match Dhatupatha::from_path("data/dhatupatha.tsv")`
# This was when run as an example *within the vidyut crate source tree*.
# When used as a library, it must find the installed data path.

# The file `vidyut/src/lib.rs` in the vidyut Rust crate will have `init_dirs()`
# which often uses `directories::ProjectDirs::from("org", "ambuda", "vidyut")`
# to get `data_dir()`.

print("\nIf no easy way to get the path, we might need to deduce it from standard user data locations.")
print("For macOS, it might be ~/Library/Application Support/org.ambuda.vidyut/") 