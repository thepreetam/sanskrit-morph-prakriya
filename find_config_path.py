import vidyut
import traceback

print("Inspecting the top-level vidyut module and its submodule `vidyut.vidyut` for Config or path functions.")

print("\ndir(vidyut):")
print(dir(vidyut))

if hasattr(vidyut, 'vidyut') and hasattr(vidyut.vidyut, '__loader__'): # Check if it's a submodule
    print("\ndir(vidyut.vidyut) submodule:")
    print(dir(vidyut.vidyut))
    
    # Look for Config class or get_data_path type functions in vidyut.vidyut
    if hasattr(vidyut.vidyut, 'Config'):
        print("\nFound vidyut.vidyut.Config")
        print("dir(vidyut.vidyut.Config):")
        print(dir(vidyut.vidyut.Config))
        
        # Try to use Config.default().data_path() or similar based on Rust patterns
        try:
            # Assuming Config has a default() or from_env() static method
            # and the instance has a data_path() method or property.
            cfg_default = None
            if hasattr(vidyut.vidyut.Config, 'default') and callable(vidyut.vidyut.Config.default):
                cfg_default = vidyut.vidyut.Config.default()
                print("Called vidyut.vidyut.Config.default()")
            elif hasattr(vidyut.vidyut.Config, 'from_env') and callable(vidyut.vidyut.Config.from_env):
                cfg_default = vidyut.vidyut.Config.from_env()
                print("Called vidyut.vidyut.Config.from_env()")
            
            if cfg_default:
                if hasattr(cfg_default, 'data_path') and callable(cfg_default.data_path):
                    data_p = cfg_default.data_path()
                    print(f"Path from Config.default().data_path(): {data_p}")
                    # This is the path we need for Data(path=data_p)
                elif hasattr(cfg_default, 'data_path'): # if it's an attribute
                     data_p = cfg_default.data_path
                     print(f"Path from Config.default().data_path attribute: {data_p}")
                else:
                    print("Config object from default/from_env does not have .data_path()")
            else:
                print("Could not get a default Config instance.")
        except Exception as e_cfg:
            print(f"Error trying to use vidyut.vidyut.Config: {e_cfg}")
            traceback.print_exc()
    else:
        print("\nvidyut.vidyut.Config not found.")
else:
    print("\nvidyut.vidyut submodule not found or not a module.") 