"""
Create registry of processes within this module.
"""
from typing import Callable
import importlib, inspect, functools, pkgutil

def register(func: Callable):
    """
    Decorator to register a function as a process. These processes will be executed by wrapping them into BIDSProcess and BIDSPipeline.
    """
    # If the function has another decorator, preserve it
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper._registered = True
    
    return wrapper
    
def get_registered_processes() -> dict:
    """
    Get all registered processes in this package, including nested modules.
    """
    registered_processes = {}

    def load_modules_recursively(package_name):
        """
        Recursively load all modules in the given package.
        """
        package = importlib.import_module(package_name)
        print(f"Package: {package_name}, Path: {package.__path__}")  # Debug print
        for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            full_module_name = f"{package_name}.{module_name.split('.')[-1]}"
            try:
                module = importlib.import_module(full_module_name)
                print(f"Loading module {full_module_name}")
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj):  # Check if the member is a function
                        if hasattr(obj, '_registered'):  # Check if the function has the '_registered' attribute
                            print(f"Found registered process: {name}")
                            registered_processes[name] = {
                                'docstring': obj.__doc__,
                                'module': obj.__module__,
                                'path': module.__file__,
                            }
                # If the module is a package, recurse into it
                if is_pkg:
                    load_modules_recursively(full_module_name)
            except Exception as e:
                print(f"Error loading module {full_module_name}: {e}")

    # Start loading from the root package
    load_modules_recursively(__name__)
    return registered_processes