import os
import pkgutil
import importlib

package_dir = os.path.dirname(__file__)

for finder, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if not is_pkg and module_name not in ("__init__", "base"):
        importlib.import_module(f"{__name__}.{module_name}")