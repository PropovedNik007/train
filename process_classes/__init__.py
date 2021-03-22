
'''

import pathlib
ph = list(pathlib.Path(globals()['__file__']).parent.glob('*.py'))
import importlib.util

for x in ph:
    if str(x.stem) == "__init__":
        continue
    module_file_path = str(x)
    module_name = str(x.stem)
    
    module_spec = importlib.util.spec_from_file_location(
        module_name, module_file_path
    )
    
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    print(dir(module))
#__all__1 = [importlib.import_module(str(x.stem)) for x in ph if x != "__init__"]

from . import *
'''

from process_classes import *