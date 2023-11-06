"""
Credit to Anthony Sottile: 
https://stackoverflow.com/questions/64925104/inspect-getsource-from-a-function-defined-in-a-string-s-def-f-return-5
"""

import os.path
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType
from typing import Any, Callable


class ShowSourceLoader:
    def __init__(self, modname: str, source: str) -> None:
        self.modname = modname
        self.source = source

    def get_source(self, modname: str) -> str:
        if modname != self.modname:
            raise ImportError(modname)
        return self.source


def make_function_from_src(s: str) -> Callable[..., Any]:
    filename = tempfile.mktemp(suffix = '.py')
    modname = os.path.splitext(os.path.basename(filename))[0]
    assert modname not in sys.modules

    # Get function name
    fn_name = s.split("def")[1].split("(")[0].strip(" ")

    # Our loader is a dummy one which just spits out our source
    loader = ShowSourceLoader(modname, s)
    spec = spec_from_loader(modname, loader, origin = filename)
    module = module_from_spec(spec)
    # The code must be compiled so the function's code object has a filename
    code = compile(s, mode = 'exec', filename = filename)
    exec(code, module.__dict__)

    # `inspect.getmodule(...)`` requires it to be in sys.modules
    sys.modules[modname] = module

    return module.__dict__[fn_name]