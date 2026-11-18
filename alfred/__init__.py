import os

__version__ = '1.3.0'

ld_grid_dir = os.path.dirname(os.path.realpath(__file__))+'/ld_grids/'
data_dir = os.path.dirname(os.path.realpath(__file__))+'/example_data/'

import importlib.util

exoctk_inst = importlib.util.find_spec("exoctk") is not None

import sys

is_notebook = False
if "IPython" in sys.modules:
    shell = sys.modules["IPython"].get_ipython().__class__.__name__
    is_notebook = shell == "ZMQInteractiveShell"


from alfred.exosystem import ExoSystem

_LAZY_IMPORTS = {
    "Init_lcs": "alfred.init_class",
    "Init_ld": "alfred.init_class",
    "Init_planets": "alfred.init_class",
    "Init_priors": "alfred.init_class",
    "Init_rv": "alfred.init_class",
    "Init_star": "alfred.init_class",
    "Init_ttvs": "alfred.init_class",
    "Init_gaia": "alfred.init_class",
    "create_folder": "alfred.init_class",
    "generate_ld_grid": "alfred.ld_grids",
    "calc_ld": "alfred.ld_grids",
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")