import os

__version__ = '1.1.5'

ld_grid_dir = os.path.dirname(os.path.realpath(__file__))+'/ld_grids/'
data_dir = os.path.dirname(os.path.realpath(__file__))+'/example_data/'

try:
    import exoctk.limb_darkening
    exoctk_inst = True
except:
    exoctk_inst = False


try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    
    if shell == 'ZMQInteractiveShell':
        is_notebook = True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
        is_notebook = False  # Terminal running IPython
    else:
        is_notebook = False  # Other IDEs / standard Python interpreter
except (NameError, ImportError):
    is_notebook = False      # IPython not installed or standard Python interpre


from alfred.exosystem import ExoSystem
from alfred.init_class import Init_lcs, Init_ld, Init_planets, Init_priors, Init_rv, Init_star, Init_ttvs, create_folder
from alfred.ld_grids import generate_ld_grid, calc_ld
