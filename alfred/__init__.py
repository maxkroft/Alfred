import os

__version__ = '1.0.0'

ld_grid_dir = os.path.dirname(os.path.realpath(__file__))+'/ld_grids/'

from alfred.exosystem import ExoSystem
from alfred.init_class import Init_lcs, Init_ld, Init_planets, Init_priors, Init_rv, Init_star, Init_ttvs

try:
    import exoctk
    exoctk_inst = True
except:
    exoctk_inst = False

try:
    import pymultinest
    pymultinest_inst = True
except:
    pymultinest_inst = False