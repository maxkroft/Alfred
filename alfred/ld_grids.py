from exoctk.limb_darkening import limb_darkening_fit as ldf
from svo_filters import svo
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pickle
import os

ld_grid_list = []
for fname in os.listdir(os.path.dirname(os.path.realpath(__file__))+'/ld_grids'):
    if fname[-7:] == '_grid.p':
        ld_grid_list.append(fname[:-7])


def generate_ld_grid(filter, filter_nickname):

    ldc = ldf.LDC()

    filt = svo.Filter(filter)

    Tarray = np.arange(2300, 7900, 100)
    loggarray = np.arange(3.0, 6.5, 0.5)
    feharray = np.arange(-0.5,1.0,0.5)

    u1array = []
    u2array = []
    coords = []

    for T in Tarray:
        for logg in loggarray:
            for feh in feharray:

                try:
                    ldc.calculate(T, logg, feh, 'quadratic', bandpass = filt, interp = False)
                    u1 = ldc.results['c1'][-1]
                    u2 = ldc.results['c2'][-1]
                    u1array.append(u1)
                    u2array.append(u2)

                    coords.append([T,logg,feh])

                except:
                    continue

    u1array = np.array(u1array)
    u2array = np.array(u2array)
    coords = np.array(coords)

    d = {'coords': coords, 'u1': u1array, 'u2': u2array}

    pickle.dump(d, open(os.path.dirname(os.path.realpath(__file__))+'/ld_grids/{0}_grid.p'.format(filter_nickname), 'wb'))


def load_ld_grid(filter_name: str):

    try:

        grid = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+'/ld_grids/{0}_grid.p'.format(filter_name), 'rb'))

    except FileNotFoundError:
        print('No limb darkening grid available for filter {0}.'.format(filter_name))
        print('Please use one of the following filters with limb darkening grids available:\n{0}'.format(ld_grid_list))
        print('Or run generate_ld_grid to create a new grid using a valid svo filter name and nickname {0}.'.format(filter_name))
        print('Available svo filters can be found by calling svo.filters().')
        raise

    interpu1 = LinearNDInterpolator(grid['coords'], grid['u1'])
    interpu2 = LinearNDInterpolator(grid['coords'], grid['u2'])

    return interpu1, interpu2