from svo_filters import svo
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pickle
import os

from alfred import ld_grid_dir, exoctk_inst

if exoctk_inst:
    from exoctk.limb_darkening import limb_darkening_fit as ldf

# Johnson-Cousins I and R are stored under distinct names because I_grid.p and
# i_grid.p (and R/r) are the same file on case-insensitive file systems.
_grid_file_names = {'I': 'Johnson_I', 'R': 'Johnson_R'}
_grid_filter_names = {v: k for k, v in _grid_file_names.items()}


def ld_grid_path(filter_name: str) -> str:
    """Returns the path of the pickled limb darkening grid for a filter."""
    return os.path.join(ld_grid_dir, '{0}_grid.p'.format(_grid_file_names.get(filter_name, filter_name)))


ld_grid_list = []
for fname in os.listdir(ld_grid_dir):
    if fname[-7:] == '_grid.p':
        ld_grid_list.append(_grid_filter_names.get(fname[:-7], fname[:-7]))


def generate_ld_grid(filter: str, filter_nickname: str):
    """Generate a limb darkening grid for a new filter. Saves it as a pickle in the alfred ld_grids folder. Must have exoctk installed to run this.

    Args:
        filter (str): The filter to generate the grid for. Must be one of the filters compatible with svo_filters. Run svo_filter.svo.filters() to see a
            list of allowed filters.
        filter_nickname (str): The name you want to give to this grid. This is what the pickle file will be named, and the filter name you should use
            in Init_lcs files and Init_ld files.
    """

    if not exoctk_inst:
        print('Cannot generate new limb darkening grids without installing exoctk and its data. See optional installation instructions in the docs.')
        return

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

    pickle.dump(d, open(ld_grid_path(filter_nickname), 'wb'))


def load_ld_grid(filter_name: str) -> tuple[LinearNDInterpolator, LinearNDInterpolator]:
    """Loads in a pickle file for a limb darkening grid, sets up the interpolators for each limb darkening parameters, and returns those. By default,
    Alfred includes TESS, Kepler, V, R, I, u, g, r, i, and z. To generate more, install exoctk and run generate_ld_grid.

    Args:
        filter_name (str): The name of the filter. This must be the name of the pickle file (before _grid).

    Returns:
        tuple: The interpolator objects for the two quadratic limb darkening parameters.
    """

    try:

        grid = pickle.load(open(ld_grid_path(filter_name), 'rb'))

    except FileNotFoundError:
        print('No limb darkening grid available for filter {0}.'.format(filter_name))
        print('Please use one of the following filters with limb darkening grids available:\n{0}'.format(ld_grid_list))
        print('Or run generate_ld_grid to create a new grid using a valid svo filter name and nickname {0}.'.format(filter_name))
        print('Note that this requires installation of exoctk and its data. See optional installation instructions in the docs.')
        print('Available svo filters can be found by calling svo_filters.svo.filters().')
        raise

    interpu1 = LinearNDInterpolator(grid['coords'], grid['u1'])
    interpu2 = LinearNDInterpolator(grid['coords'], grid['u2'])

    return interpu1, interpu2


def calc_ld(filter_name: str, T: np.typing.ArrayLike, logg: np.typing.ArrayLike, feh: np.typing.ArrayLike) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Calculates limb darkening parameters from a grid given the input stellar parameters.

    Args:
        filter_name (str): The name of the filter to generate the limb darkening parameters in. Must have an existing grid in alfred. By default, Alfred 
            includes TESS, Kepler, V, R, I, u, g, r, i, and z. More can be generated with exoctk installed and by running generate_ld_grid.
        T (ArrayLike): The stellar effective temperature in K.
        logg (ArrayLike): The log10 of the stellar surface gravity in cm/s^2.
        feh (ArrayLike): Stellar metallicity in dex.

    Returns:
        tuple: The two quadratic limb darkening parameters.
    """

    interpu1, interpu2 = load_ld_grid(filter_name)

    interp_input = np.array([T, logg, feh]).T

    return interpu1(interp_input), interpu2(interp_input)