import numpy as np

from alfred import ld_grids


def test_calc_ld():
    """Tests to make sure included limb darkening grids are loading in properly and interpolating.
    """

    u1, u2 = ld_grids.calc_ld('TESS', 5809, 4.54, 0.01)

    assert np.isclose(u1[0], 0.3628, rtol = 1e-3)
    assert np.isclose(u2[0], 0.16671, rtol = 1e-3)

    u1, u2 = ld_grids.calc_ld('Kepler', [5809, 4976], [4.54,4.3], [0.01,-0.13])

    assert np.isclose(u1, np.array([0.46873, 0.59988]), rtol = 1e-3).all()
    assert np.isclose(u2, np.array([0.15868, 0.08404]), rtol = 1e-3).all()