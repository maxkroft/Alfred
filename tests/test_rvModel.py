import numpy as np

from alfred._rv_func import _rvModel


def test_rvModel():
    """Test to make sure _rvModel c/cython function is compiled and works.
    """

    par = np.array([5.56, 100, 11, 0.23, 1.2], dtype = float)
    t = np.array([99,100,101], dtype = float)

    rv = _rvModel(par, t)

    assert np.isclose(rv, np.array([11.82788683, 0.91676512, -10.00130897]), atol = 1e-5).all()