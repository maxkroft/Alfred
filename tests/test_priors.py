from astropy.table import Table
import numpy as np
import pytest

from alfred import priors

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_all_priors():
    """Tests initializing an AllPriors object, ensuring that the correct prior functions and fixed values are stored. Tests applying the priors to
    get a log likelihood. Tests retrieving bounds from variables and transforming those bounds to new variables. Tests Priors object functionality
    in the process.
    """

    tab = Table({'Variable': ['P 1', 'P 1', 'e 1', 'w 1', 'F0 x'],
                 'Prior Type': ['G', 'U', 'U', 'U', 'F'],
                 'Param 1': [6.6, 0.0, 0.0, -np.pi, 1.0],
                 'Param 2': [0.5, 100.0, 0.3, -np.pi/6, 0.0]})
    
    x0 = {'log(P) 1': np.log(6), 'secw 1': 0.1, 'sesw 1': -0.2, 'F0 1': 1.0, 'F0 2': 1.0}

    fit_ttv = np.array([False])

    allpriors = priors.AllPriors(tab, x0, fit_ttv)

    assert len(allpriors.prior_dict) == 3
    assert 'P 1' in allpriors.prior_dict
    assert len(allpriors.prior_dict['P 1'].prior_funcs) == 2
    assert 'e 1' in allpriors.prior_dict
    assert 'w 1' in allpriors.prior_dict
    assert len(allpriors.fixed) == 2
    assert allpriors.fixed['F0 1'] == 1.0
    assert allpriors.fixed['F0 2'] == 1.0

    loglike = allpriors.apply(x0)

    assert np.isclose(loglike, -0.94579135, atol = 1e-5)

    l, u = allpriors.get_bounds('log(P) 1')
    
    assert np.isneginf(l)
    assert np.isclose(u, np.log(100), atol = 1e-5)

    l, u = allpriors.get_bounds('secw 1')

    assert np.isclose(l, -np.sqrt(0.3), rtol = 1e-4)
    assert np.isclose(u, np.sqrt(0.3)*np.cos(-np.pi/6), rtol = 1e-4)

    l, u = allpriors.get_bounds('sesw 1')

    assert np.isclose(l, -np.sqrt(0.3), rtol = 1e-4)
    assert np.isclose(u, 0, atol = 1e-4)
