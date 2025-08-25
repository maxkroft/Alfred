import numpy as np
from astropy.table import Table

class Priors:

    def __init__(self):

        self.prior_funcs = []


    def add_gaussian_prior(self, mean, std):

        def gaussian_prior(x):

            return - 0.5 * ((x - mean) / std)**2 - np.log( np.sqrt(2 * np.pi) * std)
        
        self.prior_funcs.append(gaussian_prior)

    
    def add_uniform_prior(self, lower, upper):

        def uniform_prior(x):

            if not lower <= x <= upper:

                return -np.inf
            
            return 0
        
        self.prior_funcs.append(uniform_prior)


    def apply_priors(self, x):

        loglike = 0

        for prior in self.prior_funcs:

            loglike += prior(x)

        return loglike
    

class AllPriors:

    def __init__(self, tab: Table, x0: dict):

        self.prior_dict = {}

        searchvars = {'P': 'log(P)', 'a/r': 'log(a/rs)', 'i': 'cos(i)', 'K': 'log(K)', 'e': 'secw', 'w': 'secw',
                      'rho_gp': 'log(rho_gp)', 'sigma_gp': 'log(sigma_gp)'}
        
        star_vars = ['ms', 'rs', 'rhos', 'age', 'AV']

        for i in range(len(tab)):

            var = tab['Variable'][i]

            if var in star_vars:
                continue

            svar = var.split()[0]
            if svar in searchvars:
                svar = searchvars[svar]

            fvar = ' '.join([svar]+var.split[1:])