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

            loglike += np.sum(prior(x))

        return loglike
    

class AllPriors:

    def __init__(self, tab: Table, x0: dict):

        self.prior_dict = {}

        searchvars = {'P': 'log(P)', 'a/rs': 'log(a/rs)', 'i': 'cos(i)', 'K': 'log(K)', 'e': 'secw', 'w': 'secw',
                      'rho_gp': 'log(rho_gp)', 'sigma_gp': 'log(sigma_gp)'}
        
        star_vars = ['ms', 'rs', 'rhos', 'age', 'AV']

        for i in range(len(tab)):

            var = tab['Variable'][i]

            if var in star_vars:
                continue

            split = var.split()

            svar = split[0]
            if svar in searchvars:
                svar = searchvars[svar]

            fvar = ' '.join([svar]+split[1:])

            found = False

            if fvar in x0:

                found = True

                self.set_up_prior(var, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

            elif split[-1] == 'x':

                for y in x0:

                    if y.split()[0] == svar:

                        found = True

                        yvar = ' '.join([split[0]]+y.split()[1:])

                        self.set_up_prior(yvar, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

            if not found:
                print('Unable to apply prior to {0}, this variable is not being fit.'.format(var))


    def set_up_prior(self, var, prior_type, params):

        if var not in self.prior_dict:
            self.prior_dict[var] = Priors()

        if prior_type == 'U':

            self.prior_dict[var].add_uniform_prior(params[0], params[1])

        elif prior_type == 'G':

            self.prior_dict[var].add_gaussian_prior(params[0], params[1])


    def apply(self, par: dict):

        log_like = 0

        for var in self.prior_dict:

            split = var.split()

            if split[0] in ['P', 'a/rs', 'K', 'rho_gp', 'sigma_gp']:

                svar = ' '.join(['log({0})'.format(split[0])]+split[1:])

                x = np.exp(par[svar])

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'i':

                svar = ' '.join(['cos(i)']+split[1:])

                x = np.arccos(par[svar])

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'e':

                svar1 = ' '.join(['secw']+split[1:])
                svar2 = ' '.join(['sesw']+split[1:])

                x = par[svar1]**2 + par[svar2]**2

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'w':

                svar1 = ' '.join(['secw']+split[1:])
                svar2 = ' '.join(['sesw']+split[1:])

                x = np.arctan2(par[svar2], par[svar1])

                log_like += self.prior_dict[var].apply_priors(x)

            else:

                x = par[var]

                log_like += self.prior_dict[var].apply_priors(x)

        return log_like