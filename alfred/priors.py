import numpy as np
from astropy.table import Table
from isochrones import SingleStarModel
from isochrones.priors import FlatPrior, GaussianPrior

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

    
    def add_split_gaussian_prior(self, mu, sigma_low, sigma_high):

        def split_gaussian_prior(x):

            x = np.array(x).reshape((-1,))

            like = np.zeros(x.shape) + np.log(np.sqrt(2/np.pi)) - np.log(sigma_low + sigma_high)

            ind = x < mu

            like[ind] += - 0.5 * ((x[ind] - mu) / sigma_low)**2

            like[~ind] += - 0.5 * ((x[~ind] - mu) / sigma_high)**2

            return like
        
        self.prior_funcs.append(split_gaussian_prior)


    def apply_priors(self, x):

        loglike = 0

        for prior in self.prior_funcs:

            loglike += np.sum(prior(x))

        return loglike
    

class AllPriors:

    def __init__(self, tab: Table, x0: dict, fit_ttv):

        self.prior_dict = {}

        searchvars = {'P': 'log(P)', 'a/rs': 'log(a/rs)', 'rhos': 'log(a/rs)', 'i': 'cos(i)', 'K': 'log(K)', 'e': 'secw', 'w': 'secw',
                      'rho_gp': 'log(rho_gp)', 'sigma_gp': 'log(sigma_gp)'}
        
        star_vars = ['mstar', 'rstar', 'rhostar', 'log10(age)', 'AV']

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

            if svar in ['log(P)','Tc'] and np.any(fit_ttv):

                if split[-1] == 'x':

                    for j in range(len(fit_ttv)):

                        if fit_ttv[j] and 'ror {0}'.format(j+1) in x0:

                            found = True

                            self.set_up_prior('{0} {1}'.format(split[0], j+1), tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

                elif fit_ttv[int(split[-1])-1] and 'ror {0}'.format(split[-1]) in x0:

                    found = True

                    self.set_up_prior(var, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])


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

        elif prior_type == 'SG':

            self.prior_dict[var].add_split_gaussian_prior(params[0], params[1], params[2])


    def apply(self, par: dict):

        log_like = 0

        for var in self.prior_dict:

            split = var.split()

            if split[0] in ['P', 'a/rs', 'K', 'rho_gp', 'sigma_gp']:

                svar = ' '.join(['log({0})'.format(split[0])]+split[1:])

                x = np.exp(par[svar])

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'age':

                x = 10**(par['age'])

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'rhos':

                svar1 = ' '.join(['log(P)']+split[1:])
                svar2 = ' '.join(['log(a/rs)']+split[1:])

                x = 0.018916375 * np.exp(par[svar2])**3 / np.exp(par[svar1])**2

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
    


def apply_star_priors(tab: Table, starmod: SingleStarModel):

    var_convert = {'mstar': lambda x: starmod.set_prior(mass = x),
                   'rstar': lambda x: starmod.set_prior(radius = x),
                   'rhostar': lambda x: starmod.set_prior(density = x),
                   'log10(age)': lambda x: starmod.set_prior(age = x),
                   'AV': lambda x: starmod.set_prior(AV = x)}

    for var in var_convert:

        i = np.where(tab['Variable'] == var)[0]

        if len(i) == 0:

            continue

        elif len(i) == 1:

            if tab['Prior Type'][i] == 'U':

                starprior = FlatPrior((tab['Param 1'][i], tab['Param 2'][i]))

            elif tab['Prior Type'][i] == 'G':
        
                starprior = GaussianPrior(tab['Param 1'][i], tab['Param 2'][i])

        elif len(i) > 2 and len(np.unique(tab['Prior Type'][i])) < 2:

            print('Cannot apply multiple priors of same type to {0}.'.format(var))
            continue

        else:

            j = np.where(tab['Prior Type'][i] == 'U')[0]
            bounds = (tab['Param 1'][i][j], tab['Param 2'][i][j])

            j = np.where(tab['Prior Type'][i] == 'G')[0]
            mean, std = tab['Param 1'][i][j], tab['Param 2'][i][j]

            starprior = GaussianPrior(mean, std, bounds = bounds)

        var_convert[var](starprior)

    return starmod