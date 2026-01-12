import numpy as np
from astropy.table import Table
from isochrones import SingleStarModel
from isochrones.priors import FlatPrior, GaussianPrior


class Priors:
    """Class for holding prior functions for a specific variable and applying them.
    """

    def __init__(self):
        """Initializes a Priors object with a list to hold functions and default lower and upper bounds on the variable of negative and positive
        infinity, respectively.
        """

        self.prior_funcs = []
        self.lbound = -np.inf
        self.ubound = np.inf


    def add_gaussian_prior(self, mean: float, std: float):
        """Adds a Gaussian prior function to the list of priors with the given parameters. The prior function returns the log of the prior value for
        a given trial value to be added to the log likelihood during fitting.

        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
        """

        def gaussian_prior(x):

            return - 0.5 * ((x - mean) / std)**2 - np.log( np.sqrt(2 * np.pi) * std)
        
        self.prior_funcs.append(gaussian_prior)

    
    def add_uniform_prior(self, lower: float, upper: float):
        """Adds a uniform prior function to the list of priors with the given bounds. The prior function returns negative infinity if the given trial
        value is out of the bounds, otherwise 0. Updates the lower and upper bounds of the Priors object.
        """

        self.lbound = max(self.lbound, lower)
        self.ubound = min(self.ubound, upper)

        def uniform_prior(x):

            if not lower <= x <= upper:

                return -np.inf
            
            return 0
        
        self.prior_funcs.append(uniform_prior)


    def apply_priors(self, x: float) -> float:
        """Applies all priors in the list of functions to the given trial value, and returns the log likelihood value to be added during fitting.

        Args:
            x (float): The value to apply the priors to.

        Returns:
            float: The log likelihood value from the priors.
        """

        loglike = 0

        for prior in self.prior_funcs:

            loglike += np.sum(prior(x))

        return loglike
    

    def bounds(self) -> tuple[float, float]:
        """Retrieves the boundaries on this parameter set by any priors.
        
        Returns:
            tuple: The lower and upper boundaries of this variable given the set of priors stored.
        """

        return self.lbound, self.ubound
    

class AllPriors:
    """A class for parsing the Init_priors tables into actual priors, and applying them to all parameters. Stores a dict of Priors objects for each
    parameter and a dict of all fixed parameters.
    """

    def __init__(self, tab: Table, x0: dict, fit_ttv: np.typing.NDArray):
        """Initializes an AllPriors object from an Init_priors table and a list of fit parameters. Parses the table into actual priors for any relevant
        parameters. Sets up a dict of fixed parameters and their values.

        Args:
            tab (Table): The table from an Init_priors object with a list of priors to set up.
            x0 (dict): A dict with initial values for all fit parameters from an ExoSystem fit, used to determine which variables are being fit for and
                require priors to be set up.
            fit_ttv (ndarray): An array of bools for each planet. They represent whether or not each planet is being fit for TTVs. Priors can be placed
            on P and Tc for a planet even if they are fit for TTVs and don't directly have those in the fit parameters.
        """

        self.prior_dict = {}
        self.fixed = {}

        searchvars = {'P': 'log(P)', 'a/rs': 'log(a/rs)', 'rhos': 'log(a/rs)', 'i': 'cos(i)', 'K': 'log(K)', 'e': 'secw', 'w': 'secw',
                      'rho_gp': 'log(rho_gp)', 'sigma_gp': 'log(sigma_gp)'}
        
        star_vars = ['eep', 'log10(age)', 'feh', 'distance', 'AV', 'mstar', 'rstar', 'rhostar']

        for i in range(len(tab)):

            var = tab['Variable'][i]

            if var in star_vars:

                if var in star_vars[:5] and tab['Prior Type'][i] == 'F':
                    
                    self.fixed[var] = tab['Param 1'][i]

                continue

            split = var.split()

            svar = split[0]
            if svar in searchvars:
                svar = searchvars[svar]

            fvar = ' '.join([svar]+split[1:])

            found = False

            if svar in ['log(P)','Tc'] and np.any(fit_ttv):

                if tab['Prior Type'][i] == 'F':

                    continue

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


        keys = list(self.fixed.keys())
        for f in keys:

            if f in self.prior_dict:

                self.prior_dict.pop(f)

            split = f.split()

            if split[0] in ['P', 'a/rs', 'K', 'rho_gp', 'sigma_gp']:

                sf = ' '.join(['log({0})'.format(split[0])]+split[1:])

                self.fixed[sf] = np.log(self.fixed.pop(f))

            elif split[0] == 'age':

                self.fixed['log10(age)'] = np.log10(self.fixed.pop(f))

            elif split[0] == 'i':

                sf = ' '.join(['cos(i)']+split[1:])

                self.fixed[sf] = np.cos(self.fixed.pop(f))



    def set_up_prior(self, var: str, prior_type: str, params: list):
        """Sets up a Priors object for a parameter in the dict of Priors, if one doesn't exist already. Then adds the given prior to that object with
        the given parameters.

        Args:
            var (str): The parameter the prior is for.
            prior_type (str): The type of prior to set up.
            params (list): A list of the parameters required to set up the prior (e.g. mean and std for a Gaussian prior).
        """

        if var not in self.prior_dict:
            self.prior_dict[var] = Priors()

        if prior_type == 'U':

            self.prior_dict[var].add_uniform_prior(params[0], params[1])

        elif prior_type == 'G':

            self.prior_dict[var].add_gaussian_prior(params[0], params[1])

        elif prior_type == 'F':

            self.fixed[var] = params[0]


    def apply(self, par: dict) -> float:
        """Applies all priors to all fit parameters and returns the total log likelihood.

        Args:
            par (dict): A dict of the fit parameters from the ExoSystem fit.

        Returns:
            float: The total log likelihood from the priors.
        """

        log_like = 0

        for var in self.prior_dict:

            split = var.split()

            if split[0] in ['P', 'a/rs', 'K', 'rho_gp', 'sigma_gp']:

                svar = ' '.join(['log({0})'.format(split[0])]+split[1:])

                x = np.exp(par[svar])

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'age':

                x = 10**(par['log10(age)'])

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
    

    def get_bounds(self, var: str) -> tuple[float, float]:
        """Gets the bounds on a fit parameter, if there are any from uniform priors. Used for setting up truncated Gaussians when
        initializing the MCMC chains to ensure they do not leave any bounds. Combines bounds from any variations on the actual fitted parameter, for
        instance P to log(P) or e to secw.

        Args:
            var (str): The parameter to get the bounds for.

        Returns:
            tuple: The lower and upper bounds on this fit parameter.
        """

        logvars = {'log(P)':'P', 'log(a/rs)':'a/rs', 'log(K)':'K', 'log(rho_gp)':'rho_gp', 'log(sigma_gp)':'sigma_gp'}

        lbound, ubound = -np.inf, np.inf

        if var in self.prior_dict:
                
                l, u = self.prior_dict[var].bounds()
                lbound = max(lbound, l)
                ubound = min(ubound, u)

        if var == 'log10(age)' and 'age' in self.prior_dict:

            l, u = self.prior_dict['age'].bounds()
            lbound = max(lbound, np.log10(l))
            ubound = min(ubound, np.log10(u))

        elif var[:4] == 'log(':

            svar = ' '.join([logvars[var.split()[0]]]+var.split()[1:])

            if svar in self.prior_dict:

                l, u = self.prior_dict[svar].bounds()
                lbound = max(lbound, np.log(l))
                ubound = min(ubound, np.log(u))

            if var.split()[0] == 'log(a/rs)':

                svar = ' '.join(['rhos']+var.split()[1:])

                if svar in self.prior_dict:

                    lr, ur = self.get_bounds(svar)
                    lr = max(lr, 0)

                    svar = ' '.join(['log(P)']+var.split()[1:])

                    lp, up = self.get_bounds(svar)
                    lp = np.exp(lp)
                    up = np.exp(up)

                    lbound = max(lbound, 52.86425121092176*lr**(1/3)*lp**(2/3))
                    ubound = min(ubound, 52.86425121092176*ur**(1/3)*up**(2/3))

        elif var[:6] == 'cos(i)':

            svar = ' '.join(['cos(i)']+var.split()[1:])

            if svar in self.prior_dict:

                l, u = self.prior_dict[svar].bounds()
                l = max(l, 0)
                u = min(u, np.pi/2)
                lbound = max(lbound, np.cos(u))
                ubound = min(ubound, np.cos(l))

        elif var[:4] == 'secw' or var[:4] == 'sesw':

            el, eu = 0, 0.9
            wl, wu = -np.pi, np.pi
            
            svar = ' '.join(['e']+var.split()[1:])
            if svar in self.prior_dict:
                el2, eu2 = self.prior_dict[svar].bounds()
                el = max(el, el2)
                eu = min(eu, eu2)
            
            svar = ' '.join(['w']+var.split()[1:])
            if svar in self.prior_dict:
                wl2, wu2 = self.prior_dict[svar].bounds()
                wl = max(wl, wl2)
                wu = min(wu, wu2)

            if var[:4] == 'secw':

                blist = [np.sqrt(el)*np.cos(wl), np.sqrt(el)*np.cos(wu), np.sqrt(eu)*np.cos(wl), np.sqrt(eu)*np.cos(wu)]

                if wl < 0 < wu:
                    blist.append(np.sqrt(eu))
                    blist.append(np.sqrt(el))

                lbound = max(lbound, min(blist))
                ubound = min(ubound, max(blist))

            else:

                blist = [np.sqrt(el)*np.sin(wl), np.sqrt(el)*np.sin(wu), np.sqrt(eu)*np.sin(wl), np.sqrt(eu)*np.sin(eu)]

                if wl < -np.pi/2 < wu:
                    blist.append(-np.sqrt(eu))
                    blist.append(-np.sqrt(el))

                if wl < np.pi/2 < wu:
                    blist.append(np.sqrt(eu))
                    blist.append(np.sqrt(el))

                lbound = max(lbound, min(blist))
                ubound = min(ubound, max(blist))

        return lbound, ubound
    


def setup_star_priors(tab: Table, starmod: SingleStarModel) -> SingleStarModel:
    """Sets up any priors on the stellar parameters that are directly fit for using the internal priors functionality from isochrones. Will replace
    the default priors on those parameters. Only one prior can be applied to each parameter, with the exception of Gaussian plus uniform, which is
    interpreted as a truncated Gaussian.

    Args:
        tab (Table): The table from an Init_priors object.
        starmod (SingleStarModel): The isochrones starmodel object being used in the ExoSystem fit.

    Returns:
        SingleStarModel: Returns the updated starmodel object that was input.
    """

    set_priors = {'mstar': lambda x: starmod.set_prior(mass = x),
                'rstar': lambda x: starmod.set_prior(radius = x),
                'rhostar': lambda x: starmod.set_prior(density = x),
                'eep': lambda x: starmod.set_prior(eep = x),
                'log10(age)': lambda x: starmod.set_prior(age = x),
                'feh': lambda x: starmod.set_prior(feh = x),
                'distance': lambda x: starmod.set_prior(distance = x),
                'AV': lambda x: starmod.set_prior(AV = x)}
    
    var_convert = {'mstar': 'mass',
                   'rstar': 'radius',
                   'rhostar': 'density',
                   'eep': 'eep',
                   'log10(age)': 'age',
                   'feh': 'feh',
                   'distance': 'distance',
                   'AV': 'AV'}

    for var in set_priors:

        i = np.where(tab['Variable'] == var)[0]

        if len(i) == 0:

            continue

        elif np.any(tab['Prior Type'][i] == 'F'):

            if var_convert[var] in starmod._priors:

                starmod._priors.pop(var_convert[var])

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

        set_priors[var](starprior)

    if 'age' in tab['Variable']:

        i = np.where(tab['Variable'] == 'age')[0]

        if tab['Prior Type'][i] == 'F':

            if 'age' in starmod._priors:

                starmod._priors.pop('age')

    return starmod