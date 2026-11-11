import numpy as np
from astropy.table import Table
from isochrones import SingleStarModel
from isochrones.priors import FlatPrior, GaussianPrior
from scipy.optimize import minimize_scalar


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

        self.lbound = max(self.lbound, mean-10*std)
        self.ubound = min(self.ubound, mean+10*std)

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

            if np.isscalar(x):
                return -np.log(upper-lower) if lower <= x <= upper else -np.inf

            else:
                return np.where((x >= lower) & (x <= upper), -np.log(upper-lower), -np.inf)
        
        self.prior_funcs.append(uniform_prior)


    def add_log_uniform_prior(self, lower: float, upper: float):
        """Adds a log uniform prior function to the list of priors with the given bounds. Variable must be positive definite. The prior function
        returns negative infinity if the given trial value is out of the bounds, otherwise the normalized log uniform prior. Does not work if the
        lower bound is 0!!! In that case, use a modified Jeffrey's prior. Updates the lower and upper bounds of the Priors object.
        """

        self.lbound = max(self.lbound, lower)
        self.ubound = min(self.ubound, upper)

        def log_uniform_prior(x):

            if np.isscalar(x):
                return -np.log(x) - np.log(np.log(upper/lower)) if lower <= x <= upper else -np.inf

            else:
                return np.where((x >= lower) & (x <= upper), -np.log(x) - np.log(np.log(upper/lower)), -np.inf)

        self.prior_funcs.append(log_uniform_prior)

    
    def add_modified_jeffreys_prior(self, upper: float, knee: float):
        """Adds a modified Jeffrey's prior function to the list of priors with the given bounds. Variable must be positive definite. Implicit lower
        bound of 0. The prior function returns negative infinity if the given trial value is out of the bounds, otherwise the normalized modified
        Jeffrey's prior. The modified Jeffrey's prior becomes log uniform below the knee value. Updates the lower and upper bounds of the Priors object.
        """

        self.lbound = max(self.lbound, 0)
        self.ubound = min(self.ubound, upper)

        def modified_jeffreys_prior(x):

            if np.isscalar(x):
                return -np.log(x+knee) - np.log(np.log((knee+upper)/knee)) if 0 <= x <= upper else -np.inf

            else:
                return np.where((x >= 0) & (x <= upper), -np.log(x+knee) - np.log(np.log((knee+upper)/knee)), -np.inf)

        self.prior_funcs.append(modified_jeffreys_prior)


    def add_unit_circle_prior(self):
        """Adds a unit circle prior to flatten out the prior space when sampling an angle with cos(angle) and sin(angle). This is automatically
        applied when sampling omega or w this way. The assumed input on this prior is cos^2(angle) + sin^2(angle). This is generally hidden from
        users and is not included in the init priors GUI.
        """

        def unit_circle_prior(x):

            if np.isscalar(x):
                return -0.5 * np.log(x) if x <= 1 else -np.inf

            else:
                return np.where(x <= 1, -0.5 * np.log(x), -np.inf)

        self.prior_funcs.append(unit_circle_prior)


    def apply_priors(self, x: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Applies all priors in the list of functions to the given trial value, and returns the log likelihood value to be added during fitting.

        Args:
            x (ArrayLike): The value to apply the priors to.

        Returns:
            ArrayLike: The log likelihood value from the priors.
        """

        loglike = 0

        if not np.isscalar(x):
            x = np.array(x)
            loglike = np.zeros(x.shape)

        for prior in self.prior_funcs:

            loglike += prior(x)

        return loglike
    

    def bounds(self) -> tuple[float, float]:
        """Retrieves the boundaries on this parameter set by any priors.
        
        Returns:
            tuple: The lower and upper boundaries of this variable given the set of priors stored.
        """

        return self.lbound, self.ubound


    def sample(self, n: int):

        xmin, xmax = self.bounds()

        res = minimize_scalar(lambda x: -self.apply_priors(x), bounds = (xmin, xmax) if np.all(np.isfinite([xmin, xmax])) else None)
        max_log_prob = -res.fun

        accepted = []
        col = 0

        while col < n:

            cands = np.random.uniform(low = xmin, high = xmax, size = 2*n)

            lp = self.apply_priors(cands)

            u = np.random.random(cands.shape)
            accept = np.log(u) < lp - max_log_prob
            valid = accept & np.isfinite(lp)

            valid_cands = cands[valid]

            if len(valid_cands) > 0:
                accepted.append(valid_cands)
                col += len(valid_cands)

        return np.concatenate(accepted)[:n]


class AllPriors:
    """A class for parsing the Init_priors tables into actual priors, and applying them to all parameters. Stores a dict of Priors objects for each
    parameter and a dict of all fixed parameters.
    """

    def __init__(self):

        self.prior_dict = {}
        self.fixed = {}

        self.searchvars = {'P': ['log(P)'], 'a/rs': ['log(a/rs)'], 'rhostar': ['log(rhostar)', 'eep'], 'i': ['cos(i)','sico'],
                           'cos(i)': ['cos(i)','sico'], 'K': ['log(K)'], 'e': ['secw'], 'w': ['secw'], 'rho_gp': ['log(rho_gp)'],
                           'sigma_gp': ['log(sigma_gp)'], 'age': ['log10(age)'], 'q': ['log(q)'], 'omega': ['sico','comega'],
                           'rsq_omega': ['comega','sico'], 'comega': ['comega','sico'], 'somega': ['somega','sico'], 'distance': ['plx'],
                           'rstar': ['rstar', 'eep'], 'mstar': ['mstar', 'eep'], 'log(rhostar)': ['log(rhostar)', 'eep']}


    def init_fixed(self, tab: Table, x0: dict):
        """Sets up a dict of fixed parameters and their values.

        Args:
            tab (Table): The table from an Init_priors object with a list of priors to set up.
            x0 (dict): A dict with initial values for all fit parameters from an ExoSystem fit, used to determine which variables are being fit for and
                can be set to fixed.
        """

        x0keys = list(x0.keys())

        for i in range(len(tab)):

            if tab['Prior Type'][i] != 'F':
                continue

            var = tab['Variable'][i]

            split = var.split()

            svar = split[0]
            if svar in self.searchvars:
                svar = self.searchvars[svar]
            else:
                svar = [svar]

            fvar = [' '.join([sv]+split[1:]) for sv in svar]

            if np.any(np.isin(fvar, x0keys)):

                self.fixed[var] = tab['Param 1'][i]

            elif split[-1] == 'x':

                for y in x0:

                    if np.any(np.isin(y.split()[0], svar)):

                        yvar = ' '.join([split[0]]+y.split()[1:])

                        self.fixed[yvar] = tab['Param 1'][i]

            elif var in ['log(rhostar)', 'rhostar']:

                for y in x0:

                    if y.split[0] == 'log(a/rs)':

                        self.fixed[var] = tab['Param 1'][i]

                        break


        keys = list(self.fixed.keys())
        for f in keys:

            split = f.split()

            if split[0] in ['P', 'a/rs', 'K', 'rho_gp', 'sigma_gp', 'q', 'rhostar']:

                sf = ' '.join(['log({0})'.format(split[0])]+split[1:])

                self.fixed[sf] = np.log(self.fixed.pop(f))

            elif split[0] == 'age':

                self.fixed['log10(age)'] = np.log10(self.fixed.pop(f))

            elif split[0] == 'i':

                sf = ' '.join(['cos(i)']+split[1:])

                self.fixed[sf] = np.cos(self.fixed.pop(f))

            elif split[0] == 'omega':

                sf = ' '.join(['comega']+split[1:])
                sf2 = ' '.join(['somega']+split[1:])

                self.fixed[sf] = np.cos(self.fixed[f])
                self.fixed[sf2] = np.sin(self.fixed.pop(f))

            elif split[0] == 'distance':

                self.fixed['plx'] = 1000/self.fixed.pop(f)

            if split[0] in ['i','cos(i)','omega','comega','somega']:

                sf = ' '.join(['sico']+split[1:])
                sf2 = ' '.join(['siso']+split[1:])

                if sf in self.fixed:
                    self.fixed.pop(sf)
                if sf2 in self.fixed:
                    self.fixed.pop(sf2)

            if split[0] in ['e','w']:

                sf = ' '.join(['secw']+split[1:])
                sf2 = ' '.join(['sesw']+split[1:])

                if sf in self.fixed:
                    self.fixed.pop(sf)
                if sf2 in self.fixed:
                    self.fixed.pop(sf2)


    def init_priors(self, tab: Table, x0: dict, fit_ttv: np.typing.NDArray):
        """Parses the Init_priors table into actual priors for any relevant parameters that are being fit.

        Args:
            tab (Table): The table from an Init_priors object with a list of priors to set up.
            x0 (dict): A dict with initial values for all fit parameters from an ExoSystem fit, used to determine which variables are being fit for and
                can be set to fixed.
            fit_ttv (ndarray): An array of bools for each planet. They represent whether or not each planet is being fit for TTVs. Priors can be placed
                on P and Tc for a planet even if they are fit for TTVs and don't directly have those in the fit parameters.
        """

        x0keys = list(x0.keys())

        for i in range(len(tab)):

            if tab['Prior Type'][i] == 'F':
                continue

            var = tab['Variable'][i]

            split = var.split()

            svar = split[0]
            if svar in self.searchvars:
                svar = self.searchvars[svar]
            else:
                svar = [svar]

            fvar = [' '.join([sv]+split[1:]) for sv in svar]

            if np.any(np.isin(svar, ['log(P)','Tc'])) and np.any(fit_ttv):

                if split[-1] == 'x':

                    for j in range(len(fit_ttv)):

                        if fit_ttv[j] and 'ror {0}'.format(j+1) in x0:

                            self.set_up_prior('{0} {1}'.format(split[0], j+1), tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

                elif fit_ttv[int(split[-1])-1] and 'ror {0}'.format(split[-1]) in x0:

                    self.set_up_prior(var, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])


            if np.any(np.isin(fvar, x0keys)):

                self.set_up_prior(var, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

            elif split[-1] == 'x':

                for y in x0:

                    if np.any(np.isin(y.split()[0], svar)):

                        yvar = ' '.join([split[0]]+y.split()[1:])

                        self.set_up_prior(yvar, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

            elif var in ['log(rhostar)', 'rhostar']:

                for y in x0:

                    if y.split()[0] == 'log(a/rs)':

                        self.set_up_prior(var, tab['Prior Type'][i], [tab['Param {0}'.format(k)][i] for k in range(1,3)])

                        break



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

        elif prior_type == 'LU':

            self.prior_dict[var].add_log_uniform_prior(params[0], params[1])

        elif prior_type == 'MJ':

            self.prior_dict[var].add_modified_jeffreys_prior(params[0], params[1])

        elif prior_type == 'UC':

            self.prior_dict[var].add_unit_circle_prior()


    def apply(self, par: dict) -> float:
        """Applies all priors to all fit parameters and returns the total log likelihood.

        Args:
            par (dict): A dict of the fit parameters from the ExoSystem fit.

        Returns:
            float: The total log likelihood from the priors.
        """

        log_like = np.zeros(np.array(list(par.values())[0]).shape)

        for var in self.prior_dict:

            split = var.split()

            if split[0] in ['P', 'a/rs', 'K', 'rho_gp', 'sigma_gp', 'q']:

                svar = ' '.join(['log({0})'.format(split[0])]+split[1:])

                if svar in par:

                    x = np.exp(par[svar])

                    log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'age':

                if 'log10(age)' in par:

                    x = 10**(par['log10(age)'])

                    log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] in ['log(rhostar)','rhostar']:

                if split[0] == 'rhostar' and 'log(rhostar)' in par:

                    x = np.exp(par['log(rhostar)'])

                    log_like += self.prior_dict[var].apply_priors(x)

                elif 'log(rhostar)' in par:

                    x = par['log(rhostar)']

                    log_like += self.prior_dict[var].apply_priors(x)

                else:

                    for y in par:

                        if y.split()[0] == 'log(a/rs)':

                            y2 = ' '.join(['log(P)']+y.split()[1:])

                            if y2 in par:

                                x = 0.018916375 * np.exp(par[y])**3 / np.exp(par[y2])**2

                                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'i':

                svar = ' '.join(['cos(i)']+split[1:])
                svar1 = ' '.join(['sico']+split[1:])
                svar2 = ' '.join(['siso']+split[1:])

                if svar in par:

                    x = np.arccos(par[svar])

                elif svar1 in par:

                    x = np.arccos(1-2*(par[svar1]**2 + par[svar2]**2))

                else:
                    pass

                log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'e':

                svar1 = ' '.join(['secw']+split[1:])
                svar2 = ' '.join(['sesw']+split[1:])

                if svar1 in par:

                    x = par[svar1]**2 + par[svar2]**2

                    log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'w':

                svar1 = ' '.join(['secw']+split[1:])
                svar2 = ' '.join(['sesw']+split[1:])

                if svar1 in par:

                    x = np.arctan2(par[svar2], par[svar1])

                    p0 = self.prior_dict[var].apply_priors(x)
                    p1 = self.prior_dict[var].apply_priors(x+2*np.pi)
                    p2 = self.prior_dict[var].apply_priors(x-2*np.pi)

                    log_like += np.max([p0,p1,p2])

            elif split[0] == 'omega':

                svar1 = ' '.join(['sico']+split[1:])
                svar2 = ' '.join(['siso']+split[1:])

                if svar1 not in par:

                    svar1 = ' '.join(['comega']+split[1:])
                    svar2 = ' '.join(['somega']+split[1:])

                if svar1 in par:

                    x = np.arctan2(par[svar2], par[svar1])

                    p0 = self.prior_dict[var].apply_priors(x)
                    p1 = self.prior_dict[var].apply_priors(x+2*np.pi)
                    p2 = self.prior_dict[var].apply_priors(x-2*np.pi)

                    log_like += np.max([p0,p1,p2])

            elif split[0] == 'rsq_omega':

                svar1 = ' '.join(['comega']+split[1:])
                svar2 = ' '.join(['somega']+split[1:])

                if svar1 in par:

                    x = par[svar1]**2 + par[svar2]**2

                    log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] in ['comega', 'somega'] and var not in par:
                
                svar1 = ' '.join(['sico']+split[1:])
                svar2 = ' '.join(['siso']+split[1:])

                if svar1 in par:

                    r = np.sqrt(par[svar1]**2 + par[svar2]**2)

                    if split[0] == 'comega':
                        
                        x = par[svar1]/r

                    else:

                        x = par[svar2]/r

                    log_like += self.prior_dict[var].apply_priors(x)

            elif split[0] == 'distance':

                if 'plx' in par:

                    x = 1000/par['plx']

                    log_like += self.prior_dict[var].apply_priors(x)

            elif var in par:

                x = par[var]

                log_like += self.prior_dict[var].apply_priors(x)

        return log_like