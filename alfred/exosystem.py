import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as pe
import mplcursors
import emcee
import corner
import batman
from scipy.stats import linregress, truncnorm
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy import constants
from astropy.timeseries import LombScargle
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
from isochrones import SingleStarModel, get_ichrone
import pickle
from tqdm.auto import tqdm
import os
import copy
import shutil

from alfred._rv_func import _rvModel
from alfred.init_class import *
from alfred.ld_grids import *
from alfred.priors import *
from alfred import is_notebook

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
np.set_printoptions(legacy='1.25')


class ExoSystem:
    """A class for fitting exoplanet systems.
    """

    def __init__(self, direc: str, init_planets = 'init_planets.txt', init_star = 'init_star.txt', init_lcs = 'init_lcs.txt',
                 init_rv = 'init_rv.txt', init_ld = 'init_ld.txt', init_priors = 'init_priors.txt', init_ttvs = 'init_ttvs.txt'):
        """Initialize an ExoSystem object with a directory and init files. The init files must all be created using alfred.

        Args:
            direc (str): The directory in which the init files, data, outputs, and plots of the ExoSystem are stored.
            init_planets (str, optional): Name of the planet parameter init file. Default is init_planets.txt.
            init_star (str, optional): Name of the star parameter init file. Default is init_star.txt.
            init_lcs (str, optional): Name of the light curve data init file. Default is init_lcs.txt.
            init_rv (str, optional): Name of the RV data init file. Default is init_rv.txt.
            init_ld (str, optional): Name of the limb darkening parameter init file. Default is init_ld.txt.
            init_priors (str, optional): Name of the priors init file. Default is init_priors.txt.
            init_ttvs (str, optional): Name of the TTV init file. Default is init_ttvs.txt.
        """


        self.direc = direc
        if self.direc[-1] != '/':
            self.direc += '/'
        self.init_planets = Init_planets(self.direc, init_planets).from_file()
        self.init_star = Init_star(self.direc, init_star).from_file()
        self.init_lcs = Init_lcs(self.direc, init_lcs).from_file()
        self.init_rv = Init_rv(self.direc, init_rv).from_file()
        self.init_ld = Init_ld(self.direc, init_ld).from_file()

        if os.path.exists(self.direc+'/'+init_priors):
            self.init_priors = Init_priors(self.direc, init_priors).from_file()
        else:
            self.init_priors = None

        self.init_ttvs_name = init_ttvs

        tab_planets = self.init_planets.table
        tab_star = self.init_star.table
        tab_lcs = self.init_lcs.table
        tab_rv = self.init_rv.table
        tab_ld = self.init_ld.table


        #which to fit

        self.is_transit = np.array(tab_planets['Transiting'])
        self.nt = int(np.sum(self.is_transit))
        self.n = len(self.is_transit)

        self.is_eclipse = np.array(tab_planets['Fit Eclipse']) & self.is_transit

        self.is_rv = np.array(tab_planets['RV Signal'])
        self.nr = int(np.sum(self.is_rv))

        self.fit_ttv = np.array(tab_planets['Fit TTVs'])
        self.nttv = int(np.sum(self.fit_ttv))

        self.fit_ecc = np.array(tab_planets['Fit Ecc'])
        self.ne = int(np.sum(self.fit_ecc))


        #planet params

        self.p = np.array(tab_planets['Period'])

        self.tc = np.array(tab_planets['Tc'])

        self.ror = np.array(tab_planets['Rp/Rs'])[self.is_transit]

        self.ar = np.array(tab_planets['a/Rs'])[self.is_transit]

        self.cosi = np.array(tab_planets['cos(i)'])[self.is_transit]

        self.k = np.array(tab_planets['K'])[self.is_rv]

        self.secosw = np.array(tab_planets['sqrt(e)cos(w)'])

        self.sesinw = np.array(tab_planets['sqrt(e)sin(w)'])

        self.fp = np.array(tab_planets['fp'])


        self.transitsortorder = np.argsort(self.p[self.is_transit])


        #stellar params

        self.rs = tab_star['Radius'][0]
        self.rserr = tab_star['Radius'][1]
        
        self.ms = tab_star['Mass'][0]
        self.mserr = tab_star['Mass'][1]

        self.Ts = tab_star['Teff'][0]
        self.Tserr = tab_star['Teff'][1]

        self.logg = tab_star['log(g)'][0]
        self.loggerr = tab_star['log(g)'][1]

        self.feh = tab_star['Fe/H'][0]
        self.feherr = tab_star['Fe/H'][1]

        #parallax

        self.plax = tab_star['Parallax'][0]
        self.plaxerr = tab_star['Parallax'][1]

        #photometry

        self.Jmag = tab_star['J'][0]
        self.Jmagerr = tab_star['J'][1]

        self.Hmag = tab_star['H'][0]
        self.Hmagerr = tab_star['H'][1]

        self.Kmag = tab_star['K'][0]
        self.Kmagerr = tab_star['K'][1]

        self.Gmag = tab_star['G'][0]
        self.Gmagerr = tab_star['G'][1]
        
        self.Bpmag = tab_star['Bp'][0]
        self.Bpmagerr = tab_star['Bp'][1]

        self.Rpmag = tab_star['Rp'][0]
        self.Rpmagerr = tab_star['Rp'][1]

        self.W1mag = tab_star['W1'][0]
        self.W1magerr = tab_star['W1'][1]

        self.W2mag = tab_star['W2'][0]
        self.W2magerr = tab_star['W2'][1]

        self.W3mag = tab_star['W3'][0]
        self.W3magerr = tab_star['W3'][1]

        self.magwls = [1.235,1.662,2.159,0.582239,0.503575,0.761996,3.3526,4.6028,11.5608]
        self.magobs = np.array([self.Jmag,self.Hmag,self.Kmag,self.Gmag,self.Bpmag,self.Rpmag,self.W1mag,self.W2mag,self.W3mag])
        self.magobserr = np.array([self.Jmagerr,self.Hmagerr,self.Kmagerr,self.Gmagerr,self.Bpmagerr,self.Rpmagerr,self.W1magerr,self.W2magerr,self.W3magerr])


        #limb darkening
            
        if np.any(np.unique(tab_ld['Filter'], return_counts = True)[1] > 1):
            raise AttributeError('You may only specify one set of limb darkening parameters for each filter.')
            
        self.ld = {tab_ld['Filter'][i]: [tab_ld['u1'][i], tab_ld['u2'][i]] for i in range(len(tab_ld))}


        #lightcurve files
        
        self.fnames = list(tab_lcs['File'])

        self.lcnames = np.array(tab_lcs['Nickname'])
                        
        self.exptimes = np.array(tab_lcs['Exp Time'])/60/60/24
            
        self.filters = np.array(tab_lcs['Filter'])

        self.detrend = np.array(tab_lcs['Detrend'])

        self.lc_err_scale = np.array(tab_lcs['Err Scale'])

        for filt in self.filters:
            if filt not in self.ld.keys():
                raise AttributeError('{0} in light curve initialization does not have limb darkening specified.'.format(filt))


        self.tt = []
        self.f = []
        self.ferr = []

        for i, name in enumerate(self.fnames):

            if not os.path.exists(self.direc+'/'+name):
                print('LC file {0} not found.'.format(name))
                continue

            if name[-5:] == '.fits':

                with fits.open(self.direc+'/'+name) as hdul:

                    tdata = hdul[1].data

                tt0 = tdata[tab_lcs['Time Col'][i]]
                f0 = tdata[tab_lcs['Flux Col'][i]]
                ferr0 = tdata[tab_lcs['Err Col'][i]]
                
                try:
                    q = tdata[tab_lcs['Quality Col'][i]]
                except:
                    q = np.zeros_like(ferr0)
                
                j = np.where((np.isnan(f0)) | (q != 0))[0]
                tt0 = np.delete(tt0, j)
                f0 = np.delete(f0, j)
                ferr0 = np.delete(ferr0, j)

                scale = np.median(f0)
                f0 = f0/scale
                ferr0 = ferr0/scale


            elif name[-4:] == '.npz':

                dat = np.load(self.direc+'/'+name)

                tt0 = dat[tab_lcs['Time Col'][i]]
                f0 = dat[tab_lcs['Flux Col'][i]]
                ferr0 = dat[tab_lcs['Err Col'][i]]

                try:
                    q = dat[tab_lcs['Quality Col'][i]]
                except:
                    q = np.zeros_like(ferr0)
                
                j = np.where((np.isnan(f0)) | (q != 0))[0]
                tt0 = np.delete(tt0, j)
                f0 = np.delete(f0, j)
                ferr0 = np.delete(ferr0, j)

                scale = np.median(f0)
                f0 = f0/scale
                ferr0 = ferr0/scale

            elif name[-4:] == '.csv':

                dat = Table.read(self.direc+'/'+name)

                tt0 = dat[tab_lcs['Time Col'][i]]
                f0 = dat[tab_lcs['Flux Col'][i]]
                ferr0 = dat[tab_lcs['Err Col'][i]]

                try:
                    q = dat[tab_lcs['Quality Col'][i]]
                except:
                    q = np.zeros_like(ferr0)
                
                j = np.where((np.isnan(f0)) | (q != 0))[0]
                tt0 = np.delete(tt0, j)
                f0 = np.delete(f0, j)
                ferr0 = np.delete(ferr0, j)

                scale = np.median(f0)
                f0 = f0/scale
                ferr0 = ferr0/scale


            elif name[-4:] in ['.dat', '.txt']:

                dat = Table.read(self.direc+'/'+name, format = 'ascii')

                tt0 = dat[tab_lcs['Time Col'][i]]
                f0 = dat[tab_lcs['Flux Col'][i]]
                ferr0 = dat[tab_lcs['Err Col'][i]]

                try:
                    q = dat[tab_lcs['Quality Col'][i]]
                except:
                    q = np.zeros_like(ferr0)
                
                j = np.where((np.isnan(f0)) | (q != 0))[0]
                tt0 = np.delete(tt0, j)
                f0 = np.delete(f0, j)
                ferr0 = np.delete(ferr0, j)

                scale = np.median(f0)
                f0 = f0/scale
                ferr0 = ferr0/scale


            self.tt.append(np.array(tt0)+tab_lcs['Time Offset'][i]-2450000)
            self.f.append(np.array(f0))
            self.ferr.append(np.array(ferr0 * self.lc_err_scale[i]))

        self.tt_orig = copy.deepcopy(self.tt)
        self.f_orig = copy.deepcopy(self.f)
        self.ferr_orig = copy.deepcopy(self.ferr)

        #rv files

        self.rvfiles = np.array(tab_rv['File'])

        self.rvnames = np.array(tab_rv['Nickname'])

        self.rv_err_scale = np.array(tab_rv['Err Scale'])

        self.tr = []
        self.rv = []
        self.rverr = []
        self.which_rv = []

        for i, name in enumerate(self.rvfiles):

            if os.path.exists(self.direc + name):
                
                try:
                    rvdata = Table.read(self.direc + name)
                except:
                    rvdata = Table.read(self.direc + name, format = 'ascii')
                
                rvdata.sort(tab_rv['Time Col'][i])

                msscale = 1e3 if tab_rv['m/s or km/s'][i] == 'km/s' else 1
                
                tr0 = np.array(rvdata[tab_rv['Time Col'][i]]) + tab_rv['Time Offset'][i] - 2450000
                rv0 =  np.array(rvdata[tab_rv['RV Col'][i]]) * msscale
                rv0 = rv0 - (np.max(rv0)+np.min(rv0))/2
                rverr0 = np.array(rvdata[tab_rv['Err Col'][i]]) * self.rv_err_scale[i] * msscale

                self.tr.append(tr0)
                self.rv.append(rv0)
                self.rverr.append(rverr0)
                self.which_rv.append([i]*len(tr0))

        if len(self.tr) > 0:

            self.tr = np.concatenate(self.tr)
            self.rv = np.concatenate(self.rv)
            self.rverr = np.concatenate(self.rverr)
            self.which_rv = np.concatenate(self.which_rv)

            o = np.argsort(self.tr)
            self.tr = self.tr[o]
            self.rv = self.rv[o]
            self.rverr = self.rverr[o]
            self.which_rv = self.which_rv[o]

            self.tr_ref = np.min(self.tr)
            self.tr_plot = np.linspace(np.min(self.tr)-5, np.max(self.tr+5), 1000)
            self.tr_phase = np.linspace(-0.5, 0.5, 1000)


    def fit(self, name: str, nburn: int, nrun: int, fit_transit: bool, fit_rv: bool, fit_star: bool, nwalk: int = 0, fit_ld = False,
            use_priors = False, rv_bkg_order: int = 0, star_run: str = None, save_samples = False, sigma_clip: float = 5,
            lc_supersample_size: int = 600, show_plots = True, order_a = False, skip_state_check = False) -> None:
        """Fit the light curve, RV, and/or stellar data for this ExoSystem using MCMC.
        
        Parameter optimization is done with scipy minimize before running MCMC. Additionally, when fitting transits to the light curves,
        will perform up to 10 rounds of sigma clipping to get rid of outliers in the lightcurve, stopping when the number of clipped
        points in a round is less than 10. Clipped points are masked out, and masks are saved in the Masks folder. During each round, masked
        points are shown in a plot before continuing. MCMC is performed with a joint likelihood function on all of the specified data.
        
        When fitting stellar parameters simultaneously with planet parameters, runs a star-only fit first to get physical starting values
        for the stellar parameters.
        
        Flattened and thinned MCMC chains are stored to pickle files ending in _res.p in the Output folder, as well as chains of derived parameters
        in a pickle file ending in _dres.p. A human readable table of median derived parameters, as well as their upper and lower
        uncertainties, is saved to the Results folder and printed at the end. Plots of the MCMC chains and corner plots are generated and
        saved to a folder in the Plots folder, along with plots of the best fit models to the data. Some run settings (e.g. fit_transit) are saved in
        a pickle file ending in _run_settings.p.

        Args:
            name (str): Name of the fit. This name will be attached to all ouputs, including pickle files of data, human readable results tables,
                and the folder in Plots in which this run's plots will be saved. This name is also used for loading results back in to be manipulated
                or plotted again.

            nburn (int): Number of burn-in steps for the MCMC. These steps are thrown out before saving the results and making plots. The burn-in
                allows the chains to settle into the maximum likelihood.

            nrun (int): Number of sampling steps for the MCMC. These steps are saved and used for results and making plots. They do not include the
                burn-in steps.
            
            fit_transit (bool): Whether or not to fit transits to the light curve data.
            
            fit_rv (bool): Whether or not to fit a model to the RV data.
            
            fit_star (bool): Whether or not to fit stellar parameters. Can be fit on their own, or if transit data is also being fit
                (with or without RV data as well). Cannot be fit with just RV data.

            nwalk (int, optional): Number of walkers to use for the MCMC. This needs to be at least 2 times the number of free parameters. If nwalk is
                less than that value, or if nwalk isn't provided, nwalk will be set to exactly 2 times the number of free parameters.
            
            fit_ld (bool, optional): Whether or not to fit quadratic limb darkening coefficients as free parameters. Will be overriden to False if
                fit_star is True, as limb darkening coefficients are generated at each step in that case. Default is False.
            
            use_priors (bool, optional): Whether or not to use user specified priors from the init_priors file specified during ExoSystem
                initialization. Default is False.
            
            rv_bkg_order (int, optional): Polynomial order for the background trend fit to the RV data. Can be 0 (for a flat line), 1 (for a slope),
                or 2 (for a quadratic). Default is 0.
            
            star_run (str, optional): Name of a previous fit run that included stellar parameters. If specified, provides starting values for the
                stellar parameters in this fit. If not specified, or it doesn't exist, runs a star only fit before continuing to get physical starting
                parameters. Default is None.
            
            save_samples (bool, optional): Whether or not to save the full, unflattened, un-thinned MCMC chains to a pickle file. This can be handy
                if you expect to want to remove problematic walkers that wandered off from the final results. Most of the time this isn't necessary,
                and these take up additional storage. Default is False.
            
            sigma_clip (float, optional): Sets how strict the sigma clipping is. During each round, in each light curve, points farther than
                sigma_clip times the root median squared of the residual are masked out. If you see in the plots that points are being masked out
                that probably shouldn't be, increase this parameter and run it again. To turn off sigma clipping set this parameter to 0. Default is 5.
            
            lc_supersample_size (int, optional): Sets the exposure time, in seconds, to supersample the transit light curves to. For each data set,
                the number of supersamples will be that data set's exposure time divided by this value, rounded to the nearest integer.
                
                For example, if a data set has an exposure time of 1800 seconds and this parameter is set to 600 seconds, each point in the data
                set will be modeled as the average of three points with an exposure time of 600 seconds.
                
                Decreasing this parameter will increase the number of supersamples, potentially improving model accuracy, but at the cost of
                slowing down the fit.
                
                Default is 600 seconds.
            
            show_plots (bool, optional): Whether or not to show plots at the end of the run. Plots are saved regardless. Default is True.
            
            order_a (bool, optional): Whether or not to put an order prior on the planetary semi-major axes in a transit fit with multiple transiting
                planets. If True, restricts the semi-major axis of each planet to be greater than those of planets with shorter orbital periods. If
                fitting the stellar parameters, overrides this to False, since the stellar mass is instead used to set semi-major axes. Default is
                False.

            skip_state_check (bool, optional): Passed to the emcee sampler. Whether or not to skip checking whether the initial parameters can fully
                explore the space. Only set to True if you keep getting initial state check errors after burn in. Default is False.
        """


        self.nburn = nburn
        self.nrun = nrun
        self.nwalk = nwalk
        self.rv_bkg_order = rv_bkg_order
        self.sigma_clip = sigma_clip
        self.fit_ld = fit_ld
        self.fit_transit = fit_transit
        self.fit_rv = fit_rv
        self.fit_star = fit_star
        self.order_a = order_a
        self.fit_planets = self.fit_transit or self.fit_rv
        self.lc_supersample_size = lc_supersample_size


        self.delete_run(name)


        if self.rv_bkg_order not in [0,1,2]:
            print('Invalid RV background polynomial order. Must be 0, 1, or 2.')
            return None

        if not fit_transit and not fit_rv and not fit_star:
            print('You need to fit something!')
            return None
        
        if not fit_transit and fit_rv and fit_star:
            print('To fit stellar parameters simultaneously with planet parameters, please use transit data (set fit_transit to True).')
            return None
        
        if not os.path.isdir(self.direc+'Plots/'+name):
            os.mkdir(self.direc+'Plots/'+name)
        
        self.use_priors = use_priors
        if self.init_priors is None and use_priors:
            print('No init_priors file found. Check the name in the system initialization, or run Init_priors().create(). Running with no priors.')
            self.use_priors = False
        
        
        if self.fit_transit:
            self.supersamples = np.array([max(1,int(x/self.lc_supersample_size)) for x in self.exptimes*24*60*60])


        if self.fit_star:

            self.fit_ld = False
            self.order_a = False

            self.misti = get_ichrone('mist')

            props = {'parallax': (self.plax, self.plaxerr), 'Teff': (self.Ts, self.Tserr),
            'J': (self.Jmag, self.Jmagerr), 'H': (self.Hmag, self.Hmagerr), 'K': (self.Kmag, self.Kmagerr),
            'W1': (self.W1mag, self.W1magerr), 'W2': (self.W2mag, self.W2magerr), 'W3': (self.W3mag, self.W3magerr),
            'G': (self.Gmag, self.Gmagerr), 'BP': (self.Bpmag, self.Bpmagerr), 'RP': (self.Rpmag, self.Rpmagerr)}

            if not np.isnan(self.logg):
                props['logg'] = (self.logg, self.loggerr)
            
            if not np.isnan(self.feh):
                props['feh'] = (self.feh, self.feherr)

            self.starmod = SingleStarModel(self.misti, name = name, **props)

            if self.use_priors:
                self.starmod = setup_star_priors(self.init_priors.table, self.starmod)

            self.ldgrids = {}

            for filt in np.unique(self.filters):                    

                interpu1, interpu2 = load_ld_grid(filt)

                self.ldgrids[filt] = [interpu1, interpu2]



        if np.any(self.fit_ttv) and self.fit_transit:
            self.initialize_ttvs(name = self.init_ttvs_name)


        self.x0 = {}

        if self.fit_planets:

            for i in range(self.n):

                if self.is_transit[i] and self.fit_transit:

                    j = np.sum(self.is_transit[:i])

                    if self.fit_ttv[i]:

                        for l in range(len(self.ttvs0[i+1])):
                            
                            self.x0['TT {0} {1}'.format(i+1, l+1)] = self.ttvs0[i+1][l]

                    else:

                        self.x0['log(P) {0}'.format(i+1)] = np.log(self.p[i])
                        self.x0['Tc {0}'.format(i+1)] = self.tc[i]

                    self.x0['ror {0}'.format(i+1)] = self.ror[j]
                    if not self.fit_star:
                        self.x0['log(a/rs) {0}'.format(i+1)] = np.log(self.ar[j])
                    self.x0['cos(i) {0}'.format(i+1)] = self.cosi[i]

                    if self.is_eclipse[i]:

                        self.x0['fp {0}'.format(i+1)] = self.fp[i]

                elif self.is_rv[i] and self.fit_rv:

                    self.x0['log(P) {0}'.format(i+1)] = np.log(self.p[i])
                    self.x0['Tc {0}'.format(i+1)] = self.tc[i]

                if self.is_rv[i] and self.fit_rv:

                    j = np.sum(self.is_rv[:i])

                    self.x0['log(K) {0}'.format(i+1)] = np.log(self.k[j])

                if self.fit_ecc[i]:

                    self.x0['secw {0}'.format(i+1)] = self.secosw[i]
                    self.x0['sesw {0}'.format(i+1)] = self.sesinw[i]

        
        if self.fit_star and (self.fit_transit or self.fit_rv):

            if star_run is not None:

                run_star_fit = False

                try:

                    self.load_results(star_run)

                    self.x0['eep'] = np.median(self.res['eep'])
                    self.x0['log10(age)'] = np.median(self.res['log10(age)'])
                    self.x0['feh'] = np.median(self.res['feh'])
                    self.x0['distance'] = np.median(self.res['distance'])
                    self.x0['AV'] = np.median(self.res['AV'])

                except:

                    print('star_run {0} does not exist. Will run an initial star only optimization.'.format(star_run))
                    run_star_fit = True

            else:

                run_star_fit = True

            if run_star_fit:

                print('Running initial star only optimization:')

                x = self.initial_star_fit()

                self.x0['eep'] = x['eep']
                self.x0['log10(age)'] = x['log10(age)']
                self.x0['feh'] = x['feh']
                self.x0['distance'] = x['distance']
                self.x0['AV'] = x['AV']

        elif self.fit_star:

            self.x0['eep'] = 355
            self.x0['log10(age)'] = 9.66
            self.x0['feh'] = 0
            self.x0['distance'] = 1000/self.plax
            self.x0['AV'] = 0.01


        if self.fit_transit:

            self.gps = []
            
            for i in range(len(self.tt)):

                self.x0['F0 {0}'.format(self.lcnames[i])] = 1.0

                if self.detrend[i]:

                    self.x0['log(rho_gp) {0}'.format(self.lcnames[i])] = 1
                    self.x0['log(sigma_gp) {0}'.format(self.lcnames[i])] = np.log(np.std(self.f[i]))

                    kernel = terms.SHOTerm(rho = np.exp(self.x0['log(rho_gp) {0}'.format(self.lcnames[i])]), sigma = np.exp(self.x0['log(sigma_gp) {0}'.format(self.lcnames[i])]), Q = 1/np.sqrt(2))
                    gp = GaussianProcess(kernel = kernel)
                    self.gps.append(gp)

            if self.fit_ld:

                for filt in np.unique(self.filters):

                    self.x0['u1 {0}'.format(filt)] = self.ld[filt][0]
                    self.x0['u2 {0}'.format(filt)] = self.ld[filt][1]

        if self.fit_rv:
            
            self.x0['gamma'] = 0
            if self.rv_bkg_order > 0:
                self.x0['gamma_dot'] = 0
            if self.rv_bkg_order > 1:
                self.x0['gamma_ddot'] = 0

            for i in np.unique(self.which_rv)[1:]:
                self.x0['rv_offset {0}'.format(self.rvnames[i])] = 0

        if self.use_priors:

            self.allpriors = AllPriors(self.init_priors.table, self.x0, self.fit_ttv)
            
            self.fixed = self.allpriors.fixed.copy()

            for i in range(self.n):

                if 'e {0}'.format(i+1) in self.fixed or 'w {0}'.format(i+1) in self.fixed:

                    self.x0.pop('secw {0}'.format(i+1))
                    self.x0.pop('sesw {0}'.format(i+1))

                    self.x0['e {0}'.format(i+1)] = 0.01
                    self.x0['w {0}'.format(i+1)] = np.pi/2

            for f in self.fixed:

                if f in self.x0:

                    self.x0.pop(f)

        self.keys = list(self.x0.keys())

        print('\nInitial parameters:')
        print(self.x0)
        
        if self.use_priors and len(self.fixed) > 0:
            print('Fixed Parameters:')
            print(self.fixed)


        if self.nwalk < len(self.keys) * 2:
            self.nwalk = len(self.keys) * 2

        if self.fit_transit:

            self.tt = copy.deepcopy(self.tt_orig)
            self.f = copy.deepcopy(self.f_orig)
            self.ferr = copy.deepcopy(self.ferr_orig)

            self.masks = [np.ones(len(self.tt[i]), dtype = bool) for i in range(len(self.tt))]
        

        if self.fit_transit and self.sigma_clip != 0:
            
            self.run_sigma_clip(name, show_plots)

        else:

            res = minimize(lambda x, *args: -1 * log_like({k:v for k,v in zip(self.keys, x)}, *args)[0], [self.x0[k] for k in self.keys], method = 'Nelder-Mead', args = (self,))
            self.x = {k:v for k,v in zip(self.keys, res.x)}


        if self.fit_transit:
            pickle.dump(self.masks, open(self.direc+'Masks/'+name+'_masks.p', 'wb'))
        

        print('\nInitial parameters after optimization:')
        print(self.x)

        
        pos = self.initialize_chains()

        self.run_sampler(pos, skip_state_check)
        
        if save_samples:

            self.save_samples(name)

        self.calc_gelman_rubin()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('')
        self.restab.pprint_all()


    def initial_star_fit(self):
        """Runs an initial minimization on just the stellar parameters using a minimizer to get them in the ballpark before running MCMC. Starts at
        solar parameters. Called by fit.
        
        Do not run this by itself.
        """

        x0 = {'eep': 355, 'log10(age)': 9.66, 'feh': 0, 'distance': 1000/self.plax, 'AV': 0.01}
        keys = list(x0.keys())

        if self.use_priors:

            self.allpriors = AllPriors(self.init_priors.table, x0, self.fit_ttv)

            self.fixed = self.allpriors.fixed.copy()

            for f in self.fixed:

                if f in self.x0:

                    self.x0.pop(f)

        res = minimize(lambda x, *args: -1 * log_like_staronly({k:v for k,v in zip(keys, x)}, *args), [x0[k] for k in keys], method = 'Nelder-Mead', args = (self,))
        x = {k:v for k,v in zip(keys, res.x)}

        print('\nInitial parameters after star-only optimization:')
        print(x)
        print('')

        return x


    def run_sigma_clip(self, name: str, show_plots: bool):
        """Not meant to be run on its own. Performs sigma clipping on the light curve data, with a threshold set by the sigma_clip parameter in fit.
        Initial fits are done using scipy minimize on the log likelihood function. Stops after 10 iterations or when the number of clipped points in an
        iteration is under 10 per data set. Saves plots to Plots/name/sigma_clip/.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save the plots to.
            
            show_plot (bool): Whether or not to show the plots.
        """

        print('\nStarting sigma clipping of lightcurves.')

        clipped = [0]*len(self.tt)

        for i in range(10):

            res = minimize(lambda x, *args: -1 * log_like({k:v for k,v in zip(self.keys, x)}, *args)[0], [self.x0[k] for k in self.keys], method = 'Nelder-Mead', args = (self,))
            x = {k:v for k,v in zip(self.keys, res.x)}

            if not os.path.isdir(self.direc+'Plots/'+name+'/sigma_clip'):
                os.mkdir(self.direc+'Plots/'+name+'/sigma_clip')            

            if self.use_priors:

                y = x | self.fixed

                for j in range(self.n):

                    if 'e {0}'.format(j+1) in y:

                        y['secw {0}'.format(j+1)] = np.sqrt(y['e {0}'.format(j+1)]) * np.cos(y['w {0}'.format(j+1)])
                        y['sesw {0}'.format(j+1)] = np.sqrt(y['e {0}'.format(j+1)]) * np.sin(y['w {0}'.format(j+1)])

            else:

                y = x.copy()

            if self.fit_star:

                rstar, mstar, Tstar, loggstar = self.misti.interp_value([y['eep'],y['log10(age)'],y['feh']],['radius','mass','Teff','logg'])

                arlist = []
                for j in range(self.n):

                    if not self.is_transit[j]:
                        arlist.append(np.nan)
                        continue

                    if self.fit_ttv[j]:

                        p = get_ttv_params(y, j+1, self.ttvi['{0}'.format(j+1)], ar = 1)[0]

                    else:

                        p = np.exp(y['log(P) {0}'.format(j+1)])

                    if self.is_rv[j] and self.fit_rv:

                        e = 0
                        if self.fit_ecc[j]:
                            e = y['secw {0}'.format(j+1)]**2 + y['sesw {0}'.format(j+1)]**2

                        mp = calc_m_from_k(p, np.exp(y['log(K) {0}'.format(j+1)]), e, np.arccos(y['cos(i) {0}'.format(j+1)]), mstar)

                    else:

                        mp = 0

                    ar = (((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                    arlist.append(ar)

            pars = []
            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                if self.fit_ttv[j]:

                    pars.append(get_ttv_params(y, j+1, self.ttvi['{0}'.format(j+1)], ar = arlist[j] if self.fit_star else None))

                else:

                    pars.append(get_transit_params(y, j+1, ar = arlist[j] if self.fit_star else None))


            lastclipped = [0]*len(self.tt)

            for j in range(len(self.tt)):

                mean = y['F0 {0}'.format(self.lcnames[j])]

                fm = mean

                ld = self.ld[self.filters[j]]

                if self.fit_ld:

                    ld = [y['u1 {0}'.format(self.filters[j])], y['u2 {0}'.format(self.filters[j])]]

                if self.fit_star:

                    ld = [self.ldgrids[self.filters[j]][0]([Tstar, loggstar, y['feh']])[0], self.ldgrids[self.filters[j]][1]([Tstar, loggstar, y['feh']])[0]]

                for k in range(self.n):

                    if not self.is_transit[k]:
                        continue

                    l = np.sum(self.is_transit[:k])

                    if self.fit_ttv[k]:

                        p = pars[l][0]

                        for z in range(len(self.ttvi['{0}'.format(k+1)])):

                            if self.ttvsectors['{0} {1}'.format(k+1, z+1)] != j:
                                continue

                            ttime = y['TT {0} {1}'.format(k+1, z+1)]

                            pars[l][1] = ttime

                            ind = np.where((self.tt[j] >= ttime - p/4) & (self.tt[j] <= ttime + p/4))

                            fm0 = np.zeros(len(self.tt[j]))

                            lc = lightCurve(pars[l], self.tt[j][ind], ld, self.exptimes[j], self.supersamples[j])

                            fm0[ind] += lc

                            fm += fm0

                    else:

                        if self.is_eclipse[k]:

                            fm += lightCurve(pars[l], self.tt[j], ld, self.exptimes[j], self.supersamples[j], eclipse = True, fp = y['fp {0}'.format(k+1)], rstar = rstar if self.fit_star else self.rs)

                        else:

                            fm += lightCurve(pars[l], self.tt[j], ld, self.exptimes[j], self.supersamples[j])

                resid = self.f[j] - fm

                gpf = 0

                if self.detrend[j]:
                    
                    jj = np.sum(self.detrend[:j])

                    gp = set_gp_params(np.exp(y['log(rho_gp) {0}'.format(self.lcnames[j])]), np.exp(y['log(sigma_gp) {0}'.format(self.lcnames[j])]), self.tt[j], self.ferr[j], self.gps[jj])

                    gpf = gp.predict(resid)

                    resid = self.f[j] - fm - gpf

                rms = np.sqrt(np.median(resid**2))

                mask = abs(resid) < self.sigma_clip * rms

                c = np.sum(~mask)
                lastclipped[j] = c
                clipped[j] += c

                fig, ax = plt.subplots(3 if self.detrend[j] else 2, sharex = True)

                z = 0

                if self.detrend[j]:

                    z = 1

                    ax[0].scatter(self.tt[j], self.f[j], c = 'black', marker = '.', zorder = 1)
                    ax[0].plot(self.tt[j], gpf + mean, c = 'mediumseagreen', zorder = 2)
                    ax[0].set_rasterized(True)

                ax[0+z].scatter(self.tt[j], self.f[j] - gpf, c = 'black', marker = '.', zorder = 1)
                ax[0+z].plot(self.tt[j], fm, c = 'mediumseagreen', zorder = 2)
                ax[0+z].set_rasterized(True)

                ax[1+z].scatter(self.tt[j][mask], (self.f[j] - gpf - fm)[mask], c = 'black', marker = '.', zorder = 1)
                ax[1+z].scatter(self.tt[j][~mask], (self.f[j] - gpf - fm)[~mask], c = 'red', marker = 'x', zorder = 2)
                ax[1+z].axhline(0, c = 'mediumseagreen', zorder = 3)
                ax[1+z].set_rasterized(True)

                ax[0].set_title('{0} Clipped {1}'.format(self.lcnames[j], c))

                fig.savefig(self.direc+'Plots/'+name+'/sigma_clip/'+'{0}_clip_{1}.png'.format(self.lcnames[j], i+1))

                if show_plots:
                    plt.show()

                else:
                    plt.close()

                k = np.where(self.masks[j])[0]
                self.masks[j][k[~mask]] = False

                self.tt[j] = self.tt[j][mask]
                self.f[j] = self.f[j][mask]
                self.ferr[j] = self.ferr[j][mask]


            if np.all(np.array(lastclipped) < 10):
                break

        print('\nTotal points clipped:')
        print('All light curves: {0}'.format(np.sum(clipped)))
        for i in range(len(self.tt)):
            print('{0}: {1}'.format(self.lcnames[i], clipped[i]))

        self.x = x


    def initialize_chains(self) -> np.typing.NDArray:
        """Not meant to be run on its own. Initializes random positions for the MCMC chains around the initial best fits from optimization. Ensures
        that parameters stay within their bounds when applicable using truncated Gaussian distributions. Otherwise, just uses Gaussian distributions
        to add a little variance to the chains. Also builds a dict, self.parnames, along the way which keeps track of the index of each parameter
        in the pos array.

        Returns:
            ndarray: The initial walker positions.
        """

        self.parnames = {}

        pos = []

        for i, k in enumerate(self.keys):

            self.parnames[k] = i

            if 'log(P)' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(np.log(truncnorm.rvs((np.exp(lb)-np.exp(self.x[k]))/0.0001, (np.exp(ub)-np.exp(self.x[k]))/0.0001, loc = np.exp(self.x[k]), scale = 0.0001, size = self.nwalk)))
                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.0001, self.nwalk)))

            if 'Tc' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.0001, (ub-self.x[k])/0.0001, loc = self.x[k], scale = 0.0001, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.0001, self.nwalk))

            if 'TT' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.0001, (ub-self.x[k])/0.0001, loc = self.x[k], scale = 0.0001, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.0001, self.nwalk))

            if 'ror' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/(0.1*self.x[k]), (ub-self.x[k])/(0.1*self.x[k]), loc = self.x[k], scale = 0.1*self.x[k], size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.1*self.x[k], self.nwalk))

            if 'fp' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/(0.1*self.x[k]), (ub-self.x[k])/(0.1*self.x[k]), loc = self.x[k], scale = 0.1*self.x[k], size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.1*self.x[k], self.nwalk))

            if 'log(a/rs)' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(np.log(truncnorm.rvs((np.exp(lb)-np.exp(self.x[k]))/0.1, (np.exp(ub)-np.exp(self.x[k]))/0.1, loc = np.exp(self.x[k]), scale = 0.1, size = self.nwalk)))
                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.1, self.nwalk)))

            if 'cos(i)' in k:
                lb, ub = 0, 1
                if self.use_priors:
                    lb2, ub2 = self.allpriors.get_bounds(k)
                    lb = max(lb, lb2)
                    ub = min(ub, ub2)
                pos.append(truncnorm.rvs((lb-self.x[k])/0.001, (ub-self.x[k])/0.001, loc = self.x[k], scale = 0.001, size = self.nwalk))

            if 'log(K)' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(np.log(truncnorm.rvs((np.exp(lb)-np.exp(self.x[k]))/0.01, (np.exp(ub)-np.exp(self.x[k]))/0.01, loc = np.exp(self.x[k]), scale = 0.01, size = self.nwalk)))
                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.01, self.nwalk)))

            if 'secw' in k or 'sesw' in k:
                lb, ub = -1, 1
                if self.use_priors:
                    lb2, ub2 = self.allpriors.get_bounds(k)
                    lb = max(lb, lb2)
                    ub = min(ub, ub2)
                pos.append(truncnorm.rvs((lb-self.x[k])/0.01, (ub-self.x[k])/0.01, loc = self.x[k], scale = 0.01, size = self.nwalk))

            if k.split()[0] == 'e':
                lb, ub = 0, 0.9
                if self.use_priors:
                    lb2, ub2 = self.allpriors.get_bounds(k)
                    lb = max(lb, lb2)
                    ub = min(ub, ub2)
                pos.append(truncnorm.rvs((lb-self.x[k])/0.001, (ub-self.x[k])/0.001, loc = self.x[k], scale = 0.001, size = self.nwalk))

            if k.split()[0] == 'w':
                lb, ub = -np.pi, np.pi
                if self.use_priors:
                    lb2, ub2 = self.allpriors.get_bounds(k)
                    lb = max(lb, lb2)
                    ub = min(ub, ub2)
                pos.append(truncnorm.rvs((lb-self.x[k])/1, (ub-self.x[k])/1, loc = self.x[k], scale = 1, size = self.nwalk))

            if 'F0' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.001, (ub-self.x[k])/0.001, loc = self.x[k], scale = 0.001, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.001, self.nwalk))

            if k == 'gamma':
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.1, (ub-self.x[k])/0.1, loc = self.x[k], scale = 0.1, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.1, self.nwalk))

            if k == 'gamma_dot':
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.01, (ub-self.x[k])/0.01, loc = self.x[k], scale = 0.01, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.01, self.nwalk))

            if k == 'gamma_ddot':
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.001, (ub-self.x[k])/0.001, loc = self.x[k], scale = 0.001, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.001, self.nwalk))

            if '_gp' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(np.log(truncnorm.rvs((np.exp(lb)-np.exp(self.x[k]))/(0.01*np.exp(self.x[k])), (np.exp(ub)-np.exp(self.x[k]))/(0.01*np.exp(self.x[k])), loc = np.exp(self.x[k]), scale = 0.01*np.exp(self.x[k]), size = self.nwalk)))
                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.01*np.exp(self.x[k]), self.nwalk)))

            if 'rv_offset' in k:
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(truncnorm.rvs((lb-self.x[k])/0.1, (ub-self.x[k])/0.1, loc = self.x[k], scale = 0.1, size = self.nwalk))
                else:
                    pos.append(np.random.normal(self.x[k], 0.1, self.nwalk))

            if k in ['u1','u2']:
                lb, ub = 0, 1
                if self.use_priors:
                    lb2, ub2 = self.allpriors.get_bounds(k)
                    lb = max(lb, lb2)
                    ub = min(ub, ub2)
                pos.append(truncnorm.rvs((lb-self.x[k])/0.01, (ub-self.x[k])/0.01, loc = self.x[k], scale = 0.01, size = self.nwalk))

            if k == 'eep':
                pos.append(np.random.normal(self.x[k], 0.1, self.nwalk))

            if k == 'log10(age)':
                if self.use_priors:
                    lb, ub = self.allpriors.get_bounds(k)
                    pos.append(np.log10(truncnorm.rvs((10**lb-10**self.x[k])/(10**self.x[k]/100), (10**ub-10**self.x[k])/(10**self.x[k]/100), loc = 10**self.x[k], scale = 10**self.x[k]/100, size = self.nwalk)))
                else:
                    pos.append(np.log10(np.random.normal(10**self.x[k], 10**self.x[k]/100, self.nwalk)))

            if k == 'feh':
                lb, ub = -0.5, 0.5
                pos.append(truncnorm.rvs((lb-self.x[k])/0.01, (ub-self.x[k])/0.01, loc = self.x[k], scale = 0.01, size = self.nwalk))

            if k == 'distance':
                pos.append(np.random.normal(self.x[k], 1, self.nwalk))

            if k == 'AV':
                lb, ub = 0, np.inf
                pos.append(truncnorm.rvs((lb-self.x[k])/0.01, (ub-self.x[k])/0.01, loc = self.x[k], scale = 0.01, size = self.nwalk))


        pos = np.transpose(np.array(pos))

        return pos


    def run_sampler(self, pos: np.typing.NDArray, skip_state_check: bool):
        """Not meant to be run on its own. Runs the MCMC sampler for burn-in and sample steps using emcee.

        Args:
            pos (ndarray): The initial positions of the chains generated by initialize_chains.

            skip_state_check (bool): Whether or not to skip the initial state check after the burn-in.
        """

        self.sampler = emcee.EnsembleSampler(nwalkers = self.nwalk, ndim = len(self.x), log_prob_fn = log_like, args = (self,), parameter_names = self.parnames, blobs_dtype = [('ps', np.ndarray), ('tcs', np.ndarray)])

        print('\nRunning MCMC burn-in.')

        prog = 'notebook' if is_notebook else True

        state = self.sampler.run_mcmc(pos, self.nburn, progress = prog)
        self.sampler.reset()

        print('\nRunning MCMC sampling.')

        self.sampler.run_mcmc(state, self.nrun, progress = prog, skip_initial_state_check = skip_state_check)

        self.samples = self.sampler.get_chain()

        self.log_likes = self.sampler.get_log_prob()

        if self.fit_transit and np.any(self.fit_ttv):

            self.blobs = self.sampler.get_blobs()


    def save_samples(self, name):
        """Saves the full sample chains out to a pickle file (not flattened or thinned). This is only automatically run if save_samples was set to True
        during fitting. Otherwise, can be run manually after a fit. This file takes up much more storage space than the normal thinned and flattened
        samples which are always saved. The pickle file is saved to Output/name_samples.p.

        Args:
            name (str): Name of the run. Sets the name of the pickle file to name_samples.p.
        """

        z = {'parnames': self.parnames, 'samples': self.samples, 'log_like': self.log_likes}

        if hasattr(self, 'blobs'):
            z['blobs'] = self.blobs

        pickle.dump(z, open(self.direc+'Output/'+name+'_samples.p', 'wb'))


    def continue_run(self, name: str, nrun: int, save_samples = False, show_plots = True, skip_state_check = False):
        """Continues running the MCMC sampler from where it left off, without a burn in. Remakes all results and plots, and saves to the provided name
        (does not have to be the same name as the previous run).

        Can only be run if a run has already been performed with this ExoSystem object. Will not work on a loaded in run.

        Args:
            name (str): Name of the fit. This name will be attached to all ouputs, including pickle files of data, human readable results tables,
                and the folder in Plots in which this run's plots will be saved. This name is also used for loading results back in to be manipulated
                or plotted again.

            nrun (int): Number of sampling steps for the MCMC. These steps are saved and used for results and making plots. They do not include the
                burn-in steps.

            save_samples (bool, optional): Whether or not to save the full, unflattened, un-thinned MCMC chains to a pickle file. This can be handy
                if you expect to want to remove problematic walkers that wandered off from the final results. Most of the time this isn't necessary,
                and these take up additional storage. Default is False.

            show_plots (bool, optional): Whether or not to show plots at the end of the run. Plots are saved regardless. Default is True.

            skip_state_check (bool, optional): Passed to the emcee sampler. Whether or not to skip checking whether the initial parameters can fully
                explore the space. Only set to True if you keep getting initial state check errors after burn in. Default is False.
        """

        print('\nRunning MCMC sampling.')

        state = self.samples[-1]
        self.sampler.run_mcmc(state, nrun, progress = True, skip_initial_state_check = skip_state_check)

        self.samples = self.sampler.get_chain()

        self.log_likes = self.sampler.get_log_prob()

        if self.fit_transit and np.any(self.fit_ttv):

            self.blobs = self.sampler.get_blobs()

        if save_samples:

            self.save_samples(name)

        self.calc_gelman_rubin()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('')
        self.restab.pprint_all()


    def flatten_chains(self):
        """Flattens and thins by a factor of 20 ExoSystem.samples, ExoSystem.log_likes, and ExoSystem.blobs (if it exists, stores linear regression periods and Tcs of
        TTV planets from each sample). Stores these as ExoSystem.flat_samples, ExoSystem.flat_log_likes, and ExoSystem.flat_blobs, respectively.
        """

        self.flat_samples = self.samples[19::20]
        shape = self.flat_samples.shape
        self.flat_samples = np.reshape(self.flat_samples, (shape[0]*shape[1], shape[2]))

        self.flat_log_likes = self.log_likes[19::20]
        shape = self.flat_log_likes.shape
        self.flat_log_likes = np.reshape(self.flat_log_likes, (shape[0]*shape[1]))

        if hasattr(self, 'blobs'):

            self.flat_blobs = self.blobs[19::20]
            shape = self.flat_blobs.shape
            self.flat_blobs = np.reshape(self.flat_blobs, (shape[0]*shape[1]))


    def make_results(self, name: str):
        """Converts flat_samples and flat_log_likes into a dict that stores the chains of parameters that were directly fit, as well as the log
        likelihood of each sample. Saves these to the pickle file Output/name_res.p. Accessible through ExoSystem.chains.

        Also creates a dict of flat chains of derived parameters (e.g. period from log(period), or planet equilibrium temperature). Saves these to the
        pickle file Output/name_dres.p. Accessible through ExoSystem.derived_chains.

        Combines results and derived_results into a single human-readable ascii table which is saved to Results/name.txt. The astropy table can be
        directly accesed through ExoSystem.results.

        Writes out specific run settings (fit_transit, fit_rv, fit_star, fit_ld, use_priors, lc_supersample_size, and fixed parameters if applicable)
        to the pickle file Output/name_run_settings.p. These are loaded back in with ExoSystem.load_results and are used to tell plotting functions
        what to plot.

        Args:
            name (str): Name of the run. Sets the names of the output files from this function.
        """

        if self.fit_transit and np.any(self.fit_ttv):

            ps = np.array([x for x in self.flat_blobs['ps']]).T
            tcs = np.array([x for x in self.flat_blobs['tcs']]).T

        self.res = {}

        for k, v in self.parnames.items():

            self.res[k] = self.flat_samples[:,v]

        self.res['log_like'] = self.flat_log_likes

        pickle.dump(self.res, open(self.direc+'Output/'+name+'_res.p', 'wb'))

        self.dres = {}

        n = len(self.res['log_like'])

        if self.fit_star:

            self.starmod._derived_samples = self.misti(*[self.res[x] if x in self.res else [self.fixed[x]]*n for x in ['eep','log10(age)','feh','distance','AV']])
            self.starmod._derived_samples["parallax"] = 1000.0 / self.res['distance']
            self.starmod._derived_samples["distance"] = self.res['distance']
            self.starmod._derived_samples["AV"] = self.res['AV']

            self.dres['rstar'] = self.starmod._derived_samples['radius']
            self.dres['mstar'] = self.starmod._derived_samples['mass']
            self.dres['rhostar'] = self.starmod._derived_samples['density']
            self.dres['mstar_init'] = self.starmod._derived_samples['initial_mass']
            self.dres['Tstar'] = self.starmod._derived_samples['Teff']
            self.dres['loggstar'] = self.starmod._derived_samples['logg']
            self.dres['Lstar'] = 10**self.starmod._derived_samples['logL']
            self.dres['Mbolstar'] = self.starmod._derived_samples['Mbol']

            interp_input = np.array([self.dres['Tstar'], self.dres['loggstar'], self.res['feh']]).T

            for filt in np.unique(self.filters):

                self.dres['u1 {0}'.format(filt)] = self.ldgrids[filt][0](interp_input)
                self.dres['u2 {0}'.format(filt)] = self.ldgrids[filt][1](interp_input)

        else:

            T = np.random.normal(self.Ts, self.Tserr, n)
            rstar = np.random.normal(self.rs, self.rserr, n)
            mstar = np.random.normal(self.ms, self.mserr, n)

        J = np.random.normal(self.Jmag, self.Jmagerr, n)
        einsol = 1*u.Lsun / (4 * np.pi * u.AU**2)

        if self.fit_planets:

            if self.use_priors:

                y = self.res | self.fixed

                for i in range(self.n):

                    if 'e {0}'.format(i+1) in y:

                        y['secw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.cos(y['w {0}'.format(i+1)])
                        y['sesw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.sin(y['w {0}'.format(i+1)])

            else:

                y = self.res.copy()

            for i in range(self.n):

                if self.is_transit[i] and self.fit_transit:

                    if self.fit_ttv[i]:

                        k = np.sum(self.fit_ttv[:i])

                        p = np.array(ps[k])
                        self.dres['P {0}'.format(i+1)] = p

                        tc = np.array(tcs[k])
                        self.dres['Tc {0}'.format(i+1)] = tc

                    else:

                        p = np.exp(y['log(P) {0}'.format(i+1)])
                        self.dres['P {0}'.format(i+1)] = p

                    rp = y['ror {0}'.format(i+1)] * rstar * (1*u.Rsun).to(u.earthRad).value
                    self.dres['Rp {0}'.format(i+1)] = rp

                    if not self.fit_star:

                        ars = np.exp(y['log(a/rs) {0}'.format(i+1)])
                        self.dres['a/rs {0}'.format(i+1)] = ars

                        a = ars * rstar * (1*u.Rsun).to(u.AU).value
                        self.dres['a {0}'.format(i+1)] = a

                    inc = np.arccos(y['cos(i) {0}'.format(i+1)]) * 180/np.pi
                    self.dres['i {0}'.format(i+1)] = inc


                elif self.is_rv[i] and self.fit_rv:

                    p = np.exp(y['log(P) {0}'.format(i+1)])
                    self.dres['P {0}'.format(i+1)] = p

                if self.fit_ecc[i]:

                    if 'e {0}'.format(i+1) not in y:

                        e = y['secw {0}'.format(i+1)]**2 + y['sesw {0}'.format(i+1)]**2
                        self.dres['e {0}'.format(i+1)] = e

                        w = np.arctan2(y['sesw {0}'.format(i+1)], y['secw {0}'.format(i+1)])
                        self.dres['w {0}'.format(i+1)] = w

                    else:

                        e = y['e {0}'.format(i+1)]
                        w = y['w {0}'.format(i+1)]

                else:

                    e = 0
                    w = np.pi/2

                mp = 0

                if self.is_rv[i] and self.fit_rv:

                    k = np.exp(y['log(K) {0}'.format(i+1)])
                    self.dres['K {0}'.format(i+1)] = k

                    if self.is_transit[i] and self.fit_transit:

                        mp = calc_m_from_k(p*(1*u.day).to(u.yr).value, k, e, inc*np.pi/180, mstar)
                        self.dres['Mp {0}'.format(i+1)] = mp

                        rhop = mp / (4/3 * np.pi * rp**3) * (1*u.earthMass/u.earthRad**3).to(u.g/u.cm**3).value
                        self.dres['rhop {0}'.format(i+1)] = rhop

                    else:

                        mp = calc_m_from_k(p*(1*u.day).to(u.yr).value, k, e, np.pi/2, mstar)
                        self.dres['Mpsini {0}'.format(i+1)] = mp

                if self.fit_star or (self.is_rv[i] and self.fit_rv):

                    a = ((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3)
                    self.dres['a {0}'.format(i+1)] = a

                    ars = (a*u.AU).to(u.Rsun).value / rstar
                    self.dres['a/rs {0}'.format(i+1)] = ars

                if (self.fit_transit and self.is_transit[i]) or (self.fit_rv and self.is_rv[i]):

                    teq = (1/4)**(1/4) * T * ars**(-1/2)
                    self.dres['teq {0}'.format(i+1)] = teq

                    sinc = (constants.sigma_sb * (T * u.K)**4 * ars**(-2) / einsol).to(u.dimensionless_unscaled)
                    self.dres['sinc {0}'.format(i+1)] = sinc

                if self.is_transit[i] and self.fit_transit:

                    b = ars * y['cos(i) {0}'.format(i+1)] * (1 - e**2) / (1 + e * np.sin(w))
                    self.dres['b {0}'.format(i+1)] = b

                    depth = y['ror {0}'.format(i+1)]**2 * 1e6
                    self.dres['depth {0}'.format(i+1)] = depth

                    dur = p / np.pi * np.arcsin(np.sqrt((1 + y['ror {0}'.format(i+1)])**2 - b**2) / (ars * np.sqrt(1 - y['cos(i) {0}'.format(i+1)]**2))) * (1*u.day).to(u.hr).value
                    self.dres['dur {0}'.format(i+1)] = dur

                    if not self.fit_star:

                        rhos = 0.018916375 * ars**3 / p**2
                        self.dres['rhos {0}'.format(i+1)] = rhos

                    if self.is_rv[i] and self.fit_rv:

                        sf = np.full(n, 0.19)
                        j = np.where(rp > 1.5)[0]
                        sf[j] = 1.26
                        j = np.where(rp > 2.75)[0]
                        sf[j] = 1.28
                        j = np.where(rp > 4)[0]
                        sf[j] = 1.15

                        tsm = sf * rp**3 * teq / (mp * rstar**2) * 10**(-J/5)
                        self.dres['TSM {0}'.format(i+1)] = tsm


        if not self.fit_star:
            self.dres['rstar'] = rstar
            self.dres['mstar'] = mstar
            self.dres['Tstar'] = T


        pickle.dump(self.dres, open(self.direc+'Output/'+name+'_dres.p', 'wb'))


        par_units = {'log(P)': 'days', 'Tc': 'BJD-2450000', 'ror': '', 'log(a/rs)': '', 'cos(i)': '', 'log(K)': 'm/s', 'secw': '', 'sesw': '',
                     'fp': '', 'TT': 'BJD-2450000', 'F0': '', 'log(rho_gp)': 'days', 'log(sigma_gp)': '', 'gamma': 'm/s', 'gamma_dot': 'm/s/day',
                     'gamma_ddot': 'm/s/day^2', 'rv_offset': 'm/s', 'u1': '', 'u2': '', 'eep': '', 'log10(age)': 'yr', 'feh': 'dex', 'distance': 'pc',
                     'AV': 'mag', 'rstar': 'Rsun', 'mstar': 'Msun', 'rhostar': 'g/cm^3', 'mstar_init': 'Msun', 'Tstar': 'K', 'loggstar': 'cm/s^2',
                     'Lstar': 'Lsun', 'Mbolstar': 'mag', 'P': 'days', 'Rp': 'Rearth', 'a/rs': '', 'a': 'AU', 'i': 'deg', 'e': '', 'w': 'rad',
                     'K': 'm/s', 'Mp': 'Mearth', 'rhop': 'g/cm^3', 'Mpsini': 'Mearth', 'teq': 'K', 'sinc': 'Searth', 'b': '', 'depth': 'ppm', 'dur': 'hr',
                     'rhos': 'g/cm^3', 'TSM': ''}


        out = []
        for x in self.res:
            if x == 'log_like':
                continue
            out.append([x, par_units[x.split()[0]], np.nanmedian(self.res[x])]+list(np.diff(np.nanpercentile(self.res[x], [16,50,84]))))
        for x in self.dres:
            out.append([x, par_units[x.split()[0]], np.nanmedian(self.dres[x])]+list(np.diff(np.nanpercentile(self.dres[x], [16,50,84]))))

        tab = Table(rows = out, names = ['Parameter','Units','Median','-Error','+Error'])
        self.restab = tab
        tab.write(self.direc+'Results/'+name+'.txt', format = 'ascii.fixed_width_two_line', overwrite = True, delimiter = '|', delimiter_pad = ' ', bookend = True)

        run_settings = {'fit_transit': self.fit_transit, 'fit_rv': self.fit_rv, 'fit_star': self.fit_star, 'fit_ld': self.fit_ld, 'use_priors': self.use_priors, 'lc_supersample_size': self.lc_supersample_size}
        if self.use_priors:
            run_settings['fixed'] = self.fixed

        pickle.dump(run_settings, open(self.direc+'Output/'+name+'_run_settings.p', 'wb'))


    def make_plots(self, name: str, show_plots: bool = True):
        """Generates plots for the run, including chains plots, corner plots, and plots of best fit models. When plotting best fit models, also runs
        the corresponding function to generate those best fit models.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save the plots to.

            show_plots (bool, optional): Whether or not to show plots. Plots are saved regardless. Default is True.
        """

        if self.fit_planets:

            self.plot_pl_chains(name, show_plot = show_plots)

        if self.nttv > 0 and self.fit_transit:

            self.plot_ttv_chains(name, show_plot = show_plots)

        if np.any(self.detrend) and self.fit_transit:

            self.plot_det_chains(name, show_plot = show_plots)

        if self.fit_star:

            self.plot_star_chains(name, show_plot = show_plots)

        if self.fit_planets:

            self.plot_big_chains(name)

        if self.fit_planets:
        
            self.plot_pl_corner(name, show_plot = show_plots)

        if self.fit_star:

            self.plot_star_corner(name, show_plot = show_plots)

        if self.fit_planets:

            self.plot_big_corner(name)

        if self.fit_transit:

            self.gen_lcs(name)

            self.plot_full_lc(name, show_plot = show_plots)

            self.plot_lc_phase(name, show_plot = show_plots)

        if self.fit_rv:

            self.gen_rv(name)

            self.plot_rv(name, show_plot = show_plots)

        if self.nttv > 0 and self.fit_transit:

            self.plot_ttvs(name, show_plot = show_plots)

        if self.fit_star:

            self.gen_magfit(name)

            self.plot_sed_fit(name, show_plot = show_plots)

    @property
    def results(self):
        """Astropy table storing median and 1 sigma (16th and 84th percentile) values for fit parameters and derived parameters.

        Use the chains object or derived chains object for flattened chains of these parameters, as well as a description of them in the docstrings.
        """
        return self.restab

    @property
    def chains(self):
        """Dict which stores the flattened chains of fit parameters. Keys are the parameter names.

        Potential fit parameters:

        - log(P) x: Natural log of the orbital period, in days, for planet number x.
        - Tc x: Time of conjunction, in BJD-2457000, for planet number x.
        - ror x: Planet to star radius ratio, for planet number x.
        - log(a/rs) x: Natural log of the orbital semi-major axis to stellar radius ratio, for planet number x.
        - cos(i) x: Cosine of the orbital inclination, restricted between 0 and 90 degrees, for planet number x.
        - log(K) x: Natural log of the RV semi-amplitude, in m/s, for planet number x.
        - secw x: Square root of the eccentricity times the cosine of the argument of periastron, for planet number x.
        - sesw x: Square root of the eccentricity times the sine of the argument of periastron, for planet number x.
        - e x: Orbital eccentricity, for planet number x. Only fit if w is fixed.
        - w x: Orbital argument of periastron, in radians, for planet number x. Only fit if e is fixed.
        - fp x: Planet to star flux ratio, for planet number x.
        - TT x y: Transit time (when fitting for TTVs), in BJD-2457000, for planet number x transit number y.
        - F0 x: Baseline flux value, for transit data set x.
        - log(rho_gp) x: Natural log of the GP period, in days, for transit data set x.
        - log(sigma_gp) x: Natural log of the GP standard deviation, for transit data set x.
        - gamma: RV systemic velocity, in m/s.
        - gamma_dot: 1st derivative of the RV systemic velocity, in m/s/day. Relative to time of first RV measurement.
        - gamma_ddot: 2nd derivative of the RV systemic velocity, in m/s/day^2. Relative to time of first RV measurement.
        - rv_offset x: RV offset relative to first data set, in m/s, for RV data set x.
        - u1 x: Linear coefficient for quadratic limb darkening, for filter x.
        - u2 x: Non-linear coefficient for quadratic limb darkening, for filter x.
        - eep: Equivalent evolutionary phase, used to interpolate across MIST models for stellar fitting.
        - log10(age): Log base 10 of the stellar age, in yr.
        - feh: Stellar metallicity, in dex.
        - distance: Distance to the system, in pc.
        - AV: Visual band extinction to the system, in mags.
        """
        return self.res
    
    @property
    def derived_chains(self):
        """Dict which stores flat chains of derived parameters. Keys are the parameter names.

        Potential derived parameters:
    
        - P x: Orbital period, in days, for planet number x.
        - Rp x: Planetary radius, in earth radii, for planet number x.
        - a/rs x: Orbital semi-major axis to stellar radius ratio, for planet number x.
        - a x: Orbital semi-major axis, in AU, for planet number x.
        - i x: Orbital inclination, in degrees, for planet number x (restricted between 0 and 90 degrees).
        - e x: Orbital eccentricity, for planet number x.
        - w x: Orbital argument of periastron, in radians, for planet number x.
        - K x: RV semi-amplitude, in m/s, for planet number x.
        - Mp x: Planetary mass, in earth masses, for planet number x.
        - Mpsini x: Planetary mass times the sine of the orbital inclination, in earth masses, for planet number x (used when transit wasn't fit).
        - rhop x: Planetary mean density, in g/cm^3, for planet number x.
        - teq x: Planetary equilibrium temperature (zero-albedo blackbody), in K, for planet number x.
        - sinc x: Insolation flux reaching the planet, in units of Earth insolation flux, for planet number x.
        - b x: Transit impact parameter, for planet number x.
        - depth x: Transit depth, in ppm, for planet number x.
        - dur x: Transit duration, in hours, for planet number x.
        - rhos x: Implied stellar density from transit of planet number x, in g/cm^3.
        - TSM x: Transmission spectroscopy metric, for planet number x.
        - rstar: Stellar radius, in solar radii.
        - mstar: Stellar mass, in solar masses.
        - mstar_init: Initial stellar mass at the zero-age main sequence point, in solar masses.
        - rhostar: Mean stellar density from stellar fitting, in g/cm^3.
        - Tstar: Stellar effective temperature, in K.
        - loggstar: log10 of the stellar surface gravity, in cm/s^2.
        - Lstar: Stellar bolometric luminosity, in solar luminosities.
        - Mbolstar: Stellar bolometric magnitude, in mags.
        """
        return self.dres


    def load_results(self, name: str):
        """Loads in the results from a previous run. Flattened chains will be accessable in ExoSystem.chains and ExoSystem.derived_chains.

        Also loads the median results table
        
        If the run includes a transit fit, loads in and applies masks to the light curves.
        
        Attempts to load in best fit transit models and RV models (if they exist) to ExoSystem.fm and ExoSystem.rvm, respectively.
        
        Attempts to load in SED best fit magnitudes and errors (if they exist) into ExoSystem.magfit and ExoSystem.magfiterr, respectively.
        
        Attempts to load in unflattened, un-thinned MCMC chains if they exist. Paramater name map is saved to ExoSystem.parnames,
        chains are saved to ExoSystem.samples, log likelihood values are saved to ExoSystem.log_likes_full.

        Also loads in some run settings from the run.

        Args:
            name (str): Name of the previous run to load in.
        """

        try:
            self.res = pickle.load(open(self.direc+'/Output/'+name+'_res.p', 'rb'))
            self.dres = pickle.load(open(self.direc+'/Output/'+name+'_dres.p', 'rb'))
            self.load_results_tab(name)
        
        except:
            print('No run named {0}.'.format(name))
            return None
    
        if os.path.exists(self.direc+'/Masks/'+name+'_masks.p'):
            
            self.masks = pickle.load(open(self.direc+'/Masks/'+name+'_masks.p', 'rb'))

            self.tt = copy.deepcopy(self.tt_orig)
            self.f = copy.deepcopy(self.f_orig)
            self.ferr = copy.deepcopy(self.ferr_orig)

            for i in range(len(self.tt)):
                self.tt[i] = self.tt[i][self.masks[i]]
                self.f[i] = self.f[i][self.masks[i]]
                self.ferr[i] = self.ferr[i][self.masks[i]]

        try:
            self.load_lcs_mod(name)
        except:
            pass

        try:
            self.load_rv_mod(name)
        except:
            pass

        try:
            self.load_magfit(name)
        except:
            pass

        try:
            self.load_samples(name)
        except:
            pass
        
        try:
            self.load_run_settings(name)
        except:
            pass


    def load_results_tab(self, name: str):
        """Loads in the median results table from Results/name.txt as an astropy table accessible through ExoSystem.results.

        Args:
            name (str): Name of the previous run to load in.
        """

        self.restab = Table.read(self.direc+'Results/'+name+'.txt', format = 'ascii.fixed_width_two_line', delimiter = '|', header_rows = ['name'])

        
    def load_lcs_mod(self, name: str):
        """Loads in the best fit transit model of a previous run into ExoSystem.fm.

        Args:
            name (str): Name of the previous run to load in.
        """

        self._lcm = pickle.load(open(self.direc+'/Output/'+name+'_lcm.p', 'rb'))


    def load_rv_mod(self, name: str):
        """Loads in the best fit RV model of a previous run into ExoSystem.rvm.

        Args:
            name (str): Name of the previous run to load in.
        """

        self._rvm = pickle.load(open(self.direc+'/Output/'+name+'_rvm.p', 'rb'))


    def load_magfit(self, name: str):
        """Loads in the best fit stellar SED model of a previous run into ExoSystem.magfit and ExoSystem.magfiterr.

        Args:
            name (str): Name of the previous run to load in.
        """

        mags = pickle.load(open(self.direc+'/Output/'+name+'_magfit.p', 'rb'))
        self.magfit = mags['magfit']
        self.magfiterr = mags['magfiterr']

    
    def load_samples(self, name: str):
        """Loads in unflattened, un-thinned MCMC chains. Paramater name map is saved to ExoSystem.parnames, chains are saved to ExoSystem.samples,
        log likelihood values are saved to ExoSystem.log_likes. If any TTVs were fit, linear regression periods and Tcs for each sample are saved to
        Exosystem.blobs.

        Args:
            name (str): Name of the previous run to load in.
        """

        z = pickle.load(open(self.direc+'/Output/'+name+'_samples.p', 'rb'))
        self.parnames = z['parnames']
        self.samples = z['samples']
        self.log_likes = z['log_like']
        if 'blobs' in z:
            self.blobs = z['blobs']

    
    def load_run_settings(self, name: str):
        """Loads in run settings from the run, specifically fit_transit, fit_rv, fit_star, fit_ld, use_priors, lc_supersample_size, and any parameters
        which were fixed. These are used to tell plotting functions what they need to plot.

        Args:
            name (str): Name of the previous run to load in.
        """
        
        run_settings = pickle.load(open(self.direc+'/Output/'+name+'_run_settings.p', 'rb'))
        self.fit_transit = run_settings['fit_transit']
        self.fit_rv = run_settings['fit_rv']
        self.fit_star = run_settings['fit_star']
        self.fit_ld = run_settings['fit_ld']
        self.use_priors = run_settings['use_priors']
        self.lc_supersample_size = run_settings['lc_supersample_size']
        if self.use_priors:
            self.fixed = run_settings['fixed']
        self.fit_planets = self.fit_transit or self.fit_rv


    def delete_run(self, name: str):
        """Searches for files from a run of the given name and deletes them.

        Args:
            name (str): Name of the previous run to delete.
        """

        if os.path.exists(self.direc+'Masks/'+name+'_masks.p'):
            os.remove(self.direc+'Masks/'+name+'_masks.p')

        if os.path.isdir(self.direc+'Plots/'+name):
            shutil.rmtree(self.direc+'Plots/'+name)

        if os.path.exists(self.direc+'Output/'+name+'_res.p'):
            os.remove(self.direc+'Output/'+name+'_res.p')

        if os.path.exists(self.direc+'Output/'+name+'_dres.p'):
            os.remove(self.direc+'Output/'+name+'_dres.p')

        if os.path.exists(self.direc+'Output/'+name+'_lcm.p'):
            os.remove(self.direc+'Output/'+name+'_lcm.p')

        if os.path.exists(self.direc+'Output/'+name+'_rvm.p'):
            os.remove(self.direc+'Output/'+name+'_rvm.p')

        if os.path.exists(self.direc+'Output/'+name+'_samples.p'):
            os.remove(self.direc+'Output/'+name+'_samples.p')

        if os.path.exists(self.direc+'Output/'+name+'_magfit.p'):
            os.remove(self.direc+'Output/'+name+'_magfit.p')

        if os.path.exists(self.direc+'Output/'+name+'_run_settings.p'):
            os.remove(self.direc+'Output/'+name+'_run_settings.p')

        if os.path.exists(self.direc+'Results/'+name+'.txt'):
            os.remove(self.direc+'Results/'+name+'.txt')

        if os.path.isdir(self.direc+'multinest chains/'+name):
            shutil.rmtree(self.direc+'multinest chains/'+name)


    def initialize_ttvs(self, name: str):
        """Loads in an Init_ttvs file if it exists, otherwise prompts the user to create one. Then, sets up initial variables for fitting ttvs.
        These are:
        - self.ttvs0: A dict with keys being planet numbers, and the values being a sorted array of all observed transit times.
        - self.ttvsectors: A dict with keys being the planet number followed by the transit number (1-indexed), and the values being the index of the
        transit data set which the transit occurs in.
        - self.ttvi: A dict with keys being planet numbers, and the values being an array of integers representing the number of periods occurring
        between the first observed transit and each transit being fit.

        Args:
            name (str): Name of the Init_ttvs file.
        """

        if not os.path.exists(self.direc+'/'+name):

            tab = Init_ttvs(self.direc, name = name).create().table

        else:

            tab = Init_ttvs(self.direc, name = name).from_file().table

        self.ttvs0 = {int(col): np.sort(np.array(tab[col])[~np.isnan(tab[col])]) for col in tab.columns}

        self.ttvsectors = {}
        self.ttvi = {}

        for i in range(self.n):

            if self.is_transit[i] and self.fit_ttv[i]:

                if i+1 not in self.ttvs0:

                    print('Need to input transit times in init_ttv file for planet {0}, which is set to fit_ttv = True. Either edit the file or run create_init_ttvs.')
                    break

                del_ind = []
                sectors = []

                for z in range(len(self.ttvs0[i+1])):

                    found = False

                    for l in range(len(self.tt)):

                        if np.min(self.tt[l]) <= self.ttvs0[i+1][z] <= np.max(self.tt[l]):

                            sectors.append(l)
                            found = True
                            break

                    if not found:

                        del_ind.append(z)

                self.ttvs0[i+1] = np.delete(self.ttvs0[i+1], del_ind)
                
                for z in range(len(sectors)):
                    self.ttvsectors['{0} {1}'.format(i+1, z+1)] = sectors[z]

                ttvi0 = np.round((np.array(self.ttvs0[i+1]) - self.ttvs0[i+1][0]) / self.p[i], 0).astype(int)
                self.ttvi['{0}'.format(i+1)] = ttvi0


    def plot_pl_chains(self, name: str, show_plot = True):
        """Plots the MCMC chains for the planetary parameters. Saves the plot as pl_chains.png to the folder for this run, set by the name parameter.
        Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """


        num1, num2 = 0, 0
        vargrid = []

        for i in range(self.n):

            if not ((self.fit_transit and self.is_transit[i]) or (self.fit_rv and self.is_rv[i])):

                continue

            num2 += 1
            vargrid.append([])

            for z in ['log(P)','Tc','ror','log(a/rs)','cos(i)','log(K)','secw','sesw','e','w','fp']:

                if z+' {0}'.format(i+1) in self.parnames:
                    vargrid[-1].append(z)

            num1 = max(num1, len(vargrid[-1]))

        fig, ax = plt.subplots(num1, num2, figsize = (7*num2, 18), sharex = True, layout = 'constrained')

        if num2 == 1:
            ax = np.array([ax.T]).T

        j = 0
        for i in range(self.n):

            if not ((self.fit_transit and self.is_transit[i]) or (self.fit_rv and self.is_rv[i])):

                continue

            ax[0][j].set_title('Planet {0}'.format(i+1))

            for k in range(len(vargrid[j])):

                v = self.parnames[vargrid[j][k]+' {0}'.format(i+1)]
                ax[k][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
                ax[k][j].text(0.01, 0.99, vargrid[j][k], fontsize = 20, ha = 'left', va = 'top', transform = ax[k][j].transAxes, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

            j += 1

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/pl_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_ttv_chains(self, name: str, show_plot = True):
        """Plots the MCMC chains for the planetary transit times of planets with TTVs. Saves the plot as tt_chains.png to the folder for this run,
        set by the name parameter. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        n = 0
        for x in self.ttvi:
            n = max(n,len(self.ttvi[x]))

        fig, ax = plt.subplots(n, self.nttv, figsize = (7*self.nttv, 12), sharex = True, layout = 'constrained')

        if self.nttv == 1:
            ax = np.array([ax.T]).T

        for i in range(self.n):

            if not self.is_transit[i]:
                continue

            if not self.fit_ttv[i]:
                continue

            k = np.sum(self.fit_ttv[:i])

            ax[0][k].set_title('Planet {0}'.format(i+1))

            for z in range(len(self.ttvi['{0}'.format(i+1)])):
                
                if 'TT {0} {1}'.format(i+1, z+1) in self.parnames:
                    v = self.parnames['TT {0} {1}'.format(i+1, z+1)]
                    ax[z][k].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)


        fig.supxlabel('N Steps')
        fig.supylabel('Transit Times')

        fig.savefig(self.direc+'Plots/'+name+'/tt_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_det_chains(self, name: str, show_plot = True):
        """Plots the MCMC chains for the light curve detrending GP hyper parameters. Saves the plot as gp_chains.png to the folder for this run,
        set by the name parameter. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        fig, ax = plt.subplots(np.sum(self.detrend), 3, figsize = (15, 3*len(self.tt)), sharex = True, layout = 'constrained')

        if np.sum(self.detrend) == 1:
            ax = np.array([ax])

        for j in range(len(self.tt)):

            if not self.detrend[j]:
                continue

            i = np.sum(self.detrend[:j])

            ax[i][0].set_ylabel('{0}'.format(self.lcnames[j]))

            if 'F0 {0}'.format(self.lcnames[j]) in self.parnames:
                v = self.parnames['F0 {0}'.format(self.lcnames[j])]
                ax[i][0].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            if 'log(rho_gp) {0}'.format(self.lcnames[j]) in self.parnames:
                v = self.parnames['log(rho_gp) {0}'.format(self.lcnames[j])]
                ax[i][1].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            if 'log(sigma_gp) {0}'.format(self.lcnames[j]) in self.parnames:
                v = self.parnames['log(sigma_gp) {0}'.format(self.lcnames[j])]
                ax[i][2].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

        ax[0][0].set_title('F0')
        ax[0][1].set_title('log(rho_gp)')
        ax[0][2].set_title('log(sigma_gp)')

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/gp_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_star_chains(self, name: str, show_plot = True):
        """Plots the MCMC chains for the stellar parameters. Saves the plot as star_chains.png to the folder for this run, set by the name parameter.
        Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        fig, ax = plt.subplots(5, figsize = (7, 12), sharex = True, layout = 'constrained')

        for i, z in enumerate(['eep','log10(age)','feh','distance','AV']):

            if z in self.parnames:
                v = self.parnames[z]
                ax[i].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
                ax[i].set_ylabel(z)

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/star_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_big_chains(self, name: str, show_plot = False):
        """Plots the MCMC chains for every fit parameter. Saves the plot as bigchains.png to the folder for this run, set by the name parameter.
        Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is False.
        """
         
        num1, num2 = min(len(self.parnames),10), (len(self.parnames)-1)//10 + 1

        fig, ax = plt.subplots(num1, num2, figsize = (7*num2, 18*num1/10), sharex = True, layout = 'constrained')

        if num2 == 1:
            ax = np.array([ax.T]).T

        for k, v in self.parnames.items():

            i = v%10
            j = v//10

            ax[i][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
            ax[i][j].text(0.01, 0.99, k, fontsize = 20, ha = 'left', va = 'top', transform = ax[i][j].transAxes, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/bigchains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_pl_corner(self, name: str, show_plot = True):
        """Creates corner plots of the MCMC posteriors for each planet's parameters. Saves the plots as p#_corner.png (where # is the planet number)
        to the folder for this run, set by the name parameter. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
           
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        for i in range(self.n):

            labels = []

            for z in ['log(P)','Tc','ror','log(a/rs)','cos(i)','log(K)','secw','sesw','e','w','fp']:

                if z+' {0}'.format(i+1) in self.parnames:

                    labels.append(z+' {0}'.format(i+1))

            if self.fit_transit and self.is_transit[i] and self.fit_ttv[i]:

                for k in range(len(self.ttvi['{0}'.format(i+1)])):

                    if 'TT {0} {1}'.format(i+1, k+1) in self.parnames:
                        labels.append('TT {0} {1}'.format(i+1, k+1))

            if not labels:

                continue

            j = [self.parnames[x] for x in labels]
            
            fig = corner.corner(self.flat_samples[:,j], labels = labels)

            fig.suptitle('Planet {0}'.format(i+1))

            plt.tight_layout()

            fig.savefig(self.direc+'Plots/'+name+'/p{0}_corner.png'.format(i+1))

            if show_plot:
                plt.show()

            else:
                plt.close()


    def plot_star_corner(self, name: str, show_plot = True):
        """Creates a corner plot of the MCMC posterior for the stellar parameters. Saves the plot as star_corner.png to the folder for this run,
        set by the name parameter. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        labels = []

        for z in ['eep','log10(age)','feh','distance','AV']:
            if z in self.parnames:
                labels.append(z)

        j = [self.parnames[x] for x in labels]

        fig = corner.corner(self.flat_samples[:,j], labels = labels)

        fig.suptitle('Star Fit Params')

        plt.tight_layout()

        fig.savefig(self.direc+'Plots/'+name+'/star_corner_fit.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


        fig = self.starmod.corner_observed()
        
        fig.suptitle('Star Observed Params')

        plt.tight_layout()

        fig.savefig(self.direc+'Plots/'+name+'/star_corner_observed.png')

        if show_plot:
            plt.show()
        else:
            plt.close()


        fig = self.starmod.corner_physical()

        fig.suptitle('Star Physical Params')

        plt.tight_layout()

        fig.savefig(self.direc+'Plots/'+name+'/star_corner_physical.png')

        if show_plot:
            plt.show()
        else:
            plt.close()


    def plot_big_corner(self, name: str, show_plot = False):
        """Creates a corner plot of the MCMC posterior of every parameter. Saves the plot as bigcorner.png to the folder for this run,
        set by the name parameter. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is False.
        """

        keys = list(self.res.keys())
        keys.remove('log_like')

        samps = np.array([self.res[keys[i]] for i in range(len(keys))]).T

        fig = corner.corner(samps, labels = keys)

        fig.savefig(self.direc+'Plots/'+name+'/bigcorner.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def gen_lcs(self, name: str):
        """Generates model light curves from the best fit parameters. Saves these to the pickle file Output/name_lcm.p. These model light curves are
        accessible through ExoSystem.lc_mod. See the documentation on ExoSystem.lc_mod for details on the formatting of the object.

        Args:
            name (str): Name of the run. Sets the name of the output pickle file to name_lcm.p.
        """

        print('\nGenerating light curves for plots.')

        self.supersamples = np.array([max(1,int(x/self.lc_supersample_size)) for x in self.exptimes*24*60*60])

        self._lcm = {}

        if np.any(self.fit_ttv):
            self.initialize_ttvs(name = self.init_ttvs_name)

        y = self.res | self.dres

        n = len(y['log_like'])

        if self.use_priors:

            f = self.fixed.copy()

            for ff in f:

                f[ff] = np.full((n), f[ff])

            y = y | f

            for i in range(self.n):

                if 'secw {0}'.format(i+1) not in y and 'e {0}'.format(i+1) in y:

                    y['secw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.cos(y['w {0}'.format(i+1)])
                    y['sesw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.sin(y['w {0}'.format(i+1)])

        if self.fit_star:

            self.ldgrids = {}

            for filt in np.unique(self.filters):

                try:
                    grid = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+'/ld_grids/{0}_grid.p'.format(filt), 'rb'))

                except:
                    print('No limb darkening grid available for filter {0}.'.format(filt))
                    print('Please use one of the following filters with limb darkening grids available:\n{0}'.format(ld_grid_list))
                    print('Or run generate_ld_grid to create a new grid using a valid svo filter name and nickname {0}.'.format(filt))
                    return

                interpu1 = LinearNDInterpolator(grid['coords'], grid['u1'])
                interpu2 = LinearNDInterpolator(grid['coords'], grid['u2'])

                self.ldgrids[filt] = [interpu1, interpu2]

            self.misti = get_ichrone('mist')

            rstar, mstar, Tstar, loggstar = self.misti.interp_value([y[x] for x in ['eep','log10(age)','feh']],['radius','mass','Teff','logg']).T

            arlist = []
            for j in range(self.n):

                if not self.is_transit[j]:
                    arlist.append(np.nan)
                    continue

                p = y['P {0}'.format(j+1)]

                if self.is_rv[j] and self.fit_rv:

                    e = 0
                    if self.fit_ecc[j]:
                        e = y['e {0}'.format(j+1)]

                    mp = calc_m_from_k(p, y['K {0}'.format(j+1)], e, np.arccos(y['cos(i) {0}'.format(j+1)]), mstar)

                else:

                    mp = 0

                ar = (((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                arlist.append(ar)

        pars = []
        tcs = []

        for j in range(self.n):

            if not self.is_transit[j]:
                continue
            
            if self.fit_ttv[j]:

                p = y['P {0}'.format(j+1)]
                tc = y['Tc {0}'.format(j+1)]
                ror = y['ror {0}'.format(j+1)]
                if self.fit_star:
                    ar = arlist[j]
                else:
                    ar = y['a/rs {0}'.format(j+1)]
                inc = y['i {0}'.format(j+1)]
                e = y['e {0}'.format(j+1)] if 'e {0}'.format(j+1) in y else np.full((n), 0)
                w = y['w {0}'.format(j+1)]*180/np.pi if 'w {0}'.format(j+1) in y else np.full((n), 90)
                
                pars.append(np.array([p,tc,ror,ar,inc,e,w]).T)

                tcs.append(np.median(tc))

            else:

                pars.append(get_transit_params(y, j+1, ar = arlist[j] if self.fit_star else None).T)
                tcs.append(np.median(y['Tc {0}'.format(j+1)]))


        filts = np.unique(self.filters)

        ttphase = np.linspace(-0.5, 0.5, 1000)
        phaseexp = 1/1000
        phasess = 1

        self._lcm['phase'] = {'ttphase': ttphase}

        print('\nPhase folded:\n')

        for filt in filts:

            print(filt+':')

            ld = np.array(self.ld[filt])

            if self.fit_ld:

                ld = np.array([y['u1 {0}'.format(filt)], y['u2 {0}'.format(filt)]]).T

            if self.fit_star:

                starpars = np.array([Tstar, loggstar, y['feh']]).T
                ld = np.array([self.ldgrids[filt][0](starpars)[0], self.ldgrids[filt][1](starpars)[0]]).T

            fmphase = []
            eclphase = []

            for k in range(self.n):

                if not self.is_transit[k]:
                    continue

                l = np.sum(self.is_transit[:k])

                medpar = np.median(pars[l], axis = 0)
                fmphase.append(lightCurve(medpar, ttphase + tcs[l], np.median(ld, axis = 0) if ld.ndim > 1 else ld, phaseexp, phasess))

                if self.is_eclipse[k]:
                    tc2 = calc_t_sec(medpar[0], medpar[1], medpar[5], medpar[6]*np.pi/180, medpar[3], np.median(rstar) if self.fit_star else self.rs)
                    eclphase.append(lightCurve(medpar, ttphase + tc2, np.median(ld, axis = 0) if ld.ndim > 1 else ld, phaseexp, phasess, eclipse = True, fp = np.median(y['fp {0}'.format(k+1)]), rstar = np.median(rstar) if self.fit_star else self.rs))


            fmphase_err = [[] for j in range(self.nt)]
            eclphase_err = [[] for j in range(np.sum(self.is_eclipse))]

            for j in tqdm(range(0,n,10)):

                for k in range(self.n):

                    if not self.is_transit[k]:
                        continue

                    l = np.sum(self.is_transit[:k])

                    fmphase_err[l].append(lightCurve(pars[l][j], ttphase + tcs[l], ld[j] if ld.ndim > 1 else ld, phaseexp, phasess))

                    if self.is_eclipse[k]:
                        tc2 = calc_t_sec(pars[l][j][0], pars[l][j][1], pars[l][j][5], pars[l][j][6]*np.pi/180, pars[l][j][3], rstar[j] if self.fit_star else self.rs)
                        le = np.sum(self.is_eclipse[:k])
                        eclphase_err[le].append(lightCurve(pars[l][j], ttphase + tc2, ld[j] if ld.ndim > 1 else ld, phaseexp, phasess, eclipse = True, fp = y['fp {0}'.format(k+1)][j], rstar = rstar[j] if self.fit_star else self.rs))

            fmphase_err = np.percentile(fmphase_err, [16,84], axis = 1)
            if np.any(self.is_eclipse):
                eclphase_err = np.percentile(eclphase_err, [16,84], axis = 1)
            
            self._lcm['phase'][filt] = {'fmphase': np.array(fmphase), 'fmphase_err': fmphase_err, 'eclphase': np.array(eclphase), 'eclphase_err': eclphase_err}

        print('Full:\n')

        for i in range(len(self.lcnames)):

            sec = self.lcnames[i]

            print(sec+':')

            ld = np.array(self.ld[self.filters[i]])

            if self.fit_ld:

                ld = np.array([y['u1 {0}'.format(self.filters[i])], y['u2 {0}'.format(self.filters[i])]]).T

            if self.fit_star:

                starpars = np.array([Tstar, loggstar, y['feh']]).T
                ld = np.array([self.ldgrids[self.filters[i]][0](starpars)[0], self.ldgrids[self.filters[i]][1](starpars)[0]]).T

            detrend = self.detrend[i]

            mean = y['F0 {0}'.format(sec)]

            if detrend:

                rhogp = np.exp(y['log(rho_gp) {0}'.format(sec)])
                sigmagp = np.exp(y['log(sigma_gp) {0}'.format(sec)])

                kernel = terms.SHOTerm(rho = rhogp[0], sigma = sigmagp[0], Q = 1/np.sqrt(2))
                gp = GaussianProcess(kernel = kernel)

                gpf_err = []

            fm = []

            fmsum = 0

            for k in range(self.n):

                if not self.is_transit[k]:
                    continue

                l = np.sum(self.is_transit[:k])

                fm0 = np.zeros(len(self.tt[i]))

                if self.fit_ttv[k]:

                    p = np.median(pars[l][:,0])

                    for z in range(len(self.ttvi['{0}'.format(k+1)])):

                        if self.ttvsectors['{0} {1}'.format(k+1, z+1)] != i:
                            continue

                        ttime = np.median(y['TT {0} {1}'.format(k+1, z+1)])

                        ind = np.where((self.tt[i] >= ttime - p/4) & (self.tt[i] <= ttime + p/4))

                        parstemp = np.median(pars[l], axis = 0)
                        parstemp[1] = ttime

                        fm0[ind] += lightCurve(parstemp, self.tt[i][ind], np.median(ld, axis = 0) if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])


                else:

                    if self.is_eclipse[k]:

                        fm0 = lightCurve(np.median(pars[l], axis = 0), self.tt[i], np.median(ld, axis = 0) if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i], eclipse = True, fp = np.median(y['fp {0}'.format(k+1)]), rstar = np.median(rstar) if self.fit_star else self.rs)

                    else:

                        fm0 = lightCurve(np.median(pars[l], axis = 0), self.tt[i], np.median(ld, axis = 0) if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])
                    
                fmsum += fm0
                fm.append(fm0)

            if detrend:

                gp = set_gp_params(np.median(rhogp), np.median(sigmagp), self.tt[i], self.ferr[i], gp)
                gpf0 = gp.predict(self.f[i] - fmsum - np.median(mean))
                gpf = gpf0


            z = {'fm': np.array(fm)}

            if detrend:

                z['gpf'] = np.array(gpf)


            fm_err = [[] for j in range(self.nt)]

            for j in tqdm(range(0,n,10)):

                fmsum = 0

                for k in range(self.n):

                    if not self.is_transit[k]:
                        continue

                    l = np.sum(self.is_transit[:k])

                    fm0 = np.zeros(len(self.tt[i]))

                    if self.fit_ttv[k]:

                        p = pars[l][j][0]

                        for ii in range(len(self.ttvi['{0}'.format(k+1)])):

                            if self.ttvsectors['{0} {1}'.format(k+1, ii+1)] != i:
                                continue

                            ttime = y['TT {0} {1}'.format(k+1, ii+1)][j]

                            pars[l][j][1] = ttime

                            ind = np.where((self.tt[i] >= ttime - p/4) & (self.tt[i] <= ttime + p/4))

                            fm0[ind] += lightCurve(pars[l][j], self.tt[i][ind], ld[j] if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])


                    else:

                        if self.is_eclipse[k]:

                            fm0 = lightCurve(pars[l][j], self.tt[i], ld[j] if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i], eclipse = True, fp = y['fp {0}'.format(k+1)][j], rstar = rstar[j] if self.fit_star else self.rs)

                        else:

                            fm0 = lightCurve(pars[l][j], self.tt[i], ld[j] if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])
                        
                    fmsum += fm0
                    fm_err[l].append(fm0)

                if detrend:

                    gp = set_gp_params(rhogp[j], sigmagp[j], self.tt[i], self.ferr[i], gp)
                    gpf0 = gp.predict(self.f[i] - fmsum - mean[j])
                    gpf_err.append(gpf0)


            fm_err = np.percentile(fm_err, [16,84], axis = 1)

            z['fm_err'] = fm_err

            if detrend:

                gpf_err = np.percentile(gpf_err, [16,84], axis = 0)
                z['gpf_err'] = gpf_err

            self._lcm[sec] = z

        pickle.dump(self._lcm, open(self.direc+'Output/'+name+'_lcm.p', 'wb'))

    @property
    def lc_mod(self):
        """Stores the best fit model light curves.

        fm is a dict, with a key for the phase folded models as well as for each light curve data set using their nicknames.

        The 'phase' key contains a dict with 'ttphase', a 1d numpy array of phase-folded time points used for plotting the models. Also within 'phase'
        is a key for each filter/bandpass used (e.g. TESS or Kepler). These are their own dicts with the following:

        - fmphase: A 2d numpy array, where each row corresponds to a different transiting planet. The rows contain the best fit phase-folded
          model light curve for each planet, at the times of ttphase.
        - fmphase_err: A 3d numpy array of the 1 sigma uncertainties on fmphase. In the 0th axis, the 0 index is a 2d array corresponding to the
          lower 1 sigma error and the 1 index is a 2d array corresponding to the upper 1 sigma error. Each of these is formatted like fmphase,
          with each row corresponding to a different planet.
        - eclphase: The same as fmphase, but for the phase-folded secondary eclipses of any planets that were fit for this. Will be an empty array if
          no planets were fit for secondary eclipses.
        - eclphase_err: The same as fmphase_err, but for the phase-folded secondary eclipses of any planets that were fit for this. Will be an empty
          array if no planets were fit for secondary eclipses.
        
        Each of the data set keys is a dict with the following:

        - fm: A 2d numpy array, where each row corresponds to a different transiting planet. The rows contain the best fit model light curve for
          each planet at the times of this data set.
        - fm_err: A 3d numpy array of the 1 sigma uncertainties on fm. In the 0th axis, the 0 index is a 2d array corresponding to the lower 1 sigma
          error and the 1 index is a 2d array corresponding to the upper 1 sigma error. Each of these is formatted like fm, with each row
          corresponding to a different planet.
        - gpf: A 1d numpy array of the GP model for detrending this light curve data set, using best fit hyper parameters.
        - gpf_err: A 2d numpy array of the 1 sigma uncertainties on gpf. In the 0th axis, the 0 index is a 1d array corresponding to the lower
          1 sigma error and the 1 index is a 1d array corresponding to the upper 1 sigma error.        
        """
        return self._lcm


    def plot_full_lc(self, name: str, show_plot = True):
        """Plots full light curve fits for each data set. Saves to the folder in Plots set by name. Each data set plot is named
        transit_full_nickname.png, where nickname is the data set's nickname. Shows the plot if show_plot is True.

        If the data set was detrended, plots the GP prediction with best fit hyperparameters in the first panel. The flattened light curve and
        best fit transit models are plotted in the next panel. The residuals are plotted in the final panel.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """
        
        for i in range(len(self.lcnames)):

            sec = self.lcnames[i]

            alpha = 0.1
            if self.exptimes[i] >= 1000/60/60/24:
                alpha = 0.3

            if self.detrend[i]:
                mosaic = [['a'],['a'],['b'],['b'],['c']]
            else:
                mosaic = [['b'],['b'],['c']]

            fig, ax = plt.subplot_mosaic(mosaic, figsize = (14,10), layout = 'constrained')

            fm = self._lcm[sec]['fm']
            fm_err = self._lcm[sec]['fm_err']
            mean = np.median(self.res['F0 {0}'.format(sec)] if 'F0 {0}'.format(sec) in self.res else self.fixed['F0 {0}'.format(sec)])

            if self.detrend[i]:

                gpf = self._lcm[sec]['gpf']
                gpf_err = self._lcm[sec]['gpf_err']

                ax['a'].errorbar(self.tt[i], self.f[i], yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
                ax['a'].plot(self.tt[i], gpf + mean, color = 'green', label = 'GP Model', zorder = 3, linewidth = 2)
                ax['a'].fill_between(self.tt[i], gpf_err[0]+mean, gpf_err[1]+mean, color = 'green', edgecolor = 'none', alpha = 0.5, zorder = 2)

                ax['a'].tick_params(axis = 'both', labelsize = 15)
                ax['a'].set_title(str(sec), fontsize = 20)

                ax['b'].text(0.01, 0.99, 'Flattened', fontsize = 20, ha = 'left', va = 'top', transform = ax['b'].transAxes)

            ax['b'].errorbar(self.tt[i], self.f[i] - (gpf if self.detrend[i] else 0), yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)

            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                k = np.sum(self.is_transit[:j])

                ax['b'].plot(self.tt[i], fm[k]+mean, label = 'Planet {0}'.format(j+1), zorder = 3, linewidth = 2)
                ax['b'].fill_between(self.tt[i], fm_err[0][k]+mean, fm_err[1][k]+mean, zorder = 2, alpha = 0.5, edgecolor = 'none')
            
            ax['b'].tick_params(axis = 'both', labelsize = 15)
            ax['b'].legend(fontsize = 15)

            if not self.detrend[i]:
                ax['b'].set_title(str(sec), fontsize = 20)

            mod = np.sum(fm, axis = 0) + mean + (gpf if self.detrend[i] else 0)
            ax['c'].errorbar(self.tt[i], self.f[i] - mod, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
            ax['c'].axhline(0, color = 'red', lw = 1, zorder = 2)

            ax['c'].text(0.01, 0.99, 'Residuals', fontsize = 20, ha = 'left', va = 'top', transform = ax['c'].transAxes)

            ax['c'].tick_params(axis = 'both', labelsize = 15)
            ax['b'].sharex(ax['c'])
            plt.setp(ax['b'].get_xticklabels(), visible = False)
            ax['c'].set_xlabel('Time [BJD-2450000]', fontsize = 20)

            if self.detrend[i]:
                ax['b'].sharey(ax['a'])
                ax['a'].sharex(ax['c'])
                plt.setp(ax['a'].get_xticklabels(), visible = False)

            fig.supylabel('Relative Flux', fontsize = 20)

            for a in ax:
                ax[a].set_rasterized(True)

            fig.savefig(self.direc+'Plots/'+name+'/transit_full_{0}.png'.format(sec))

            if show_plot:
                plt.show()

            else:
                plt.close()


    def plot_lc_phase(self, name: str, show_plot = True):
        """Plots phased light curve fits for each planet in each data set. Saves to the folder in Plots set by name. Each data set plot is named
        transits_nickname.png where nickname is the data set's nickname. Shows the plot if show_plot is True.

        The phased transit for each planet is plotted in a separate panel for each data set, with its residual below it.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        if np.any(self.fit_ttv):
            self.initialize_ttvs(name = self.init_ttvs_name)

        eclipse = False
        if np.any(self.is_eclipse):
            eclipse = True

        y = self.res | self.dres

        if self.use_priors:

            y = y | self.fixed

        for i in range(len(self.lcnames)):

            sec = self.lcnames[i]

            alpha = 0.1
            if self.exptimes[i] >= 1200/60/60/24:
                alpha = 0.3

            mos = []
            for j in range(self.nt):
                mos.append(['a{0}'.format(j),'a{0}'.format(j)] + (['c{0}'.format(j)] if eclipse else []))
                mos.append(['a{0}'.format(j),'a{0}'.format(j)] + (['c{0}'.format(j)] if eclipse else []))
                mos.append(['b{0}'.format(j),'b{0}'.format(j)] + (['d{0}'.format(j)] if eclipse else []))

            fig, ax = plt.subplot_mosaic(mos, figsize = (21 if eclipse else 14,6*self.nt), sharex = True, layout = 'constrained')

            mean = np.median(y['F0 {0}'.format(sec)])

            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                k = np.sum(self.is_transit[:j])

                p = np.median(y['P {0}'.format(j+1)])
                tc = np.median(y['Tc {0}'.format(j+1)])

                newt = self.tt[i].copy()

                if self.fit_ttv[j]:

                    for z in range(len(self.ttvi['{0}'.format(j+1)])):

                        if self.ttvsectors['{0} {1}'.format(j+1, z+1)] != i:
                            continue

                        ttime = np.median(y['TT {0} {1}'.format(j+1, z+1)])

                        ind = np.where((self.tt[i] >= ttime - 0.5) & (self.tt[i] <= ttime + 0.5))

                        newt[ind] -= ttime - (tc + p * self.ttvi['{0}'.format(j+1)][z])


                other = np.sum(self._lcm[sec]['fm'], axis = 0) - self._lcm[sec]['fm'][k] + (self._lcm[sec]['gpf'] if self.detrend[i] else 0)
                xfold = ((newt - tc + 0.5 * p) % p - 0.5 * p) * 24
                ttphase = self._lcm['phase']['ttphase'] * 24
                fmphase = self._lcm['phase'][self.filters[i]]['fmphase'][k] + mean
                fmphase_err = self._lcm['phase'][self.filters[i]]['fmphase_err'][:,k] + mean
                fm = self._lcm[sec]['fm'][k]

                ax['a{0}'.format(j)].errorbar(xfold, self.f[i] - other, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)

                exp = self.exptimes[i]*60*60*24
                bins = np.linspace(-12, 12, 25 if exp == 1800 else (40 if exp == 600 else 50))
                denom, _ = np.histogram(xfold, bins)
                num, _ = np.histogram(xfold, bins, weights = self.f[i] - other)
                denom[num == 0] = 1.0
                ax['a{0}'.format(j)].scatter(0.5 * (bins[1:] + bins[:-1]), num / denom, color='mediumseagreen', zorder = 2)

                ax['a{0}'.format(j)].plot(ttphase, fmphase, zorder = 4, linewidth = 2, color = 'dodgerblue')
                ax['a{0}'.format(j)].fill_between(ttphase, fmphase_err[0], fmphase_err[1], zorder = 3, alpha = 0.5, color = 'dodgerblue', edgecolor = 'none')

                ax['a{0}'.format(j)].text(0.01, 0.99, 'Planet {0}'.format(j+1), fontsize = 20, ha = 'left', va = 'top', transform = ax['a{0}'.format(j)].transAxes)

                l = np.where((xfold >= -12) & (xfold <= 12))[0]
                if len(l) > 0:
                    high = np.max(self.f[i][l] - other[l] + self.ferr[i][l])
                    low = np.min(self.f[i][l] - other[l] - self.ferr[i][l])

                else:
                    l = np.where((ttphase >= -12) & (ttphase <= 12))[0]
                    high = np.max(fmphase[l])
                    low = np.min(fmphase[l])

                ax['a{0}'.format(j)].set_ylim((low - 0.1*(high-low), high + 0.1*(high-low)))

                ax['a{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

                ax['b{0}'.format(j)].errorbar(xfold, self.f[i] - other - fm - mean, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
                ax['b{0}'.format(j)].axhline(0, c = 'red', lw = 1, zorder = 2)

                ax['b{0}'.format(j)].text(0.01, 0.99, 'Residuals', fontsize = 20, ha = 'left', va = 'top', transform = ax['b{0}'.format(j)].transAxes)

                ax['b{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

                if eclipse and not self.is_eclipse[j]:

                    plt.setp(ax['c{0}'.format(j)].get_xticklabels(), visible = False)
                    plt.setp(ax['c{0}'.format(j)].get_yticklabels(), visible = False)
                    ax['c{0}'.format(j)].axis('off')

                    plt.setp(ax['d{0}'.format(j)].get_xticklabels(), visible = False)
                    plt.setp(ax['d{0}'.format(j)].get_yticklabels(), visible = False)
                    ax['d{0}'.format(j)].axis('off')

                    continue

                if eclipse:

                    e = np.median(y['e {0}'.format(j+1)]) if 'e {0}'.format(j+1) in y else 0
                    w = np.median(y['w {0}'.format(j+1)]) if 'w {0}'.format(j+1) in y else np.pi/2
                    ar = np.median(y['a/rs {0}'.format(j+1)])
                    rs = np.median(y['rstar'])
                    tc2 = calc_t_sec(p, tc, e, w, ar, rs)

                    ke = np.sum(self.is_eclipse[:j])

                    other = np.sum(self._lcm[sec]['fm'], axis = 0) - self._lcm[sec]['fm'][k] + (self._lcm[sec]['gpf'] if self.detrend[i] else 0)
                    xfold = ((newt - tc2 + 0.5 * p) % p - 0.5 * p) * 24
                    eclphase = self._lcm['phase'][self.filters[i]]['eclphase'][ke] + mean
                    eclphase_err = self._lcm['phase'][self.filters[i]]['eclphase_err'][:,ke] + mean

                    ax['c{0}'.format(j)].errorbar(xfold, self.f[i] - other, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)

                    exp = self.exptimes[i]*60*60*24
                    bins = np.linspace(-12, 12, 25 if exp == 1800 else (40 if exp == 600 else 50))
                    denom, _ = np.histogram(xfold, bins)
                    num, _ = np.histogram(xfold, bins, weights = self.f[i] - other)
                    denom[num == 0] = 1.0
                    ax['c{0}'.format(j)].scatter(0.5 * (bins[1:] + bins[:-1]), num / denom, color='mediumseagreen', zorder = 2)

                    ax['c{0}'.format(j)].plot(ttphase, eclphase, zorder = 4, linewidth = 2, color = 'dodgerblue')
                    ax['c{0}'.format(j)].fill_between(ttphase, eclphase_err[0], eclphase_err[1], zorder = 3, alpha = 0.5, color = 'dodgerblue', edgecolor = 'none')

                    ax['c{0}'.format(j)].text(0.01, 0.99, 'Eclipse'.format(j+1), fontsize = 20, ha = 'left', va = 'top', transform = ax['c{0}'.format(j)].transAxes)

                    ax['c{0}'.format(j)].sharey(ax['a{0}'.format(j)])
                    plt.setp(ax['c{0}'.format(j)].get_yticklabels(), visible = False)

                    ax['c{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

                    ax['d{0}'.format(j)].errorbar(xfold, self.f[i] - other - fm - mean, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
                    ax['d{0}'.format(j)].axhline(0, c = 'red', lw = 1, zorder = 2)

                    ax['d{0}'.format(j)].text(0.01, 0.99, 'Residuals', fontsize = 20, ha = 'left', va = 'top', transform = ax['d{0}'.format(j)].transAxes)

                    ax['d{0}'.format(j)].sharey(ax['b{0}'.format(j)])
                    plt.setp(ax['d{0}'.format(j)].get_yticklabels(), visible = False)

                    ax['d{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

            
            ax['a0'].set_xlim(-12,12)
            ax['a0'].set_title(str(sec), fontsize = 20)
            ax['b{0}'.format(self.nt-1)].set_xlabel('Time since $T_{C}$ [hours]', fontsize = 20)
            fig.supylabel('Relative Flux', fontsize = 20)

            for a in ax:
                ax[a].set_rasterized(True)

            fig.savefig(self.direc+'/Plots/'+name+'/transits_{0}.png'.format(sec))

            if show_plot:
                plt.show()

            else:
                plt.close()


    def gen_rv(self, name: str):
        """Generates radial velocity models from the best fit parameters. Saves these to the pickle file Output/name_rvm.p. These RV models are
        accessible through ExoSystem.rv_mod. See the documentation on ExoSystem.rv_mod for details on the formatting of the object.

        Args:
            name (str): Name of the run. Sets the name of the output pickle file to name_rvm.p.
        """

        print('\nGenerating RV models for plots.')

        y = self.res | self.dres

        n = len(y['log_like'])

        if self.use_priors:

            f = self.fixed.copy()

            for ff in f:

                f[ff] = np.full((n), f[ff])

            y = y | f

            for i in range(self.n):

                if 'secw {0}'.format(i+1) not in y and 'e {0}'.format(i+1) in y:

                    y['secw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.cos(y['w {0}'.format(i+1)])
                    y['sesw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.sin(y['w {0}'.format(i+1)])

        pars = []
        ps = []
        tcs = []

        for i in range(self.n):

            if not self.is_rv[i]:
                continue

            pars.append(get_rv_params(y, i+1).T)
            ps.append(np.median(y['P {0}'.format(i+1)]))
            tcs.append(np.median(y['Tc {0}'.format(i+1)]))
        

        bkg_order = 0

        trend0 = y['gamma']

        if 'gamma_dot' in y:
            trend1 = y['gamma_dot']
            bkg_order = 1

        if 'gamma_ddot' in y:
            trend2 = y['gamma_ddot']
            bkg_order = 2

        rvm = []
        rvmplot = []

        num_min_p = (np.max(self.tr)+5 - np.min(self.tr)+5)/np.min(ps)
        num_pts = int(min(10000,max(1000, 8*num_min_p)))
        trplot = np.linspace(np.min(self.tr)-5, np.max(self.tr)+5, num_pts)

        trphase = np.linspace(-0.5, 0.5, 1000)
        rvallplot = 0
        rvmphase = []

        for j in range(self.n):

            if not self.is_rv[j]:
                continue

            k = np.sum(self.is_rv[:j])

            rvm.append(rvModel(np.median(pars[k], axis = 0), self.tr))

            rvmplot0 = rvModel(np.median(pars[k], axis = 0), trplot)
            rvmplot.append(rvmplot0)
            rvallplot += rvmplot0

            rvmphase.append(rvModel(np.median(pars[k], axis = 0), trphase * ps[k] + tcs[k]))

        bkg = np.array([np.median(trend0)]*len(self.tr)) * np.full(self.tr.shape, 1) + (np.median(trend1) * (self.tr - self.tr_ref) if bkg_order > 0 else 0) + (np.median(trend2) * (self.tr - self.tr_ref)**2 if bkg_order > 1 else 0)

        bkgplot = np.median(trend0) * np.full(trplot.shape, 1) + (np.median(trend1) * (trplot - self.tr_ref) if bkg_order > 0 else 0) + (np.median(trend2) * (trplot - self.tr_ref)**2 if bkg_order > 1 else 0)

        rvallplot += bkgplot

        rvallplot_err = []
        rvmphase_err = [[] for i in range(self.nr)]

        for i in tqdm(range(0,n)):

            rvallplot0 = 0

            for j in range(self.n):

                if not self.is_rv[j]:
                    continue

                k = np.sum(self.is_rv[:j])

                rvallplot0 += rvModel(pars[k][i], trplot)

                rvmphase_err[k].append(rvModel(pars[k][i], trphase * ps[k] + tcs[k]))

            bkgplot0 = trend0[i] * np.full(trplot.shape, 1) + (trend1[i] * (trplot - self.tr_ref) if bkg_order > 0 else 0) + (trend2[i] * (trplot - self.tr_ref)**2 if bkg_order > 1 else 0)

            rvallplot_err.append(rvallplot0 + bkgplot0)


        rvallplot_err = np.percentile(rvallplot_err, [16,84], axis = 0)
        rvmphase_err = np.percentile(rvmphase_err, [16,84], axis = 1)

        self._rvm = {'rvm': np.array(rvm), 'bkg': bkg, 'rvallplot': rvallplot, 'rvallplot_err': rvallplot_err, 'rvmplot': np.array(rvmplot), 'bkgplot': bkgplot, 'rvmphase': np.array(rvmphase), 'rvmphase_err': rvmphase_err, 'trplot': trplot, 'trphase': trphase}

        pickle.dump(self._rvm, open(self.direc+'Output/'+name+'_rvm.p', 'wb'))

    @property
    def rv_mod(self):
        """Stores the best fit radial velocity models.

        rvm is a dict,  with the following keys:

        - rvm: A 2d numpy array, where each row corresponds to a different RV planet. The rows contain the best fit RV model for each planet
          at the times of the RV data.
        - bkg: A 1d numpy array of the values of the background polynomial trend at the times of the RV data.
        - trplot: A 1d numpy array of more finely sampled times to make plotted models look smooth.
        - rvmplot: A 2d numpy array, where each row corresponds to a different RV planet. The rows contain the best fit RV model for each planet
          at the times of trplot, for smooth plotting.
        - bkgplot: A 1d numpy array of the values of the background polynomial trend at the times of trplot, for smooth plotting.
        - rvallplot: A 1d numpy array of the full best fit RV model at the times of trplot, for smooth plotting. It is the sum of all planet
          components and the background polynomial.
        - rvallplot_err: A 2d numpy array of the 1 sigma uncertainties on rvallplot. In the 0th axis, the 0 index is a 1d array corresponding to
          the lower 1 sigma error and the 1 index is a 1d array corresponding to the upper 1 sigma error.
        - trphase: A 1d numpy array of phase folded times.
        - rvmphase: A 2d numpy array, where each row corresponds to a different RV planet. The rows contain the best fit phase-folded RV
          model for each planet, at the times of trphase.
        - rvmphase_err: A 3d numpy array of the 1 sigma uncertainties on rvmphase. In the 0th axis, the 0 index is a 2d array corresponding to the
          lower 1 sigma error and the 1 index is a 2d array corresponding to the upper 1 sigma error. Each of these is formatted like rvmphase,
          with each row corresponding to a different planet.
        """
        return self._rvm


    def plot_rv(self, name: str, show_plot = True):
        """Plots the best fit radial velocity model to the data. Saves to the folder in Plots set by name. The plot is named rv.png. Shows the plot
        if show_plot is True.

        The top panel shows the full RV time series, with the combined best fit model. It also plots the contribution of each planet to the RV model,
        as well as the background polynomial. The residual is plotted below it.

        The lower panels are the RV models of each individual planet, phased to its period. The data have the contributions of the other planets and
        the background polynomial subtracted out. The residual is plotted below each of these panels as well.

        In each plot, the data are colored according to the data set they come from, and labeled with the data set's nickname.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        mos = [['a']*2, ['a']*2, ['b']*2]
        if self.nr == 1:
            mos += [['c0']*2, ['c0']*2, ['d0']*2]
        else:
            for i in range((self.nr+1)//2):
                mos += [['c{0}'.format(2*i),'c{0}'.format(2*i+1)], ['c{0}'.format(2*i),'c{0}'.format(2*i+1)], ['d{0}'.format(2*i),'d{0}'.format(2*i+1)]]

        fig, ax = plt.subplot_mosaic(mos, figsize = (28,6+6*(self.nr+1)//2), layout = 'constrained')

        trplot = self._rvm['trplot']
        rvallplot = self._rvm['rvallplot']
        rvallplot_err = self._rvm['rvallplot_err']
        rvmplot = self._rvm['rvmplot']
        bkgplot = self._rvm['bkgplot']
        rvm = self._rvm['rvm']
        bkg = self._rvm['bkg']
        rvmphase = self._rvm['rvmphase']
        rvmphase_err = self._rvm['rvmphase_err']
        trphase = self._rvm['trphase']

        y = self.res | self.dres

        if self.use_priors:

            y = y | self.fixed

        rv = self.rv.copy()
        for i, k in enumerate(np.unique(self.which_rv)):

            j = np.where(self.which_rv == k)[0]

            if i > 0:

                rv[j] -= np.median(y['rv_offset {0}'.format(self.rvnames[k])])

            ax['a'].errorbar(self.tr[j], rv[j], yerr = self.rverr[j], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][i], label = self.rvnames[k], zorder = 3, markersize = 10, elinewidth = 2)


        ax['a'].plot(trplot, rvallplot, color = 'mediumseagreen', linewidth = 2, zorder = 2)
        ax['a'].fill_between(trplot, rvallplot_err[0], rvallplot_err[1], color = 'mediumseagreen', zorder = 1, alpha = 0.5, edgecolor = 'none')

        for i in range(self.n):

            if not self.is_rv[i]:
                continue

            j = np.sum(self.is_rv[:i])

            ax['a'].plot(trplot, rvmplot[j], linewidth = 2, linestyle = '--', zorder = 1, label = 'Planet {0}'.format(i+1))

        ax['a'].plot(trplot, bkgplot, color = 'black', linewidth = 2, linestyle = ':', zorder = 0, label = 'Background')

        ax['a'].legend(fontsize = 15)
        ax['a'].tick_params(axis = 'both', labelsize = 15)
        ax['a'].set_ylabel('RV [m/s]', fontsize = 20)

        for i, k in enumerate(np.unique(self.which_rv)):

            j = np.where(self.which_rv == k)[0]

            ax['b'].errorbar(self.tr[j], rv[j] - np.sum(rvm, axis = 0)[j] - bkg[j], yerr = self.rverr[j], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][i], zorder = 3, markersize = 10, elinewidth = 2)

        ax['b'].axhline(0, c = 'red', lw = 1, zorder = 1)

        ax['b'].tick_params(axis = 'both', labelsize = 15)
        ax['b'].sharex(ax['a'])
        plt.setp(ax['a'].get_xticklabels(), visible = False)
        ax['b'].set_xlabel('Time [BJD-2450000]', fontsize = 20)
        ax['b'].set_ylabel('Resid. [m/s]', fontsize = 20)
        ax['b'].set_xlim([np.min(trplot),np.max(trplot)])

        if self.nr > 1 and self.nr%2 == 1:

            plt.setp(ax['c{0}'.format(self.nr)].get_xticklabels(), visible = False)
            plt.setp(ax['c{0}'.format(self.nr)].get_yticklabels(), visible = False)
            ax['c{0}'.format(self.nr)].axis('off')

            plt.setp(ax['d{0}'.format(self.nr)].get_xticklabels(), visible = False)
            plt.setp(ax['d{0}'.format(self.nr)].get_yticklabels(), visible = False)
            ax['d{0}'.format(self.nr)].axis('off')

        for i in range(self.n):

            if not self.is_rv[i]:
                continue

            j = np.sum(self.is_rv[:i])

            p = np.median(y['P {0}'.format(i+1)])
            
            tc = np.median(y['Tc {0}'.format(i+1)])

            other = np.sum(rvm, axis = 0) - rvm[j]

            xfold = ((self.tr - tc + 0.5 * p) % p - 0.5 * p)/p

            for ii, kk in enumerate(np.unique(self.which_rv)):

                jj = np.where(self.which_rv == kk)[0]

                ax['c{0}'.format(j)].errorbar(xfold[jj], rv[jj] - other[jj] - bkg[jj], yerr = self.rverr[jj], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][ii], zorder = 3, markersize = 10, elinewidth = 2)

            ax['c{0}'.format(j)].plot(trphase, rvmphase[j], linewidth = 2, zorder = 2)
            ax['c{0}'.format(j)].fill_between(trphase, rvmphase_err[0][j], rvmphase_err[1][j], alpha = 0.5, edgecolor = 'none', zorder = 1)

            ax['c{0}'.format(j)].text(0.01, 0.99, 'Planet {0}'.format(i+1), fontsize = 20, ha = 'left', va = 'top', transform = ax['c{0}'.format(j)].transAxes)

            ax['c{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

            if j%2 == 0:
                ax['c{0}'.format(j)].set_ylabel('RV [m/s]', fontsize = 20)


            for ii, kk in enumerate(np.unique(self.which_rv)):

                jj = np.where(self.which_rv == kk)[0]

                ax['d{0}'.format(j)].errorbar(xfold[jj], rv[jj] - np.sum(rvm, axis = 0)[jj] - bkg[jj], yerr = self.rverr[jj], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][ii], zorder = 3, markersize = 10, elinewidth = 2)


            ax['d{0}'.format(j)].axhline(0, c = 'red', lw = 1, zorder = 1)

            ax['c{0}'.format(j)].sharex(ax['d{0}'.format(j)])
            plt.setp(ax['c{0}'.format(j)].get_xticklabels(), visible = False)
            ax['d{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

            if j + 2 >= self.nr:
                ax['d{0}'.format(j)].set_xlabel('Orbital Phase', fontsize = 20)
                ax['d{0}'.format(j)].set_xlim([-0.5,0.5])

            else:
                ax['d{0}'.format(j)].sharex(ax['d{0}'.format(j+2)])
                plt.setp(ax['d{0}'.format(j)].get_xticklabels(), visible = False)

            if j%2 == 0:
                ax['d{0}'.format(j)].set_ylabel('Resid. [m/s]', fontsize = 20)

        
        fig.savefig(self.direc+'Plots/'+name+'/rv.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def gen_magfit(self, name: str):
        """Generates SED magnitudes and errors from the best fit parameters. Saves these to the pickle file Output/name_magfit.p. These model magnitudes are
        accessible through ExoSystem.magfit and ExoSystem.magfiterr.

        Args:
            name (str): Name of the run. Sets the name of the output pickle file to name_magfit.p.
        """

        y = self.res.copy()

        if self.use_priors:
            
            y = y | self.fixed

        mags = self.misti.interp_mag([y['eep'],y['log10(age)'],y['feh'],y['distance'],y['AV']], ['J','H','K','G','BP','RP','W1','W2','W3'])[3]
        self.magfit = np.median(mags, axis = 0)
        self.magfiterr = np.diff(np.percentile(mags, [16,50,84], axis = 0), axis = 0)

        pickle.dump({'magfit': self.magfit, 'magfiterr': self.magfiterr}, open(self.direc+'Output/'+name+'_magfit.p', 'wb'))


    def plot_sed_fit(self, name: str, show_plot = True):
        """Plots the magnitudes of the best fit stellar parameters against the data. Residuals are shown below. Saves to the folder in Plots set
        by name. The plot is named sed.png. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
           
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        fig, ax = plt.subplot_mosaic([['a'],['a'],['b']], figsize = (14, 12), layout = 'constrained', sharex = True)

        ax['a'].errorbar(self.magwls, self.magfit, yerr = self.magfiterr, fmt = '.r', label = 'Fit')
        ax['a'].errorbar(self.magwls, self.magobs, yerr = self.magobserr, fmt = '.k', label = 'Data')
        ax['a'].invert_yaxis()
        ax['a'].set_ylabel('Magnitude', fontsize = 20)
        ax['a'].tick_params(axis = 'both', labelsize = 15)
        ax['a'].legend(fontsize = 15)

        ax['b'].axhline(0, c = 'red')
        ax['b'].errorbar(self.magwls, self.magobs-self.magfit, yerr = np.sqrt(self.magobserr**2 + self.magfiterr**2), fmt = '.k')
        ax['b'].invert_yaxis()
        ax['b'].set_ylabel('Resid.', fontsize = 20)
        ax['b'].set_xlabel('Wavelength [$\\mu$m]', fontsize = 20)
        ax['b'].tick_params(axis = 'both', labelsize = 15)

        fig.savefig(self.direc+'Plots/'+name+'/sed.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_ttvs(self, name: str, show_plot = True):
        """Plots the best fit transit times versus the difference from a linear ephemeris for planets with TTVs. Saves to the folder in Plots set
        by name. The plot is named ttvs.png. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
           
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        self.initialize_ttvs(name = self.init_ttvs_name)

        y = self.res | self.dres

        if self.use_priors:

            y = y | self.fixed

        fig, ax = plt.subplots(self.nttv, figsize = (14, 6*self.nttv), sharex = True, layout = 'constrained')

        if self.nttv == 1:
            ax = np.array([ax])

        for i in range(self.n):

            if not self.fit_ttv[i]:
                continue

            j = np.sum(self.fit_ttv[:i])

            p = y['P {0}'.format(i+1)]
            tc = y['Tc {0}'.format(i+1)]

            ax[j].axhline(0, c = 'red')

            for k in range(len(self.ttvi['{0}'.format(i+1)])):

                tt = y['TT {0} {1}'.format(i+1,k+1)]
                ttmed = np.median(tt)

                diff = ((tt - (tc + p * self.ttvi['{0}'.format(i+1)][k]))*u.day).to(u.min).value
                diffmed = np.median(diff)
                differr = np.diff(np.percentile(diff, [16,50,84]))

                ax[j].errorbar(ttmed, diffmed, yerr = [[differr[0]],[differr[1]]], fmt = '.k')

            ax[j].text(0.01, 0.99, 'Planet {0}'.format(i+1), fontsize = 20, ha = 'left', va = 'top', transform = ax[j].transAxes)
            ax[j].tick_params(axis = 'both', labelsize = 15)

        ax[-1].set_xlabel('Time [BJD-2450000]', fontsize = 20)
        fig.supylabel('TTV (min)', fontsize = 20)

        fig.savefig(self.direc+'Plots/'+name+'/ttvs.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def examine_chains(self, parname: str):
        """Allows you to examine the chains of a specific parameter. Hovering over a chain with your mouse will highlight it and display the index
        of the chain in the 1 axis (0-indexed) of ExoSystem.samples.

        Useful for identifying the index of an outlier chain so that you can remove it using ExoSystem.remove_chains.

        Must have the full samples of the run available as Exosystem.samples. Either this is from having just run a fit with this ExoSystem object,
        or from loading in the results of a previous run which had save_samples set to True.

        Args:
            parname (str): The name of the parameter whose chains will be plotted. See the parameters that were fit in this run with
                ExoSystem.parnames.
        """

        k = self.parnames[parname]

        fig, ax = plt.subplots(figsize = (14,8), layout = 'constrained')

        for i in range(self.samples.shape[1]):
            ax.plot(self.samples[:,i,k], color = 'black', alpha = 0.2, label = 'Index = {0}'.format(i))

        cursor = mplcursors.cursor(hover = mplcursors.HoverMode.Transient, highlight = True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

        ax.set_xlabel('N Steps', fontsize = 15)
        ax.set_ylabel(parname, fontsize = 15)
        ax.set_title('Mouse over a chain to see its index.')

        plt.show()


    def remove_chains(self, name: str, idxs: np.typing.ArrayLike, save_samples: bool = False, show_plots: bool = True):
        """Removes chains of the specified indices from samples, log_likes, and the lists of derived periods and tcs in a ttv fit. Then, reruns
        ExoSystem.calc_gelman_rubin, ExoSystem.flatten_chains, ExoSystem.make_results, and ExoSystem.make_plots.

        One way to find outlier chains is by running ExoSystem.examine_chains first, then hovering over the plot to get indices of the outliers.

        Must have the full samples of the run available as Exosystem.samples. Either this is from having just run a fit with this ExoSystem object,
        or from loading in the results of a previous run which had save_samples set to True.

        Args:
            name (str): Name of the run. Does not have to be the same as the previous run. Sets the folder in Plots to save the plots to, as well as
                the names of the output pickle files.

            idxs (ArrayLike): Indices of the chains to be removed.

            save_samples (bool, optional): Whether or not to save the full, unflattened, un-thinned MCMC chains to a pickle file. This can be handy
                if you expect to want to remove problematic walkers that wandered off from the final results. Most of the time this isn't necessary,
                and these take up additional storage. Default is False.

            show_plots (bool, optional): Whether or not to show plots. Plots are saved regardless. Default is True.
        """

        self.samples = np.delete(self.samples, idxs, axis = 1)
        self.log_likes = np.delete(self.log_likes, idxs, axis = 1)

        if hasattr(self, 'blobs'):
            self.blobs = np.delete(self.blobs, idxs, axis = 1)

        if save_samples:

            self.save_samples(name)

        self.calc_gelman_rubin()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('')
        self.restab.pprint_all()
        

    def burn_steps(self, name: str, num_steps: int, save_samples: bool = False, show_plots: bool = True):
        """Useful if the burn-in wasn't long enough. Cuts the first num_steps steps from samples, log_likes, and the lists of derived periods and tcs
        in a ttv fit. Then, reruns ExoSystem.calc_gelman_rubin, ExoSystem.flatten_chains, ExoSystem.make_results, and ExoSystem.make_plots.

        Must have the full samples of the run available as Exosystem.samples. Either this is from having just run a fit with this ExoSystem object,
        or from loading in the results of a previous run which had save_samples set to True.

        Args:
            name (str): Name of the run. Does not have to be the same as the previous run. Sets the folder in Plots to save the plots to, as well as
                the names of the output pickle files.

            num_steps (int): Number of steps to cut from the beginning of the samples.

            save_samples (bool, optional): Whether or not to save the full, unflattened, un-thinned MCMC chains to a pickle file. This can be handy
                if you expect to want to remove problematic walkers that wandered off from the final results. Most of the time this isn't necessary,
                and these take up additional storage. Default is False.

            show_plots (bool, optional): Whether or not to show plots. Plots are saved regardless. Default is True.
        """

        self.samples = self.samples[num_steps:]
        self.log_likes = self.log_likes[num_steps:]

        if hasattr(self, 'blobs'):
            self.blobs = self.blobs[num_steps:]

        if save_samples:

            self.save_samples(name)

        self.calc_gelman_rubin()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('')
        self.restab.pprint_all()


    def calc_rv_bic(self):
        """Calculates the Bayesian information criterion (BIC) of the best fit RV model. Can be used to compare different RV models with different
        numbers of planets, background trends, planet eccentricities, etc.

        Not implemented for transit models, as the out of transit points tend to dominate and so it is less reliable.
        """

        k = len(self.res)-1
        n = len(self.tr.ravel())

        y = self.res.copy()

        if self.use_priors:

            y = y | self.fixed

        rvm = np.sum(self._rvm['rvm'], axis = 0) + self._rvm['bkg']

        for i in np.unique(self.which_rv)[1:]:

            j = np.where(self.which_rv == i)[0]

            rvm[j] += np.median(y['rv_offset {0}'.format(self.rvnames[i])])

        L = np.sum(lnNorm(self.rv, rvm, self.rverr))

        BIC = k * np.log(n) - 2 * L

        return BIC


    def calc_transit_err_scale(self):
        """Calculates the standard deviation of the residual of the best fit transit model and compares this to the median error on the light curve.
        Does this light curve by light curve. Prints out the ratio of this standard deviation to the median errors. Prints out a new error scaling
        value to put into init_lcs for each light curve, accounting for any error scaling already in effect.

        It can be useful to rescale errors to if you believe they are under or overestimated, and a good place to start is looking at residual scatter.
        This is an alternative to using a jitter term in other fitting packages. Rescaling the errors can be very important when detrending light
        curves using a GP, as underestimated errors will cause the GP to try to detrend to each individual point.
        """

        y = self.res.copy()

        if self.use_priors:

            y = y | self.fixed

        for i in range(len(self.tt)):

            sec = self.lcnames[i]

            fm = np.sum(self._lcm[sec]['fm'], axis = 0) + np.median(y['F0 {0}'.format(sec)])

            if 'gpf' in self._lcm[sec]:
                fm += self._lcm[sec]['gpf']
        
            resstd = np.std(self.f[i] - fm)

            mederr = np.median(self.ferr[i])

            print(sec, ': resid. std', resstd, ', median err', mederr, ', resid. std / median err', resstd / mederr, ', new err scale', resstd/mederr*self.lc_err_scale[i])


    def calc_rv_err_scale(self):
        """Calculates the standard deviation of the residual of the best fit RV model and compares this to the median error on the RV data.
        Does this data set by data set. Prints out the ratio of this standard deviation to the median errors. Prints out a new error scaling
        value to put into init_rv for each data set, accounting for any error scaling already in effect.

        It can be useful to rescale errors to if you believe they are under or overestimated, and a good place to start is looking at residual scatter.
        This is an alternative to using a jitter term in other fitting packages.
        """

        y = self.res.copy()

        if self.use_priors:

            y = y | self.fixed

        rvm = np.sum(self._rvm['rvm'], axis = 0) + self._rvm['bkg']

        rv = self.rv.copy()
        for i in np.unique(self.which_rv):

            j = np.where(self.which_rv == i)[0]

            if i > 0:

                rv[j] -= np.median(y['rv_offset {0}'.format(self.rvnames[i])])

            resstd = np.std(rv[j] - rvm[j])

            mederr = np.median(self.rverr[j])

            print(self.rvnames[i], ': resid. std', resstd, ', median err', mederr, ', resid. std / median err', resstd / mederr, ', new err scale', resstd/mederr*self.rv_err_scale[i])


    def calc_gelman_rubin(self):
        """Calculates the Gelman-Rubin Statistic for each parameter in the MCMC fit. Generally, if the Gelman-Rubin Statistic is below 1.1 for all
        parameters, then the MCMC has converged. If it looks converged, but the statistic is above 1.1 for some parameters, try running for more steps.
        """

        print('\nGelman-Rubin Statistics:')

        for k, v in self.parnames.items():

            s = self.samples[:,:,v]

            L, J = s.shape

            xj = np.mean(s, axis = 0)
            xs = np.mean(xj)
            B = L/(J-1)*np.sum((xj - xs)**2)
            W = np.mean([1/(L-1)*np.sum((s[:,j]-xj[j])**2) for j in range(J)])
            R = ((L-1)/L*W + B/L)/W
            
            print(k, R)


    def rv_lomb_scargle(self, min_freq: float = None, max_freq: float = None, freq: np.typing.ArrayLike = None, plot_periods = False, use_residual = True) -> tuple:
        """Creates a Lomb-Scargle periodogram of the RV data. Also plots the 1% and 5% false alarm levels in red and orange, respectively. Generally,
        a signal is significant if it crosses above the 1% false alarm level. Can be useful for looking for additional periodic signals (like planets!)
        in the RV data.

        Args:
            min_freq (float, optional): If it is not None, sets the minimum frequency (in 1/days) to search in the periodogram. Default is None.

            max_freq (float, optional): If it is not None, sets the maximum frequency (in 1/days) to search in the periodgram. Default is None.

            freq (ArrayLike, optional): Manually sets the frequency grid to use in the periodogram. Units are 1/days. Default is None.

            plot_periods (bool, optional): Whether or not to plot the periodgram relative to period (True) or frequency (False). Default is False.

            use_residual (bool, optional): Whether or not to make this periodogram off of just the RV data (False), or the residual of the RV data with the
                best fit model (True). If True, uses whatever model is currently in ExoSystem.rv_mod. Default is True.

        Returns:
            ArrayLike: The frequency grid, the periodogram power at each frequency, and the Lomb-Scargle object as a tuple. Allows you to calculate the maxmimum power
            frequency, or make other plots.
        """

        if use_residual:
            rvres = self.rv - np.sum(self._rvm['rvm'], axis = 0) - self._rvm['bkg']

        else:
            rvres = self.rv

        ls = LombScargle(self.tr, rvres, dy = self.rverr)

        if freq is not None:

            power = ls.power(freq)

        else:
             
            freq, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq)

        fig, ax = plt.subplots()

        if plot_periods:
            ax.plot(1/freq, power, c = 'black')
            ax.set_xscale('log')
            ax.set_xlabel('Period (days)', fontsize = 20)

        else:
            ax.plot(freq, power, c = 'black')
            ax.set_xlabel('Frequency (1/days)', fontsize = 20)

        ax.axhline(ls.false_alarm_level(0.01), c = 'red', label = '1% FA')
        ax.axhline(ls.false_alarm_level(0.05), c = 'orange', label = '5% FA')
        ax.legend(fontsize = 15)
        ax.tick_params(axis = 'both', labelsize = 15)
        plt.show()

        return freq, power, ls





def calc_m_from_k(p: np.typing.ArrayLike, k: np.typing.ArrayLike, e: np.typing.ArrayLike, inc: np.typing.ArrayLike, mstar: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Calculates the planet mass from the orbital period, RV semi-amplitude, eccentricity, inclination, and stellar mass. Does not make the
        assumption that the planet mass is much smaller than the stellar mass, so it is still valid even in brown dwarf-M dwarf scenarios.

        Args:
            p (ArrayLike): Orbital period in years.
            k (ArrayLike): RV semi-amplitude in m/s.
            e (ArrayLike): Orbital eccentricity.
            inc (ArrayLike): Orbital inclination in radians.
            mstar (ArrayLike): Stellar mass in solar masses.

        Returns:
            ArrayLike: The planet mass in Earth masses.
        """

        sini = np.sin(inc)
        
        b = (1*u.earthMass).to(u.Msun).value / mstar
        c = (k * np.sqrt(1-e**2) / 0.08946 / sini)**(-3/2) * p**(-1/2) / mstar

        m = b**2 / (3*c**2) + (2*b**6 + 18*b**3*c**2 + 3*np.sqrt(3)*np.sqrt(4*b**3*c**6 + 27*c**8) + 27*c**4)**(1/3) / (3*2**(1/3)*c**2) - 2**(1/3)*(-b**4 - 6*b*c**2) / (3*c**2*(2*b**6 + 18*b**3*c**2 + 3*np.sqrt(3)*np.sqrt(4*b**3*c**6 + 27*c**8) + 27*c**4)**(1/3))  

        return m


def lightCurve(par: np.typing.NDArray, t: np.typing.ArrayLike, ld: list, expt: float, ss: int, eclipse = False, fp = 0, rstar = 0) -> np.typing.ArrayLike:
        """Generates a model transit light curve using batman.

        Args:
            par (ndarray): Array of transit parameters. Parameters must be specifically in this order.
                - Orbital period in same units as t.
                - Time of conjunction in same units as t.
                - Planet to star radius ratio.
                - Orbital semi-major axis to stellar radius ratio.
                - Orbital inclination in degrees.
                - Orbital eccentricity. Should not be over 0.9 or batman has issues.
                - Argument of periastron in degrees.

            t (ArrayLike): Times at which to calculate the transit model.
            ld (list): A list only containing the linear and non-linear quadratic limb darkening coefficients, in that order.
            expt (float): The exposure time of each point. Used for supersampling.
            ss (int): Number of supersamples per exposure. For each point, this many points are generated across the whole exposure time, then averaged.

        Returns:
            ArrayLike: The model transit light curve at each of the input times, with a baseline of 0 flux.
        """
                        
        params = batman.TransitParams()
        
        params.per = par[0]
        params.t0 = par[1]
        params.rp = par[2]
        params.a = par[3]
        params.inc = par[4]
        params.ecc = par[5]
        params.w = par[6]
        
        params.limb_dark = 'quadratic'
        params.u = ld

        m = batman.TransitModel(params, t, supersample_factor = ss, exp_time = expt)
        
        flux = m.light_curve(params) - 1

        if eclipse:

            tc2 = calc_t_sec(par[0], par[1], par[5], par[6]*np.pi/180, par[3], rstar)

            params.fp = fp
            params.t_secondary = tc2

            m2 = batman.TransitModel(params, t, supersample_factor = ss, exp_time = expt, transittype = 'secondary')

            flux += m2.light_curve(params) - 1 - fp

        return flux


def calc_t_sec(p: np.typing.ArrayLike, tc: np.typing.ArrayLike, e: np.typing.ArrayLike, w: np.typing.ArrayLike, a: np.typing.ArrayLike, r: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """Calculates the time of secondary eclipse center, including light travel delay.

    Args:
        p (ArrayLike): Orbital period in days.
        tc (ArrayLike): Time of conjunction in days.
        e (ArrayLike): Orbital eccentricity.
        w (ArrayLike): Argument of periastron in radians.
        a (ArrayLike): Semi-major axis to stellar radius ratio.
        r (ArrayLike): Stellar radius in solar radii.

    Returns:
        ArrayLike: The time of secondary eclipse center.
    """

    f1 = np.pi/2 - w
    E1 = np.arctan2(np.sqrt(1-e**2)*np.sin(f1), e + np.cos(f1))
    M1 = E1 - e*np.sin(E1)

    f2 = 3*np.pi/2 - w
    E2 = np.arctan2(np.sqrt(1-e**2)*np.sin(f2), e + np.cos(f2))
    M2 = E2 - e*np.sin(E2)

    tc2 = p/(2*np.pi)*(M2-M1) + tc + (2*a*r*u.Rsun / constants.c).to(u.day).value

    return tc2


def rvModel(par: np.typing.NDArray, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """Generates model RVs using _rvModel. This is compiled with cython to run faster for large amounts of data.

        Args:
            par (Array): Array of RV parameters. Parameters must be specifically in this order.
                - Orbital period in same units as t.
                - Time of conjunction in same units as t.
                - RV semi-amplitude. Returns model RV points in same units as this.
                - Orbital eccentricity. Should not be over 0.9 or batman has issues.
                - Argument of periastron in radians.

            t (ArrayLike): Times at which to calculate the RV model.

        Returns:
            ArrayLike: The model RVs at each of the input times, centered at 0.
    """

    par = np.array(par, dtype = float)
    t = np.array(t, dtype = float)

    rv = _rvModel(par, t)

    return rv


def get_transit_params(par: dict, i: int, ar: np.typing.ArrayLike = None) -> np.typing.NDArray:
    """Generates the specific list of transit parameters needed for the lightCurve function from the parameter dict used in fitting.

    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        ar (ArrayLike, optional): Direct input for the semi-major axis to stellar radius ratio if it is not being fit for (during stellar fitting).
            Default is None, in which case the value is pulled from par.
    
    Returns:
        ndarray: Array of transit parameters that can be input directly into lightCurve.
    """

    p = np.exp(par['log(P) {0}'.format(i)])
    tc = par['Tc {0}'.format(i)]
    ror = par['ror {0}'.format(i)]
    if ar is None:
        ar = np.exp(par['log(a/rs) {0}'.format(i)])
    inc = np.arccos(par['cos(i) {0}'.format(i)]) * 180/np.pi

    shape = np.shape(par['log(P) {0}'.format(i)])

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) * 180/np.pi if 'secw {0}'.format(i) in par else np.full(shape, 90)

    return np.array([p, tc, ror, ar, inc, e, w])


def get_ttv_params(par: dict, i: int, ttvi: np.typing.NDArray[np.int64], ar: np.typing.ArrayLike = None) -> np.typing.NDArray:
    """Generates the specific list of transit parameters needed for the lightCurve function from the parameter dict used in fitting, in the case
    when the planet is being fit for TTVs. Calculates the period and time of conjunction using linear regression on the transit times.

    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        ttvi (ndarray): Array of integers representing the number of periods that have occurred between the first observed transit and each
            observed transit that is being fit.
        ar (ArrayLike, optional): Direct input for the semi-major axis to stellar radius ratio if it is not being fit for (during stellar fitting).
            Default is None, in which case the value is pulled from par.
        
    Returns:
        ndarray: Array of transit parameters that can be input directly into lightCurve.
    """

    ror = par['ror {0}'.format(i)]
    if ar is None:
        ar = np.exp(par['log(a/rs) {0}'.format(i)])
    inc = np.arccos(par['cos(i) {0}'.format(i)]) * 180/np.pi

    shape = np.shape(par['log(P) {0}'.format(i)])

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) * 180/np.pi if 'secw {0}'.format(i) in par else np.full(shape, 90)

    tts = [par['TT {0} {1}'.format(i, j+1)] for j in range(len(ttvi))]

    res = linregress(ttvi, tts)
    p = res.slope
    tc = res.intercept

    return np.array([p, tc, ror, ar, inc, e, w])


def get_rv_params(par: dict, i: int, p: np.typing.ArrayLike = None, tc: np.typing.ArrayLike = None) -> np.typing.NDArray:
    """Generates the specific list of RV parameters needed for the rvModel function from the parameter dict used in fitting.

    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        p (ArrayLike, optional): Direct input for the orbital period if it is not being fit for (if the planet is being fit for ttvs).
            Default is None, in which case the value is pulled from par.
        tc (ArrayLike, optional): Direct input for the time of conjunction if it is not being fit for (if the planet is being fit for ttvs).
            Default is None, in which case the value is pulled from par.
    
    Returns:
        ndarray: Array of RV parameters that can be input directly into rvModel.
    """

    if p is None:
        p = np.exp(par['log(P) {0}'.format(i)])
        tc = par['Tc {0}'.format(i)]

    shape = np.shape(par['log(K) {0}'.format(i)])

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) if 'secw {0}'.format(i) in par else np.full(shape, np.pi/2)

    k = np.exp(par['log(K) {0}'.format(i)])

    return np.array([p, tc, k, e, w])


def set_gp_params(rho: float, sigma: float, t: np.typing.ArrayLike, ferr: np.typing.ArrayLike, gp: GaussianProcess) -> GaussianProcess:
    """Updates a GP object with new parameters, then computes the factorization of the covariance matrix on the data.

    Args:
        rho (float): The period of the GP in days.
        sigma (float): The standard deviation of the GP.
        t (ArrayLike): The times of the data points.
        ferr (ArrayLike): The flux uncertainties of the data points.
        gp (GaussianProcess): The GP to update and then return.

    Returns:
        GaussianProcess: Returns the input GP, with updated parameters and a recomputed factorization.
    """

    gp.kernel = terms.SHOTerm(rho = rho, sigma = sigma, Q = 1/np.sqrt(2))
    gp.compute(t, yerr = ferr)

    return gp


def lnNorm(data: np.typing.ArrayLike, model: np.typing.ArrayLike, err: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """Computes the natural log of a normal distribution pdf. Used for the log likelihood.

    Args:
        data (ArrayLike): The observed values.
        model (ArrayLike): The predicted values.
        err (ArrayLike): The uncertainties in the observed values.

    Returns:
        ArrayLike: The log probability of each input point.
    """
    
    return - 0.5 * ( (model - data) / err)**2 - np.log( np.sqrt(2 * np.pi) * err)


def log_like(par_in: dict, exs: ExoSystem) -> tuple[float, np.typing.ArrayLike, np.typing.ArrayLike]:
    """The log likelihood function for fitting.

    Checks certain parameters to make sure they are in bounds. Calculates likelihoods from any priors. Calculates log likelihood and for stellar
    fitting using the isochrones MIST star model. Generates transit and RV models, then calculates log likelihood relative to data using lnNorm.

    Args:
        par_in (dict): Dictionary of fit parameters.
        exs (ExoSystem): The exosystem object currently being used to fit. Stores all of the options, data sets, etc.

    Returns:
        tuple: A tuple of the log likelihood, an array of periods of any planets that are fit for ttvs, and an array of conjunction times of any planets
        that are fit for ttvs. The former is the primary return for running MCMC with emcee. The latter two are saved as "blobs" in the emcee
        sampler object, and are used to track the period and time of conjunction at each step rather than recalculating the linear regression
        after the fact, since these are not directly fit parameters. If any parameter is out of bounds or any portion of the log likelihood is nan,
        returns negative infinity and empty lists.
    """

    if exs.use_priors:

        par = par_in | exs.fixed

        for i in range(exs.n):

            if  'w {0}'.format(i+1) in par_in and not -np.pi < par_in['w {0}'.format(i+1)] <= np.pi:

                return -np.inf, [], []

            if 'e {0}'.format(i+1) in par:

                par['secw {0}'.format(i+1)] = np.sqrt(par['e {0}'.format(i+1)]) * np.cos(par['w {0}'.format(i+1)])
                par['sesw {0}'.format(i+1)] = np.sqrt(par['e {0}'.format(i+1)]) * np.sin(par['w {0}'.format(i+1)])

    else:

        par = par_in.copy()

    if exs.fit_planets:

        if exs.order_a and exs.fit_transit:
            logalist = []

        for i in range(exs.n):
        
            if exs.is_transit[i] and exs.fit_transit:

                if not 0 <= par['cos(i) {0}'.format(i+1)] <= 1:
                    return -np.inf, [], []
                
                if exs.order_a:
                    logalist.append(par['log(a/rs) {0}'.format(i+1)])
                
                
            if exs.fit_ecc[i]:

                if par['secw {0}'.format(i+1)]**2 + par['sesw {0}'.format(i+1)]**2 > 0.9:
                    return -np.inf, [], []
                

        if exs.fit_transit and exs.order_a:
            logadiff = np.diff(np.array(logalist)[exs.transitsortorder])
            if np.any(logadiff <= 0):
                return -np.inf, [], []


        if exs.fit_transit and exs.fit_ld:
            if not 0 <= par['u1'] <= 1 or not 0 <= par['u2'] <= 1:
                return -np.inf, [], []
        


    like = 0

    if exs.fit_star:

        if not -0.5 <= par['feh'] <= 0.5:
            return -np.inf, [], []
        
        if par['AV'] < 0:
            return -np.inf, [], []

        starlike = exs.starmod.lnlike([par['eep'],par['log10(age)'],par['feh'],par['distance'],par['AV']])
        starlike += exs.starmod.lnprior([par['eep'],par['log10(age)'],par['feh'],par['distance'],par['AV']])

        if np.isnan(starlike):
            return -np.inf, [], []
        
        like += starlike

        if exs.fit_planets:

            rstar, mstar, Tstar, loggstar = exs.misti.interp_value([par['eep'],par['log10(age)'],par['feh']],['radius','mass','Teff','logg'])

            if not 2300 <= Tstar <= 7800 or not 3 <= loggstar <= 6:
                return -np.inf, [], []

            arlist = []
            for i in range(exs.n):

                if not exs.is_transit[i]:
                    arlist.append(np.nan)
                    continue

                if exs.fit_ttv[i]:

                    p = get_ttv_params(par, i+1, exs.ttvi['{0}'.format(i+1)], ar = 1)[0]

                else:

                    p = np.exp(par['log(P) {0}'.format(i+1)])

                if exs.is_rv[i] and exs.fit_rv:

                    e = 0
                    if exs.fit_ecc[i]:
                        e = par['secw {0}'.format(i+1)]**2 + par['sesw {0}'.format(i+1)]**2

                    mp = calc_m_from_k(p, np.exp(par['log(K) {0}'.format(i+1)]), e, np.arccos(par['cos(i) {0}'.format(i+1)]), mstar)

                else:

                    mp = 0

                ar = (((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                arlist.append(ar)
        

            
    tpars = []
    rpars = []
    ps = []
    tcs = []
    
    if exs.use_priors:
        priorpar = par.copy()

    if exs.fit_planets:

        for i in range(exs.n):
            
            if exs.is_transit[i] and exs.fit_transit:

                if exs.fit_ttv[i]:

                    pars = get_ttv_params(par, i+1, exs.ttvi['{0}'.format(i+1)], ar = arlist[i] if exs.fit_star else None)
                    ps.append(pars[0])
                    tcs.append(pars[1])
                    tpars.append(pars)

                    if exs.use_priors:
                        priorpar['log(P) {0}'.format(i+1)] = np.log(pars[0])
                        priorpar['Tc {0}'.format(i+1)] = pars[1]

                    if exs.is_rv[i] and exs.fit_rv:

                        rpars.append(get_rv_params(par, i+1, pars[0], pars[1]))

                else:

                    tpars.append(get_transit_params(par, i+1, ar = arlist[i] if exs.fit_star else None))

                    if exs.is_rv[i] and exs.fit_rv:

                        rpars.append(get_rv_params(par, i+1))

            elif exs.is_rv[i] and exs.fit_rv:

                rpars.append(get_rv_params(par, i+1))
            

    if exs.use_priors:

        priorlike = exs.allpriors.apply(priorpar)

        if np.isinf(priorlike):
            return -np.inf, [], []
        
        else:
            like += priorlike


    if exs.fit_transit:

        for i in range(len(exs.tt)):

            fm = par['F0 {0}'.format(exs.lcnames[i])]

            ld = exs.ld[exs.filters[i]]

            if exs.fit_ld:

                ld = [par['u1 {0}'.format(exs.filters[i])], par['u2 {0}'.format(exs.filters[i])]]

            if exs.fit_star:

                ld = [exs.ldgrids[exs.filters[i]][0]([Tstar, loggstar, par['feh']])[0], exs.ldgrids[exs.filters[i]][1]([Tstar, loggstar, par['feh']])[0]]

            for j in range(exs.n):

                if not exs.is_transit[j]:
                    continue

                k = np.sum(exs.is_transit[:j])

                if exs.fit_ttv[j]:

                    p = tpars[k][0]

                    for z in range(len(exs.ttvi['{0}'.format(j+1)])):

                        if exs.ttvsectors['{0} {1}'.format(j+1, z+1)] != i:
                            continue

                        ttime = par['TT {0} {1}'.format(j+1, z+1)]

                        tpars[k][1] = ttime

                        ind = np.where((exs.tt[i] >= ttime - p/4) & (exs.tt[i] <= ttime + p/4))

                        fm0 = np.zeros(len(exs.tt[i]))

                        fm0[ind] += lightCurve(tpars[k], exs.tt[i][ind], ld, exs.exptimes[i], exs.supersamples[i])

                        fm += fm0

                else:

                    if exs.is_eclipse[j]:

                        fm += lightCurve(tpars[k], exs.tt[i], ld, exs.exptimes[i], exs.supersamples[i], eclipse = True, fp = par['fp {0}'.format(j+1)], rstar = rstar if exs.fit_star else exs.rs)

                    else:

                        fm += lightCurve(tpars[k], exs.tt[i], ld, exs.exptimes[i], exs.supersamples[i])

            if exs.detrend[i]:

                resid = exs.f[i] - fm
                
                ii = np.sum(exs.detrend[:i])

                try:

                    gp = set_gp_params(np.exp(par['log(rho_gp) {0}'.format(exs.lcnames[i])]), np.exp(par['log(sigma_gp) {0}'.format(exs.lcnames[i])]), exs.tt[i], exs.ferr[i], exs.gps[ii])

                except:

                    return -np.inf, [], []

                like += gp.log_likelihood(resid)

            else:

                like += np.sum(lnNorm(exs.f[i], fm, exs.ferr[i]))

    if exs.fit_rv:

        rvm = np.array([par['gamma']]*len(exs.tr)) + (par['gamma_dot'] * (exs.tr - exs.tr_ref) if exs.rv_bkg_order > 0 else 0) + (par['gamma_ddot'] * (exs.tr - exs.tr_ref)**2 if exs.rv_bkg_order > 1 else 0)

        for i in np.unique(exs.which_rv)[1:]:

            j = np.where(exs.which_rv == i)[0]

            rvm[j] += par['rv_offset {0}'.format(exs.rvnames[i])]

        for i in range(exs.n):

            if not exs.is_rv[i]:
                continue

            j = np.sum(exs.is_rv[:i])

            rvm += rvModel(rpars[j], exs.tr)


        like += np.sum(lnNorm(exs.rv, rvm, exs.rverr))



    return like if not np.isnan(like) else -np.inf, np.array(ps), np.array(tcs)


def log_like_staronly(par_in: dict, exs: ExoSystem) -> float:
    """The log likelihood function for fitting stellar parameters only. Used for the initial parameter estimation before a main fit.

    Args:
        par_in (dict): Dictionary of fit parameters.
        exs (ExoSystem): The exosystem object currently being used to fit. Stores all of the options, data sets, etc.

    Returns:
        float: The log likelihood. Returns negative infinity if any parameter is out of bounds or any of the log likelihood functions return nan.
    """

    if exs.use_priors:

        par = par_in | exs.fixed

    else:

        par = par_in.copy()

    if not -0.5 <= par['feh'] <= 0.5:
        return -np.inf
    
    if par['AV'] < 0:
        return -np.inf

    starlike = exs.starmod.lnlike([par['eep'],par['log10(age)'],par['feh'],par['distance'],par['AV']])
    starlike += exs.starmod.lnprior([par['eep'],par['log10(age)'],par['feh'],par['distance'],par['AV']])

    if exs.use_priors:

        starlike += exs.allpriors.apply(par)

    if np.isnan(starlike):
        return -np.inf
    
    return starlike