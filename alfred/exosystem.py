import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as pe
import mplcursors
import emcee
import zeus
import corner
import batman
from scipy.stats import linregress, truncnorm
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize
from astropy.table import Table, vstack
from astropy.io import fits
from astropy import units as u
from astropy import constants
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
from isochrones import SingleStarModel, get_ichrone
import pickle
from tqdm.auto import tqdm
import os
import copy
import shutil
from multiprocessing import Pool
import time
from typing import Literal
from contextlib import nullcontext

from alfred._orbit_funcs import _rvModel, _astrometryModel
from alfred.init_class import Init_lcs, Init_rv, Init_ld, Init_star, Init_planets, Init_priors, Init_ttvs, Init_gaia
from alfred.ld_grids import *
from alfred.priors import *
from alfred import is_notebook

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
np.set_printoptions(legacy='1.25')

miniexs = None


class ExoSystem:
    """A class for fitting exoplanet systems.
    """

    def __init__(self, direc: str, init_planets = 'init_planets.txt', init_star = 'init_star.txt', init_lcs = 'init_lcs.txt',
                 init_rv = 'init_rv.txt', init_gaia = 'init_gaia.txt', init_ld = 'init_ld.txt', init_priors = 'init_priors.txt',
                 init_ttvs = 'init_ttvs.txt'):
        """Initialize an ExoSystem object with a directory and init files. The init files must all be created using alfred.

        Args:
            direc (str): The directory in which the init files, data, outputs, and plots of the ExoSystem are stored.
            init_planets (str, optional): Name of the planet parameter init file. Default is init_planets.txt.
            init_star (str, optional): Name of the star parameter init file. Default is init_star.txt.
            init_lcs (str, optional): Name of the light curve data init file. Default is init_lcs.txt.
            init_rv (str, optional): Name of the RV data init file. Default is init_rv.txt.
            init_gaia (str, optional): Name of the Gaia data init file. Default is init_gaia.txt.
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
        self.init_gaia = Init_gaia(self.direc, init_gaia).from_file()
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
        tab_gaia = self.init_gaia.table
        tab_ld = self.init_ld.table


        #which to fit

        self.is_transit = np.array(tab_planets['Transiting'])
        self.nt = int(np.sum(self.is_transit))
        self.n = len(self.is_transit)

        self.is_eclipse = np.array(tab_planets['Fit Eclipse']) & self.is_transit

        self.is_rv = np.array(tab_planets['RV Signal'])
        self.nr = int(np.sum(self.is_rv))

        self.is_astro = np.array(tab_planets['Fit Gaia'])

        self.fit_ttv = np.array(tab_planets['Fit TTVs'])
        self.nttv = int(np.sum(self.fit_ttv))

        self.fit_ecc = np.array(tab_planets['Fit Ecc'])
        self.ne = int(np.sum(self.fit_ecc))


        #planet params

        self.p = np.array(tab_planets['Period'])

        self.tc = np.array(tab_planets['Tc'])

        self.ror = np.array(tab_planets['Rp/Rs'])

        self.ar = np.array(tab_planets['a/Rs'])

        self.cosi = np.array(tab_planets['cos(i)'])

        self.k = np.array(tab_planets['K'])

        self.secosw = np.array(tab_planets['sqrt(e)cos(w)'])

        self.sesinw = np.array(tab_planets['sqrt(e)sin(w)'])

        self.fp = np.array(tab_planets['fp'])

        self.q = np.array(tab_planets['Mp/Ms'])

        self.omega = np.array(tab_planets['omega'])



        #stellar params

        self.rs = tab_star['Value'][0]
        self.rserr = tab_star['Error'][0]
        
        self.ms = tab_star['Value'][1]
        self.mserr = tab_star['Error'][1]

        self.Ts = tab_star['Value'][2]
        self.Tserr = tab_star['Error'][2]

        self.logg = tab_star['Value'][3]
        self.loggerr = tab_star['Error'][3]

        self.feh = tab_star['Value'][4]
        self.feherr = tab_star['Error'][4]

        self.plx = tab_star['Value'][5]
        self.plxerr = tab_star['Error'][5]

        #photometry

        #microns
        bandwls = {'J':1.235,'H':1.662,'K':2.159,'G':0.582239,'BP':0.503575,'RP':0.761996,'W1':3.3526,'W2':4.6028,'W3':11.5608,
                   'u':0.360804,'g':0.467178,'r':0.614112,'i':0.745789,'z':0.892278,'U':0.365988,'B':0.438074,'V':0.544543,'R':0.641147,'I':0.798209,
                   'TESS':0.745264}

        bandwidths = {'J':0.152026,'H':0.241018,'K':0.250619,'G':0.405297,'BP':0.215750,'RP':0.292444,'W1':0.662642,'W2':1.042266,'W3':5.505523,
                      'u':0.054097,'g':0.106468,'r':0.105551,'i':0.110257,'z':0.116401,'U':0.064040,'B':0.095920,'V':0.089310,'R':0.159100,'I':0.149510,
                      'TESS':0.389865}

        self.magwls = []
        self.magwidths = []
        self.magbands = []
        self.magobs = []
        self.magobserr = []

        for i in range(6, len(tab_star)):

            band = tab_star['Parameter'][i]
            self.magwls.append(bandwls[band])
            self.magwidths.append(bandwidths[band])
            self.magbands.append(band)
            self.magobs.append(tab_star['Value'][i])
            self.magobserr.append(tab_star['Error'][i])

        self.magwls = np.array(self.magwls)
        self.magwidths = np.array(self.magwidths)
        self.magbands = np.array(self.magbands)
        self.magobs = np.array(self.magobs)
        self.magobserr = np.array(self.magobserr)


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
            
            if not os.path.exists(self.direc+'/'+name) or name == '':
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

            if not os.path.exists(self.direc+'/'+name) or name == '':
                print('RV file {0} not found.'.format(name))
                continue
                
            try:
                rvdata = Table.read(self.direc + name)
            except:
                rvdata = Table.read(self.direc + name, format = 'ascii')
            
            rvdata.sort(tab_rv['Time Col'][i])

            msscale = 1e3 if tab_rv['m/s or km/s'][i] == 'km/s' else 1
            
            tr0 = np.array(rvdata[tab_rv['Time Col'][i]]) + tab_rv['Time Offset'][i] - 2450000
            rv0 =  np.array(rvdata[tab_rv['RV Col'][i]]) * msscale
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


        #Gaia file

        self.gaia_err_scale = np.array(tab_gaia['Err Scale'][0])

        fname = tab_gaia['File'][0]
        print(fname)

        if not os.path.isfile(self.direc+'/'+fname) and fname != '':
            print('Gaia file {0} not found.'.format(fname))

        elif fname != '':

            if tab_gaia['Format'][0] == 'DR4':

                gdata = Table.read(self.direc+fname)

                self.tgref = 2457936.875-2450000
                
                i = np.where(gdata['source_id'] == tab_gaia['Source ID'][0])[0]
                gdata = gdata[i]

                self.ra = gdata['ra0'][0]
                self.dec = gdata['dec0'][0]

                self.tg = []
                self.gal = []
                self.galerr = []
                self.psi = []
                self.plxf = []

                for i in range(len(gdata)):

                    j = ~gdata['centroid_pos_al'][i].mask & (gdata['ccd_proc_flags'][i] == 0)

                    self.tg.append(gdata['obs_time_tcb'][i][j] + gdata['obs_time_bary_corr'][i])
                    self.gal.append(gdata['centroid_pos_al'][i][j])
                    self.galerr.append(gdata['centroid_pos_error_al'][i][j])
                    self.psi.append(gdata['scan_pos_angle'][i][j]*np.pi/180)
                    self.plxf.append(gdata['parallax_factor_al'][i][j])

                self.tg = np.concatenate(self.tg)
                self.gal = np.concatenate(self.gal)
                self.galerr = np.concatenate(self.galerr) * self.gaia_err_scale
                self.psi = np.concatenate(self.psi)
                self.plxf = np.concatenate(self.plxf)

                i = np.argsort(self.tg)
                self.tg = self.tg[i]
                self.gal = self.gal[i]
                self.galerr = self.galerr[i]
                self.psi = self.psi[i]
                self.plxf = self.plxf[i]

                time = Time(2455197.5+(self.tg*u.ns).to(u.d).value, format = 'jd', scale = 'tcb')
                self.tg = time.tdb.value-2450000

            else:

                try:
                    gdata = Table.read(self.direc + fname)
                except:
                    rvdata = Table.read(self.direc + fname, format = 'ascii')

                gdata.sort(tab_gaia['Time Col'][0])

                self.tg = gdata[tab_gaia['Time Col'][0]] - 2450000
                self.gal = gdata[tab_gaia['AL Col'][0]]
                self.galerr = gdata[tab_gaia['Err Col'][0]]
                self.psi = gdata[tab_gaia['Psi Col'][0]]
                self.plxf = gdata[tab_gaia['Plx Fac Col'][0]]

                self.ra = tab_gaia['RA'][0]
                self.dec = tab_gaia['Dec'][0]
                self.tgref = tab_gaia['Ref Time'][0]

                

    def fit(self, name: str, fit_transit: bool, fit_rv: bool, fit_gaia: bool, fit_star: bool, nburn: int | float, nrun: int,
            parallel: bool | int | Literal['auto'] = False, sampler_type: Literal['emcee','zeus'] = 'emcee', min_nrun: int = 500, ntau: int = 50,
            dtau: float = 0.01, nwalk: int = 0, fit_ld = False, use_priors = False, rv_bkg_order: int = 0, star_run: str = None, save_samples = False,
            sigma_clip: float = 5, lc_supersample_size: int = 600, show_plots = True, skip_state_check = False, conv_check_inter: int = 100) -> None:
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
            
            fit_transit (bool): Whether or not to fit transits to the light curve data.
            
            fit_rv (bool): Whether or not to fit a model to the RV data.

            fit_gaia (bool): Whether or not to fit a model to Gaia epoch astrometry.
            
            fit_star (bool): Whether or not to fit stellar parameters. Can be fit on their own, or if transit data is also being fit
                (with or without RV data as well). Cannot be fit with just RV data.

            nburn (int or float): If an integer, the number of burn-in steps for the MCMC. If a float between 0 and 1, the fraction of the total
                steps to burn. These steps are thrown out before saving the results and making plots. The burn-in allows the chains to settle into
                the maximum likelihood.

            nrun (int): Maximum number of sampling steps to run the MCMC if convergence isn't reached. Does not include burn-in steps.

            parallel (bool or int or 'auto', optional): Whether or not to run MCMC sampling in parallel. If False, MCMC will run in serial. If True,
                MCMC will run in parallel with the total number of cores (or logical processors) your computer has. If an int, MCMC runs in parallel
                with that many cores, capped at your total number. If 'auto', runs a short test before sampling to find the optimal number of cores
                to use for best performance, including testing serial. Default is False.

            sampler_type ('emcee' or 'zeus', optional): Which MCMC sampler to use. emcee is an affine invariant MCMC ensemble sampler. zeus is an MCMC
                ensemble slice sampler. emcee steps are computed more quickly than zeus, but zeus will converge with less steps. The default is 'emcee'.

            min_nrun (int, optional): Minimum number of MCMC steps to run even if the chains converge sooner. Default is 500.

            ntau (int, optional): For convergence, the chain length must reach ntau times the maximum integrated autocorrelation time of the chains.
                Default is 50.

            dtau (float, optional): For convergence, the rate of change of the maximum integrated autocorrelation time of the chains must drop below dtau.
                Default is 0.01.

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

            skip_state_check (bool, optional): Passed only to the emcee sampler. Whether or not to skip checking whether the initial parameters can
                fully explore the space. Only set to True if you keep getting initial state check errors after burn in. Default is False.

            conv_check_inter (int, optional): Number of MCMC steps between convergence checks. Default is 100.
        """


        self.nburn = nburn
        self.nrun = nrun
        self.min_nrun = min_nrun
        self.ntau = ntau
        self.dtau = dtau
        self.nwalk = nwalk
        self.rv_bkg_order = rv_bkg_order
        self.sigma_clip = sigma_clip
        self.fit_ld = fit_ld
        self.fit_transit = fit_transit
        self.fit_rv = fit_rv
        self.fit_gaia = fit_gaia
        self.fit_star = fit_star
        self.parallel = parallel
        self.sampler_type = sampler_type
        self.lc_supersample_size = lc_supersample_size
        self.conv_check_inter = conv_check_inter


        if self.fit_transit and not np.any(self.is_transit):
            self.fit_transit = False
            print('No transiting planets found.')

        if self.fit_rv and not np.any(self.is_rv):
            self.fit_rv = False
            print('No radial velocity planets found.')

        if self.fit_gaia and not np.any(self.is_astro):
            self.fit_gaia = False
            print('No Gaia astrometric planets found.')

        self.fit_planets = self.fit_transit or self.fit_rv or self.fit_gaia


        if isinstance(self.nburn, float) and not 0 <= self.nburn < 1:

            print('nburn must be an integer or a float between 0 and 1.')
            return None

        if self.rv_bkg_order not in [0,1,2]:
            print('Invalid RV background polynomial order. Must be 0, 1, or 2.')
            return None

        if not fit_transit and not fit_rv and not fit_star and not fit_gaia:
            print('You need to fit something!')
            return None
        
        if not fit_transit and not fit_gaia and fit_rv and fit_star:
            print('To fit stellar parameters simultaneously with planet parameters, please use transit data and/or Gaia epoch astrometry.')
            return None
        
        if not os.path.isdir(self.direc+'Plots/'+name):
            os.mkdir(self.direc+'Plots/'+name)
        
        self.use_priors = use_priors
        if self.init_priors is None and use_priors:
            print('No init_priors file found. Check the name in the system initialization, or run Init_priors().create(). Running with no priors.')
            self.use_priors = False


        self.delete_run(name)
        
        
        if self.fit_transit:
            self.supersamples = np.array([max(1,int(x/self.lc_supersample_size)) for x in self.exptimes*24*60*60])


        if self.fit_star:

            self.fit_ld = False

            self.misti = get_ichrone('mist', bands = list(self.magbands))

            props = {'parallax': (self.plx, self.plxerr), 'Teff': (self.Ts, self.Tserr)}

            if not np.isnan(self.logg):
                props['logg'] = (self.logg, self.loggerr)
            
            if not np.isnan(self.feh):
                props['feh'] = (self.feh, self.feherr)

            for i in range(len(self.magbands)):
                props[self.magbands[i]] = (self.magobs[i], self.magobserr[i])

            self.starmod = SingleStarModel(self.misti, name = name, **props)

            self.ldgrids = {}

            for filt in np.unique(self.filters):                    

                interpu1, interpu2 = load_ld_grid(filt)

                self.ldgrids[filt] = [interpu1, interpu2]



        if np.any(self.fit_ttv) and self.fit_transit:
            self.initialize_ttvs(name = self.init_ttvs_name)


        self.x0 = {}

        if self.fit_planets:

            if not self.fit_star and self.fit_gaia:

                self.x0['plx'] = self.plx

                if self.fit_transit:

                    self.x0['log(rhostar)'] = np.log((self.ms*u.Msun/(4/3*np.pi*(self.rs*u.Rsun)**3)).to(u.g/u.cm**3).value)
                    self.x0['rstar'] = self.rs

                else:

                    self.x0['mstar'] = self.ms

            for i in range(self.n):

                if self.is_transit[i] and self.fit_transit:

                    if self.fit_ttv[i]:

                        for l in range(len(self.ttvs0[i+1])):
                            
                            self.x0['TT {0} {1}'.format(i+1, l+1)] = self.ttvs0[i+1][l]

                    else:

                        self.x0['log(P) {0}'.format(i+1)] = np.log(self.p[i])
                        self.x0['Tc {0}'.format(i+1)] = self.tc[i]

                    self.x0['ror {0}'.format(i+1)] = self.ror[i]

                    if not self.fit_star and not self.fit_gaia:

                        self.x0['log(a/rs) {0}'.format(i+1)] = np.log(self.ar[i])                    

                    self.x0['cos(i) {0}'.format(i+1)] = self.cosi[i]

                    if self.is_eclipse[i]:

                        self.x0['fp {0}'.format(i+1)] = self.fp[i]

                    if self.is_astro[i] and self.fit_gaia:

                        self.x0['comega {0}'.format(i+1)] = np.cos(self.omega[i])
                        self.x0['somega {0}'.format(i+1)] = np.sin(self.omega[i])

                elif self.is_astro[i] and self.fit_gaia:
                
                    self.x0['log(P) {0}'.format(i+1)] = np.log(self.p[i])
                    self.x0['Tc {0}'.format(i+1)] = self.tc[i]

                    self.x0['sico {0}'.format(i+1)] = np.sqrt((1-self.cosi[i])/2)*np.cos(self.omega[i])
                    self.x0['siso {0}'.format(i+1)] = np.sqrt((1-self.cosi[i])/2)*np.sin(self.omega[i])

                elif self.is_rv[i] and self.fit_rv:

                    self.x0['log(P) {0}'.format(i+1)] = np.log(self.p[i])
                    self.x0['Tc {0}'.format(i+1)] = self.tc[i]

                if self.is_astro[i] and self.fit_gaia:

                    self.x0['log(q) {0}'.format(i+1)] = np.log(self.q[i])

                elif self.is_rv[i] and self.fit_rv:

                    self.x0['log(K) {0}'.format(i+1)] = np.log(self.k[i])

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
                    self.x0['plx'] = np.median(self.res['plx'])
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
                self.x0['plx'] = x['plx']
                self.x0['AV'] = x['AV']

        elif self.fit_star:

            self.x0['eep'] = 355
            self.x0['log10(age)'] = 9.66
            self.x0['feh'] = 0
            self.x0['plx'] = self.plx
            self.x0['AV'] = 0.01


        if self.fit_transit:

            self.gps = []
            
            for i in range(len(self.tt)):

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


        self.lin_names = []

        if self.fit_transit:

            for i in range(self.n):
                if self.is_transit[i] and self.fit_ttv[i]:
                    self.lin_names.append('P {0}'.format(i+1))
                    self.lin_names.append('Tc {0}'.format(i+1))

            for i in range(len(self.lcnames)):
                self.lin_names.append('F0 {0}'.format(self.lcnames[i]))

        if self.fit_rv:

            self.lin_names.append('gamma')
            if self.rv_bkg_order > 0:
                self.lin_names.append('gamma_dot')
            if self.rv_bkb_order > 1:
                self.lin_names.append('gamma_ddot')

            for i in range(1,len(self.rvnames)):
                self.lin_names.append('rv_offset {0}'.format(self.rvnames[i]))

        if self.fit_gaia:

            self.lin_names.append('ra*_offset')
            self.lin_names.append('dec_offset')
            self.lin_names.append('pmra*')
            self.lin_names.append('pmdec')

        
        self.allpriors = AllPriors()

        if self.use_priors:

            self.allpriors.init_fixed(self.init_priors.table, self.x0)

            self.all_priors.init_fixed(self.init_priors.table, {ln: 0 for ln in self.lin_names})

            self.fixed = self.allpriors.fixed.copy()

            for i in range(self.n):

                if 'e {0}'.format(i+1) in self.fixed or 'w {0}'.format(i+1) in self.fixed:

                    self.x0.pop('secw {0}'.format(i+1))
                    self.x0.pop('sesw {0}'.format(i+1))

                    self.x0['e {0}'.format(i+1)] = self.secosw[i]**2 + self.sesinw[i]**2
                    self.x0['w {0}'.format(i+1)] = np.arctan2(self.sesinw[i], self.secosw[i])

                if ('cos(i) {0}'.format(i+1) in self.fixed or 'comega {0}' in self.fixed or 'somega {0}'.format(i+1) in self.fixed) and 'sico {0}'.format(i+1) in self.x0:

                    self.x0.pop('sico {0}'.format(i+1))
                    self.x0.pop('siso {0}'.format(i+1))

                    self.x0['cos(i) {0}'.format(i+1)] = self.cosi[i]
                    self.x0['comega {0}'.format(i+1)] = np.cos(self.omega)
                    self.x0['somega {0}'.format(i+1)] = np.sin(self.omega)

                if 'log(rhostar)' in self.fixed and 'log(a/rs) {0}'.format(i+1) in self.fixed:

                    self.x0.pop('log(a/rs) {0}'.format(i+1))

            for f in self.fixed:

                if f in self.x0:

                    self.x0.pop(f)

                if f in self.lin_names:

                    self.lin_names.remove(f)


        req_priors = Table(rows = [
            ['e x', 'U', 0, 0.9],
            ['w x', 'U', -np.pi, np.pi],
            ['u1 x', 'U', 0, 1],
            ['u2 x', 'U', 0, 1],
            ['feh', 'U', -0.5, 0.5],
            ['AV', 'U', 0, np.inf]
        ], names = ['Variable', 'Prior Type', 'Param 1', 'Param 2'])

        if self.fit_transit and not self.fit_gaia:
            req_priors.add_row(['cos(i) x', 'U', 0, 1])

        elif not self.fit_transit and self.fit_gaia:
            req_priors.add_row(['cos(i) x', 'U', -1, 1])

        elif self.fit_transit and self.fit_gaia:

            for i in range(self.n):

                if self.is_astro[i]:
                    req_priors.add_row(['cos(i) {0}'.format(i+1), 'U', -1, 1])

                    if self.is_transit[i]:
                        req_priors.add_row(['rsq_omega {0}'.format(i+1), 'UC', np.nan, np.nan])

                elif self.is_transit[i]:
                    req_priors.add_row(['cos(i) {0}'.format(i+1), 'U', 0, 1])

        

        if self.use_priors:

            prior_tab = vstack([self.init_priors.table, req_priors])

            self.allpriors.init_priors(prior_tab, self.x0, self.fit_ttv) 

            self.allpriors.init_priors(prior_tab, {ln: 0 for ln in self.lin_names}, self.fit_ttv)           

        else:

            self.allpriors = AllPriors()

            self.allpriors.init_priors(req_priors, self.x0, self.fit_ttv)

            self.fixed = {}


        if self.fit_gaia and not self.fit_star:

            if self.fit_transit:

                if 'rstar' not in self.allpriors.prior_dict:

                    print('It is highly recommended to place an informative prior on rstar with astrometry+transit fits.')

            else:

                if 'mstar' not in self.allpriors.prior_dict:

                    print('It is highly recommended to place an informative prior on mstar with astrometry fits.')


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

            self.setup_miniexs()
            res = minimize(lambda x, *args: -1 * log_like({k:v for k,v in zip(self.keys, x)}, *args)[0], [self.x0[k] for k in self.keys], method = 'Nelder-Mead')
            self.x = {k:v for k,v in zip(self.keys, res.x)}


        if self.fit_transit:
            pickle.dump(self.masks, open(self.direc+'Masks/'+name+'_masks.p', 'wb'))
        

        print('\nInitial parameters after optimization:')
        print(self.x)

        
        pos = self.initialize_chains()

        if type(self.parallel) == int:
            self.cores = self.parallel
            self.cores = min(self.cores, os.cpu_count())

        elif self.parallel == True:
            self.cores = os.cpu_count()

        elif self.parallel.lower() == 'auto':
            self.cores = self.find_opt_cores(pos)

        else:
            self.cores = 0


        self.run_sampler(pos, skip_state_check)

        
        if save_samples:

            self.save_samples(name)

        self.calc_autocorr()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('\nResults')
        self.restab.pprint_all()


    def initial_star_fit(self):
        """Runs an initial minimization on just the stellar parameters using a minimizer to get them in the ballpark before running MCMC. Starts at
        solar parameters. Called by fit.
        
        Do not run this by itself.
        """

        x0 = {'eep': 355, 'log10(age)': 9.66, 'feh': 0, 'plx': self.plx, 'AV': 0.01}
        keys = list(x0.keys())

        if self.use_priors:

            self.allpriors = AllPriors()

            self.allpriors.init_fixed(self.init_priors.table, x0)

            self.fixed = self.allpriors.fixed.copy()

            for f in self.fixed:

                if f in self.x0:

                    self.x0.pop(f)

            self.allpriors.init_priors(self.init_priors.table, x0, np.empty(0))

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

            self.setup_miniexs()
            res = minimize(lambda x, *args: -1 * log_like({k:v for k,v in zip(self.keys, x)}, *args)[0], [self.x0[k] for k in self.keys], method = 'Nelder-Mead')
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

                rstar, mstar, Tstar, loggstar, rhostar = self.misti.interp_value([y['eep'],y['log10(age)'],y['feh']],['radius','mass','Teff','logg','density'])

                y['rstar'] = rstar
                y['log(rhostar)'] = np.log(rhostar)

                for j in range(self.n):

                    if not self.is_transit[j]:
                        continue

                    if self.fit_ttv[j]:

                        p = get_ttv_params(y, j+1, self.ttvi['{0}'.format(j+1)])[0]

                    else:

                        p = np.exp(y['log(P) {0}'.format(j+1)])

                    if self.is_astro[j] and self.fit_gaia:

                        q = np.exp(y['log(q) {0}'.format(j+1)])
                        mp = q * mstar

                    elif self.is_rv[j] and self.fit_rv:

                        e = 0
                        if self.fit_ecc[j]:
                            e = y['secw {0}'.format(j+1)]**2 + y['sesw {0}'.format(j+1)]**2

                        mp = calc_m_from_k(p, np.exp(y['log(K) {0}'.format(j+1)]), e, np.arccos(y['cos(i) {0}'.format(j+1)]), mstar)

                    else:

                        mp = 0

                    ar = (((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                    y['log(a/rs) {0}'.format(i+1)] = np.log(ar)

            pars = []
            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                if self.fit_ttv[j]:

                    pars.append(get_ttv_params(y, j+1, self.ttvi['{0}'.format(j+1)]))

                else:

                    pars.append(get_transit_params(y, j+1))


            lastclipped = [0]*len(self.tt)

            for j in range(len(self.tt)):

                fm = 0

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

                            ms = y['mstar'] if 'mstar' in y else self.ms
                            rs = y['rstar'] if 'rstar' in y else self.rs
                            q = 0

                            if self.is_astro[k] and self.fit_gaia:

                                q = np.exp(y['log(q) {0}'.format(k+1)])

                            elif self.is_rv[k] and self.fit_rv:

                                e = 0
                                if self.fit_ecc[k]:
                                    e = y['secw {0}'.format(k+1)]**2 + y['sesw {0}'.format(k+1)]**2
                                
                                mp = calc_m_from_k(np.exp(y['log(P) {0}'.format(k+1)]), np.exp(y['log(K) {0}'.format(k+1)]), e, np.arccos(y['cos(i) {0}'.format(k+1)]), ms)

                                q = mp/ms

                            fm += lightCurve(pars[l], self.tt[j], ld, self.exptimes[j], self.supersamples[j], eclipse = True, fp = y['fp {0}'.format(k+1)], rstar = rs, q = q)

                        else:

                            fm += lightCurve(pars[l], self.tt[j], ld, self.exptimes[j], self.supersamples[j])

                resid = self.f[j] - fm

                if 'F0 {0}'.format(self.lcnames[j]) in self.fixed:

                    mean = self.fixed['F0 {0}'.format(self.lcnames[j])]
                    resid -= mean

                gpf = 0

                if self.detrend[j]:
                    
                    jj = np.sum(self.detrend[:j])

                    gp = set_gp_params(np.exp(y['log(rho_gp) {0}'.format(self.lcnames[j])]), np.exp(y['log(sigma_gp) {0}'.format(self.lcnames[j])]), self.tt[j], self.ferr[j], self.gps[jj])

                    if 'F0 {0}'.format(self.lcnames[j]) not in self.fixed:

                        M = np.ones((len(self.tt[j]), 1))

                        invKM = gp.apply_inverse(M)
                        invKr = gp.apply_inverse(resid)

                        MTinvKM = M.T @ invKM
                        MTinvKr = M.T @ invKr

                        X = np.linalg.solve(MTinvKM, MTinvKr)
                        mean = X[0]
                        resid -= M @ X                                                  

                    gpf = gp.predict(resid)

                    resid = self.f[j] - fm - gpf

                elif 'F0 {0}'.format(self.lcnames[j]) not in self.fixed:

                    M = np.ones((len(miniexs.tt[j]), 1))

                    W = 1/miniexs.ferr[j]**2
                    Mw = M * np.sqrt(W)[:, None]
                    Yw = resid * np.sqrt(W)

                    X, _, _, _ = np.linalg.lstsq(Mw, Yw, rcond = None)

                    mean = X[0]
                    resid -= mean

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

                    acc = []
                    cnt = 0
                    while cnt < self.nwalk:
                        x = np.log(np.random.normal(np.exp(self.x[k]), 0.0001, 2*self.nwalk))
                        lp = self.allpriors.apply({k: x})
                        valid = np.isfinite(lp)
                        acc.append(x[valid])
                        cnt += len(np.sum(valid))

                    pos.append(np.concatenate(acc)[:self.nwalk])

                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.0001, self.nwalk)))

            if 'Tc' in k or 'TT' in k:
                if self.use_priors:

                    acc = []
                    cnt = 0
                    while cnt < self.nwalk:
                        x = np.random.normal(self.x[k], 0.0001, 2*self.nwalk)
                        lp = self.allpriors.apply({k: x})
                        valid = np.isfinite(lp)
                        acc.append(x[valid])
                        cnt += len(np.sum(valid))

                    pos.append(np.concatenate(acc)[:self.nwalk])
                    
                else:
                    pos.append(np.random.normal(self.x[k], 0.0001, self.nwalk))

            if 'ror' in k or 'fp' in k or 'rstar' in k or 'mstar' in k:
                if self.use_priors:

                    acc = []
                    cnt = 0
                    while cnt < self.nwalk:
                        x = np.random.normal(self.x[k], 0.1*self.x[k], 2*self.nwalk)
                        lp = self.allpriors.apply({k: x})
                        valid = np.isfinite(lp)
                        acc.append(x[valid])
                        cnt += len(np.sum(valid))

                    pos.append(np.concatenate(acc)[:self.nwalk])
                    
                else:
                    pos.append(np.random.normal(self.x[k], 0.1*self.x[k], self.nwalk))

            if 'log(a/rs)' in k or 'log(rhostar)' in k:
                if self.use_priors:
                    
                    acc = []
                    cnt = 0
                    while cnt < self.nwalk:
                        x = np.log(np.random.normal(np.exp(self.x[k]), 0.1, 2*self.nwalk))
                        lp = self.allpriors.apply({k: x})
                        valid = np.isfinite(lp)
                        acc.append(x[valid])
                        cnt += len(np.sum(valid))

                    pos.append(np.concatenate(acc)[:self.nwalk])

                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.1, self.nwalk)))

            if 'cos(i)' in k or k.split()[0] == 'e':

                acc = []
                cnt = 0
                while cnt < self.nwalk:
                    x = np.random.normal(self.x[k], 0.001, 2*self.nwalk)
                    lp = self.allpriors.apply({k: x})
                    valid = np.isfinite(lp)
                    acc.append(x[valid])
                    cnt += len(np.sum(valid))

                pos.append(np.concatenate(acc)[:self.nwalk])

            if 'log(K)' in k or 'log(q)' in k:
                if self.use_priors:

                    acc = []
                    cnt = 0
                    while cnt < self.nwalk:
                        x = np.log(np.random.normal(np.exp(self.x[k]), 0.01, 2*self.nwalk))
                        lp = self.allpriors.apply({k: x})
                        valid = np.isfinite(lp)
                        acc.append(x[valid])
                        cnt += len(np.sum(valid))

                    pos.append(np.concatenate(acc)[:self.nwalk])

                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.01, self.nwalk)))

            if 'secw' in k or 'sesw' in k or 'comega' in k or 'somega' in k or 'sico' in k or 'siso' in k or  k in ['u1','u2']:

                acc = []
                cnt = 0
                while cnt < self.nwalk:
                    x = np.random.normal(self.x[k], 0.01, 2*self.nwalk)
                    lp = self.allpriors.apply({k: x})
                    valid = np.isfinite(lp)
                    acc.append(x[valid])
                    cnt += len(np.sum(valid))

                pos.append(np.concatenate(acc)[:self.nwalk])
                
            if k.split()[0] == 'w':

                acc = []
                cnt = 0
                while cnt < self.nwalk:
                    x = np.random.normal(self.x[k], 1, 2*self.nwalk)
                    lp = self.allpriors.apply({k: x})
                    valid = np.isfinite(lp)
                    acc.append(x[valid])
                    cnt += len(np.sum(valid))

                pos.append(np.concatenate(acc)[:self.nwalk])

            if '_gp' in k:
                if self.use_priors:

                    acc = []
                    cnt = 0
                    while cnt < self.nwalk:
                        x = np.log(np.random.normal(np.exp(self.x[k]), 0.01*np.exp(self.x[k]), 2*self.nwalk))
                        lp = self.allpriors.apply({k: x})
                        valid = np.isfinite(lp)
                        acc.append(x[valid])
                        cnt += len(np.sum(valid))

                    pos.append(np.concatenate(acc)[:self.nwalk])
                    
                else:
                    pos.append(np.log(np.random.normal(np.exp(self.x[k]), 0.01*np.exp(self.x[k]), self.nwalk)))

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

            if k == 'plx':
                pos.append(np.random.normal(self.x[k], 0.001, self.nwalk))

            if k == 'AV':
                lb, ub = 0, np.inf
                pos.append(truncnorm.rvs((lb-self.x[k])/0.01, (ub-self.x[k])/0.01, loc = self.x[k], scale = 0.01, size = self.nwalk))


        pos = np.transpose(np.array(pos))

        return pos


    def run_sampler(self, pos: np.typing.NDArray, skip_state_check: bool):
        """Not meant to be run on its own. Runs the MCMC sampler for burn-in and sample steps using the chosen sampler.

        Args:
            pos (ndarray): The initial positions of the chains generated by initialize_chains.

            skip_state_check (bool): Whether or not to skip the initial state check after the burn-in. Only used
                with emcee sampler.
        """

        blobs_dtype = [(name, float) for name in self.lin_names]

        sampler_kwargs = {'nwalkers': self.nwalk,
                          'ndim': len(self.x),
                          'blobs_dtype': blobs_dtype,
                          }

        if self.sampler_type == 'emcee':

            sampler_kwargs.update({'log_prob_fn': log_like,
                                   'parameter_names': self.parnames
                                   })

            run_sample_kwargs = {'skip_initial_state_check': skip_state_check}

            conv_func = check_emcee_conv
            sampler = emcee.EnsembleSampler

        elif self.sampler_type == 'zeus':

            sampler_kwargs.update({'logprob_fn': log_like_zeus,
                                   'args': (self.keys,),
                                   'verbose': False,
                                   })

            run_sample_kwargs = {'progress': False}

            conv_func = check_zeus_conv
            sampler = zeus.EnsembleSampler

        if self.cores == 0:
            self.setup_miniexs()
            ctx = nullcontext()
        else:
            ctx = Pool(processes = self.cores, initializer = self.setup_miniexs)

        with ctx as pool:
        
            if self.cores > 0:
                sampler_kwargs['pool'] = pool

            self.sampler = sampler(**sampler_kwargs)

            if isinstance(self.nburn, int):

                burnfrac = 0.

                print('\nRunning MCMC burn-in.')

                prog = 'notebook' if is_notebook and self.sampler_type == 'emcee' else True

                pos = self.sampler.run_mcmc(pos, self.nburn, progress = prog)
                self.sampler.reset()

            elif isinstance(self.nburn, float):

                burnfrac = self.nburn

            print('\nRunning MCMC sampling.')

            last_tau = None
            min_it = int(self.min_nrun/(1-burnfrac))

            with tqdm(total=self.nrun, desc="MCMC Sampling", unit="step") as pbar:

                for it, state in enumerate(self.sampler.sample(pos, iterations = self.nrun, **run_sample_kwargs), start = 1):

                    pbar.update(1)

                    if it < min_it or it%self.conv_check_inter != 0:
                        continue

                    burnin = int(burnfrac*it)

                    conv, last_tau = conv_func(self.sampler, it, last_tau, burnin, ntau = self.ntau, deltatau = self.dtau)

                    if last_tau is not None:
                        pbar.set_description(f"MCMC Sampling (𝜏: {last_tau:.1f})")

                    if conv:
                        pbar.total = it
                        
                        if hasattr(pbar, "container"):
                            pbar.container.children[1].max = it
                            pbar.container.children[1].value = it
                            pbar.container.children[1].bar_style = 'success'

                        pbar.refresh()
                        break


            self.samples = self.sampler.get_chain(discard = int(burnfrac*it))

            self.log_likes = self.sampler.get_log_prob(discard = int(burnfrac*it))

            self.lin_par = self.sampler.get_blobs(flat = True)


    def find_opt_cores(self, pos: np.typing.NDArray) -> int:
        """Finds the optimal number of cores to run in parallel for best performance (or whether to use serial instead) by running short, 15-step
        MCMC runs. This is called when parallel is set to 'auto' in ExoSystem.fit.

        Args:
            pos (ndarray): The initial positions of the chains generated by initialize_chains.

        Returns:
            int: The optimal number of cores to use, with 0 for serial.
        """

        max_cores = os.cpu_count()

        test_cores = list(range(0, max_cores+1, 2))
        if max_cores not in test_cores:
            test_cores.append(max_cores)

        best_time = np.inf
        opt_cores = 0

        sampler_kwargs = {'nwalkers': self.nwalk,
                         'ndim': len(self.x),
                         }

        if self.sampler_type == 'emcee':
            sampler_kwargs.update({'log_prob_fn': log_like,
                                   'parameter_names': self.parnames})

        elif self.sampler_type == 'zeus':
            sampler_kwargs.update({'logprob_fn': log_like_zeus,
                                   'args': (self.keys,),
                                   'verbose': False,
                                   })

        for c in test_cores:

            if c == 0:
                self.setup_miniexs()
                ctx = nullcontext()
            else:
                ctx = Pool(processes = c, initializer = self.setup_miniexs)

            try:
                with ctx as pool:

                    if c == 0:
                        sampler_kwargs['pool'] = None
                    else:
                        sampler_kwargs['pool'] = pool

                    if self.sampler_type == 'emcee':
                        benchsampler = emcee.EnsembleSampler(**sampler_kwargs)

                    elif self.sampler_type == 'zeus':
                        benchsampler = zeus.EnsembleSampler(**sampler_kwargs)

                    t0 = time.perf_counter()
                    benchsampler.run_mcmc(pos, 15, progress = False)
                    elapsed = time.perf_counter() - t0

                    if elapsed < best_time:
                        best_time = elapsed
                        opt_cores = c

            except Exception as e:
                print('{0} cores failed benchmarking: {1}'.format(c, e))
                continue

        if opt_cores == 0:
            print('\nRunning in serial.')
        else:
            print('\nRunning in parallel with {0} cores.'.format(opt_cores))

        return opt_cores
    

    def setup_miniexs(self):
        """Creates a MiniExoSystem object from the current ExoSystem, storing the data and options used in the log likelihood function for fitting.
        This MiniExoSystem object is made global, allowing the log likelihood function to access it, and avoiding pickling all of the data during
        parallel MCMC sampling.
        """

        global miniexs

        miniexs = MiniExoSystem(self)


    def save_samples(self, name):
        """Saves the full sample chains out to a pickle file (not flattened or thinned). This is only automatically run if save_samples was set to True
        during fitting. Otherwise, can be run manually after a fit. This file takes up much more storage space than the normal thinned and flattened
        samples which are always saved. The pickle file is saved to Output/name_samples.p.

        Args:
            name (str): Name of the run. Sets the name of the pickle file to name_samples.p.
        """

        z = {'parnames': self.parnames, 'samples': self.samples, 'log_like': self.log_likes}

        pickle.dump(z, open(self.direc+'Output/'+name+'_samples.p', 'wb'))


    def continue_run(self, name: str, nrun: int, check_converge = True, min_nrun = 500, save_samples = False, show_plots = True, skip_state_check = False):
        """Continues running the MCMC sampler from where it left off, without a burn in. Remakes all results and plots, and saves to the provided name
        (does not have to be the same name as the previous run).

        Can only be run if a run has already been performed with this ExoSystem object. Will not work on a loaded in run.

        Args:
            name (str): Name of the fit. This name will be attached to all ouputs, including pickle files of data, human readable results tables,
                and the folder in Plots in which this run's plots will be saved. This name is also used for loading results back in to be manipulated
                or plotted again.

            nrun (int): Maximum number of sampling steps to run the MCMC if convergence isn't reached. If check_converge is False, runs for this many
                steps regardless.

            check_converge (bool, optional): Whether or not to stop running MCMC once convergence criteria from original fit is reached. Default is True.

            min_nrun (int, optional): Minimum number of MCMC steps to run even if the chains converge sooner. Ignored if check_converge is False.
                Default is 500.

            save_samples (bool, optional): Whether or not to save the full, unflattened, un-thinned MCMC chains to a pickle file. This can be handy
                if you expect to want to remove problematic walkers that wandered off from the final results. Most of the time this isn't necessary,
                and these take up additional storage. Default is False.

            show_plots (bool, optional): Whether or not to show plots at the end of the run. Plots are saved regardless. Default is True.

            skip_state_check (bool, optional): Passed only to the emcee sampler. Whether or not to skip checking whether the initial parameters can
                fully explore the space. Only set to True if you keep getting initial state check errors after burn in. Default is False.
        """

        if self.cores == 0:
            self.setup_miniexs()
            ctx = nullcontext()
        else:
            ctx = Pool(processes = self.cores, initializer = self.setup_miniexs)

        nprev = len(self.samples)

        burnin = 0
        if isinstance(self.nburn, float):
            burnin = int(self.nburn*nprev/(1-self.nburn))

        run_sample_kwargs = {}
        
        if self.sampler_type == 'emcee':

            run_sample_kwargs['skip_initial_state_check'] = skip_state_check

            conv_func = check_emcee_conv

            last_tau = np.max(emcee.autocorr.integrated_time(self.samples, quiet=True))

        with ctx as pool:
        
            if self.cores > 0:
                self.sampler.pool = pool

            print('\nRunning MCMC sampling.')

            state = self.samples[-1]

            if not check_converge:

                prog = 'notebook' if is_notebook and self.sampler_type == 'emcee' else True

                self.sampler.run_mcmc(state, nrun, progress = prog, **run_sample_kwargs)

            else:

                min_it = min_nrun

                if self.sampler_type == 'zeus':

                    run_sample_kwargs['progress'] = False

                    conv_func = check_zeus_conv

                    last_tau = np.max(zeus.AutoCorrTime(self.samples))
    
                with tqdm(total=nrun, desc="MCMC Sampling", unit="step") as pbar:
    
                    for it, state in enumerate(self.sampler.sample(state, iterations = nrun, **run_sample_kwargs), start = 1):
    
                        pbar.update(1)
    
                        if it < min_it or it%self.conv_check_inter != 0:
                            continue
        
                        conv, last_tau = conv_func(self.sampler, it + nprev, last_tau, burnin, ntau = self.ntau, deltatau = self.dtau)
    
                        if last_tau is not None:
                            pbar.set_description(f"MCMC Sampling (𝜏: {last_tau:.1f})")
    
                        if conv:
                            pbar.total = it
                            
                            if hasattr(pbar, "container"):
                                pbar.container.children[1].max = it
                                pbar.container.children[1].value = it
                                pbar.container.children[1].bar_style = 'success'
    
                            pbar.refresh()
                            break

            self.samples = self.sampler.get_chain(discard = burnin)

            self.log_likes = self.sampler.get_log_prob(discard = burnin)

        if save_samples:

            self.save_samples(name)

        self.calc_autocorr()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('\nResults')
        self.restab.pprint_all()


    def flatten_chains(self):
        """Flattens and thins by a factor of 20 ExoSystem.samples and ExoSystem.log_likes. Stores these as ExoSystem.flat_samples and ExoSystem.flat_log_likes, respectively.
        """

        shape = self.samples.shape
        self.flat_samples = np.reshape(self.samples, (shape[0]*shape[1], shape[2]))

        shape = self.log_likes.shape
        self.flat_log_likes = np.reshape(self.log_likes, (shape[0]*shape[1]))


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

        self.res = {}

        for k, v in self.parnames.items():

            self.res[k] = self.flat_samples[:,v]

        for i in range(len(self.lin_names)):

            self.res[self.lin_names[i]] = self.lin_par[self.lin_names[i]]

        self.res['log_like'] = self.flat_log_likes

        pickle.dump(self.res, open(self.direc+'Output/'+name+'_res.p', 'wb'))

        self.dres = {}

        if self.use_priors:
        
            y = self.res | self.fixed

        else:
        
            y = self.res.copy()

        n = len(self.res['log_like'])

        if self.fit_star:

            self.starmod._derived_samples = self.misti(*[self.res[x] if x in self.res else [self.fixed[x]]*n for x in ['eep','log10(age)','feh','distance','AV']])
            self.starmod._derived_samples["parallax"] = self.res['plx']
            self.starmod._derived_samples["distance"] = 1000/self.res['plx']
            self.starmod._derived_samples["AV"] = self.res['AV']

            self.dres['rstar'] = self.starmod._derived_samples['radius']
            self.dres['mstar'] = self.starmod._derived_samples['mass']
            self.dres['rhostar'] = self.starmod._derived_samples['density']
            self.dres['mstar_init'] = self.starmod._derived_samples['initial_mass']
            self.dres['Tstar'] = self.starmod._derived_samples['Teff']
            self.dres['loggstar'] = self.starmod._derived_samples['logg']
            self.dres['Lstar'] = 10**self.starmod._derived_samples['logL']
            self.dres['Mbolstar'] = self.starmod._derived_samples['Mbol']
            self.dres['distance'] = self.starmod._derived_samples['distance']

            rstar = self.dres['rstar']
            mstar = self.dres['mstar']
            T = self.dres['Tstar']

            interp_input = np.array([self.dres['Tstar'], self.dres['loggstar'], self.res['feh']]).T

            for filt in np.unique(self.filters):

                self.dres['u1 {0}'.format(filt)] = self.ldgrids[filt][0](interp_input)
                self.dres['u2 {0}'.format(filt)] = self.ldgrids[filt][1](interp_input)

        else:

            T = np.random.normal(self.Ts, self.Tserr, n)
            self.dres['Tstar'] = T

            if 'plx' in y:
                self.dres['distance'] = 1000/y['plx']

            if 'rstar' in y:
                rstar = y['rstar']
            else:
                rstar = np.random.normal(self.rs, self.rserr, n)
                self.dres['rstar'] = rstar

            if 'mstar' in y:
                mstar = y['mstar']

            elif 'log(rhostar)' in y:
                rhostar = np.exp(y['log(rhostar)'])
                self.dres['rhostar'] = rhostar

                mstar = (rhostar*u.g/u.cm**3 * 4/3*np.pi * (rstar*u.Rsun)**3).to(u.Msun).value
                self.dres['mstar'] = mstar

            else:
                mstar = np.random.normal(self.ms, self.mserr, n)
                self.dres['mstar'] = mstar

        hasJ = False
        if 'J' in self.magbands:
            i = np.where(self.magbands == 'J')[0]
            J = np.random.normal(self.magobs[i], self.magobserr[i], n)
            hasJ = True

        einsol = 1*u.Lsun / (4 * np.pi * u.AU**2)

        if self.fit_planets:

            if self.use_priors:

                for i in range(self.n):

                    if 'e {0}'.format(i+1) in y:

                        y['secw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.cos(y['w {0}'.format(i+1)])
                        y['sesw {0}'.format(i+1)] = np.sqrt(y['e {0}'.format(i+1)]) * np.sin(y['w {0}'.format(i+1)])

            for i in range(self.n):

                if self.is_transit[i] and self.fit_transit:

                    if self.fit_ttv[i]:

                        p = y['P {0}'.format(i+1)]

                    else:

                        p = np.exp(y['log(P) {0}'.format(i+1)])
                        self.dres['P {0}'.format(i+1)] = p

                    rp = y['ror {0}'.format(i+1)] * rstar * (1*u.Rsun).to(u.earthRad).value
                    self.dres['Rp {0}'.format(i+1)] = rp

                    if not self.fit_star and not self.fit_gaia:

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

                fc = np.pi/2 - w
                Ec = np.arctan2(np.sqrt(1-e**2) * np.sin(fc), e + np.cos(fc))
                tc = y['Tc {0}'.format(i+1)]
                tp = tc - p/(2*np.pi) * (Ec - e * np.sin(Ec))
                self.dres['Tp {0}'.format(i+1)] = tp

                mp = 0

                if self.is_astro[i] and self.fit_gaia:

                    q = np.exp(y['log(q) {0}'.format(i+1)])
                    self.dres['q {0}'.format(i+1)] = q

                    mp = (q * mstar*u.Msun).to(u.earthMass).value
                    self.dres['Mp {0}'.format(i+1)] = mp

                    if self.is_transit[i] and self.fit_transit:

                        rhop = mp / (4/3 * np.pi * rp**3) * (1*u.earthMass/u.earthRad**3).to(u.g/u.cm**3).value
                        self.dres['rhop {0}'.format(i+1)] = rhop

                        omega = np.arctan2(y['somega {0}'.format(i+1)], y['comega {0}'.format(i+1)])
                        self.dres['omega {0}'.format(i+1)] = omega

                    else:

                        inc = np.arccos(1-2*(y['sico {0}'.format(i+1)]**2 + y['siso {0}'.format(i+1)]**2))*180/np.pi
                        self.dres['i {0}'.format(i+1)] = inc

                        omega = np.arctan2(y['siso {0}'.format(i+1)], y['sico {0}'.format(i+1)])
                        self.dres['omega {0}'.format(i+1)] = omega

                    k = 29785.326 * np.sin(inc*np.pi/180) / np.sqrt(1-e**2) * (p*u.day).to(u.yr).value**(-1/3) * q/(1+q)**(2/3) * mstar**(1/3)
                    self.dres['K {0}'.format(i+1)] = k


                elif self.is_rv[i] and self.fit_rv:

                    k = np.exp(y['log(K) {0}'.format(i+1)])
                    self.dres['K {0}'.format(i+1)] = k

                    if (self.is_transit[i] and self.fit_transit) or (self.is_astro[i] and self.fit_gaia):

                        mp = calc_m_from_k(p*(1*u.day).to(u.yr).value, k, e, inc*np.pi/180, mstar)
                        self.dres['Mp {0}'.format(i+1)] = mp

                        if self.is_transit[i] and self.fit_transit:

                            rhop = mp / (4/3 * np.pi * rp**3) * (1*u.earthMass/u.earthRad**3).to(u.g/u.cm**3).value
                            self.dres['rhop {0}'.format(i+1)] = rhop

                    else:

                        mp = calc_m_from_k(p*(1*u.day).to(u.yr).value, k, e, np.pi/2, mstar)
                        self.dres['Mpsini {0}'.format(i+1)] = mp

                if self.fit_star or self.fit_gaia or (not (self.fit_transit and self.is_transit[i]) and (self.is_rv[i] and self.fit_rv)):

                    a = ((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3)
                    self.dres['a {0}'.format(i+1)] = a

                    ars = (a*u.AU).to(u.Rsun).value / rstar
                    self.dres['a/rs {0}'.format(i+1)] = ars

                if (self.fit_transit and self.is_transit[i]) or (self.fit_rv and self.is_rv[i]) or (self.fit_gaia and self.is_astro[i]):

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

                    if not self.fit_star and not self.fit_gaia:

                        rhos = 0.018916375 * ars**3 / p**2
                        self.dres['rhos {0}'.format(i+1)] = rhos

                    if ((self.is_rv[i] and self.fit_rv) or (self.is_astro[i] and self.fit_gaia)) and hasJ:

                        sf = np.full(n, 0.19)
                        j = np.where(rp > 1.5)[0]
                        sf[j] = 1.26
                        j = np.where(rp > 2.75)[0]
                        sf[j] = 1.28
                        j = np.where(rp > 4)[0]
                        sf[j] = 1.15

                        tsm = sf * rp**3 * teq / (mp * rstar**2) * 10**(-J/5)
                        self.dres['TSM {0}'.format(i+1)] = tsm



        pickle.dump(self.dres, open(self.direc+'Output/'+name+'_dres.p', 'wb'))


        par_units = {'log(P)': 'days', 'Tc': 'BJD-2450000', 'ror': '', 'log(a/rs)': '', 'cos(i)': '', 'log(K)': 'm/s', 'log(q)': '', 'sico': '',
                     'siso': '', 'comega': '', 'somega': '', 'secw': '', 'sesw': '', 'fp': '', 'TT': 'BJD-2450000', 'F0': '', 'log(rho_gp)': 'days',
                     'log(sigma_gp)': '', 'gamma': 'm/s', 'gamma_dot': 'm/s/day', 'gamma_ddot': 'm/s/day^2', 'rv_offset': 'm/s', 'ra*_offset': 'mas',
                     'dec_offset': 'mas', 'pmra*': 'mas/yr', 'pmdec': 'mas/yr', 'u1': '', 'u2': '', 'eep': '', 'log10(age)': 'yr', 'feh': 'dex',
                     'plx': 'mas', 'distance': 'pc', 'AV': 'mag', 'rstar': 'Rsun', 'mstar': 'Msun', 'log(rhostar)': 'g/cm^3', 'rhostar': 'g/cm^3',
                     'mstar_init': 'Msun', 'Tstar': 'K', 'loggstar': 'cm/s^2', 'Lstar': 'Lsun', 'Mbolstar': 'mag', 'P': 'days', 'Rp': 'Rearth',
                     'a/rs': '', 'a': 'AU', 'i': 'deg', 'e': '', 'w': 'rad', 'K': 'm/s', 'q': '', 'omega': 'rad', 'Mp': 'Mearth', 'rhop': 'g/cm^3',
                     'Mpsini': 'Mearth', 'teq': 'K', 'sinc': 'Searth', 'b': '', 'depth': 'ppm', 'dur': 'hr', 'rhos': 'g/cm^3', 'TSM': '',
                     'Tp': 'BJD-2450000',}


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

        run_settings = {'fit_transit': self.fit_transit, 'fit_rv': self.fit_rv, 'fit_gaia': self.fit_gaia, 'fit_star': self.fit_star, 'fit_ld': self.fit_ld, 'use_priors': self.use_priors, 'lc_supersample_size': self.lc_supersample_size}
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

        if self.fit_gaia:

            self.gen_Gaia_astrometry(name)

            self.plot_Gaia_AL(name, show_plot = show_plots)

            self.plot_Gaia_onsky(name, show_plot = show_plots)

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
        - Tc x: Time of conjunction, in BJD-2457000, for planet number x. Defined for non-transiting planets as the time when the true anomaly is
            pi - w. For circular orbits, w is pi/2, and this is equal to Tp.
        - ror x: Planet to star radius ratio, for planet number x.
        - log(a/rs) x: Natural log of the orbital semi-major axis to stellar radius ratio, for planet number x.
        - cos(i) x: Cosine of the orbital inclination, for planet number x. Restricted between 0 and 90 degrees for transiting planets, extended
            to -90 to 90 degrees for astrometric planets.
        - log(K) x: Natural log of the RV semi-amplitude, in m/s, for planet number x.
        - secw x: Square root of the eccentricity times the cosine of the argument of periastron, for planet number x.
        - sesw x: Square root of the eccentricity times the sine of the argument of periastron, for planet number x.
        - e x: Orbital eccentricity, for planet number x. Only fit if w is fixed.
        - w x: Orbital argument of periastron, in radians, for planet number x. Only fit if e is fixed.
        - fp x: Planet to star flux ratio, for planet number x.
        - log(q) x: Natural log of the planet to star mass ratio, for planet number x.
        - sico x: Sine of the inclination/2 times the cosine of the longitude of the ascending node, for planet number x. Fit for astrometric-only planets.
        - siso x: Sine of the inclination/2 times the sine of the longitude of the ascending node, for planet number x. Fit for astrometric-only planets.
        - comega x: Cosine of the longitude of the ascending node, for planet x. Fit for astrometric+transiting planets, or astrometric-only planets
            in the case of fixed cos(i) or somega.
        - somega x: Sine of the longitude of the ascending node, for planet x. Fit for astrometric+transiting planets, or astrometric-only planets
            in the case of fixed cos(i) or comega.
        - mstar: The stellar mass, in solar masses. Fit in Gaia-only fits.
        - rstar: The stellar radius, in solar radii. Fit in Gaia+transit fits.
        - log(rhostar): The natural log of the mean stellar density, in g/cm^3. Fit in Gaia+transit fits.
        - TT x y: Transit time (when fitting for TTVs), in BJD-2457000, for planet number x transit number y.
        - F0 x: Baseline flux value, for transit data set x.
        - log(rho_gp) x: Natural log of the GP period, in days, for transit data set x.
        - log(sigma_gp) x: Natural log of the GP standard deviation, for transit data set x.
        - gamma: RV systemic velocity, in m/s.
        - gamma_dot: 1st derivative of the RV systemic velocity, in m/s/day. Relative to time of first RV measurement.
        - gamma_ddot: 2nd derivative of the RV systemic velocity, in m/s/day^2. Relative to time of first RV measurement.
        - rv_offset x: RV offset relative to first data set, in m/s, for RV data set x.
        - ra*_offset: The offset from the reference value of right ascension times cosine of declination, in mas.
        - dec_offset: The offset from the reference value of declination, in mas.
        - pmra*: Proper motion in right ascension times cosine of declination, in mas/yr
        - pmdec: Proper motion in declination, in mas/yr.
        - u1 x: Linear coefficient for quadratic limb darkening, for filter x.
        - u2 x: Non-linear coefficient for quadratic limb darkening, for filter x.
        - eep: Equivalent evolutionary phase, used to interpolate across MIST models for stellar fitting.
        - log10(age): Log base 10 of the stellar age, in yr.
        - feh: Stellar metallicity, in dex.
        - plx: Parallax angle of the star, in mas.
        - AV: Visual band extinction to the system, in mags.
        """
        return self.res
    
    @property
    def derived_chains(self):
        """Dict which stores flat chains of derived parameters. Keys are the parameter names.

        Potential derived parameters:
    
        - P x: Orbital period, in days, for planet number x.
        - Tp x: Time of periastron, in BJD-2457000, for planet number x. Equal to Tc for circular orbits.
        - Rp x: Planetary radius, in earth radii, for planet number x.
        - a/rs x: Orbital semi-major axis to stellar radius ratio, for planet number x.
        - a x: Orbital semi-major axis, in AU, for planet number x.
        - i x: Orbital inclination, in degrees, for planet number x. Restricted between 0 and 90 degrees for transiting planets, extended
            to -90 to 90 degrees for astrometric planets.
        - e x: Orbital eccentricity, for planet number x.
        - w x: Orbital argument of periastron, in radians, for planet number x.
        - omega x: Orbital longitude of the ascending node, in radians, for planet number x.
        - K x: RV semi-amplitude, in m/s, for planet number x.
        - q x: The planet to star mass ratio, for planet number x.
        - Mp x: Planetary mass, in earth masses, for planet number x.
        - Mpsini x: Planetary mass times the sine of the orbital inclination, in earth masses, for planet number x (used when transit wasn't fit).
        - rhop x: Planetary mean density, in g/cm^3, for planet number x.
        - teq x: Planetary equilibrium temperature (zero-albedo blackbody), in K, for planet number x.
        - sinc x: Insolation flux reaching the planet, in units of Earth insolation flux, for planet number x.
        - b x: Transit impact parameter, for planet number x.
        - depth x: Transit depth, in ppm, for planet number x.
        - dur x: Transit duration, in hours, for planet number x.
        - rhos x: Implied stellar density from transit of planet number x, in g/cm^3. Only when fitting transits with stellar parameters or Gaia data.
        - TSM x: Transmission spectroscopy metric, for planet number x.
        - rstar: Stellar radius, in solar radii.
        - mstar: Stellar mass, in solar masses.
        - distance: Distance to the system, in pc.
        - mstar_init: Initial stellar mass at the zero-age main sequence point, in solar masses.
        - rhostar: Mean stellar density, in g/cm^3.
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
            self.load_gaia_mod(name)
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
        """Loads in the best fit transit model of a previous run into ExoSystem._lcm.

        Args:
            name (str): Name of the previous run to load in.
        """

        self._lcm = pickle.load(open(self.direc+'/Output/'+name+'_lcm.p', 'rb'))


    def load_rv_mod(self, name: str):
        """Loads in the best fit RV model of a previous run into ExoSystem._rvm.

        Args:
            name (str): Name of the previous run to load in.
        """

        self._rvm = pickle.load(open(self.direc+'/Output/'+name+'_rvm.p', 'rb'))

    
    def load_gaia_mod(self, name: str):
        """Loads in the best fit Gaia astrometry model of a previous run into ExoSystem._gaiam.

        Args:
            name (str): Name of the previous run to load in.
        """

        self._gaiam = pickle.load(open(self.direc+'/Output/'+name+'_gaiam.p', 'rb'))


    def load_magfit(self, name: str):
        """Loads in the best fit stellar SED model of a previous run into ExoSystem.magfit and ExoSystem.magfiterr.

        Args:
            name (str): Name of the previous run to load in.
        """

        mags = pickle.load(open(self.direc+'/Output/'+name+'_magfit.p', 'rb'))
        magbands = mags['magbands']
        idx = [np.where(magbands == self.magbands[i])[0][0] for i in range(len(self.magbands))]
        self.magfit = mags['magfit'][idx]
        self.magfiterr = mags['magfiterr'][idx]

    
    def load_samples(self, name: str):
        """Loads in unflattened, un-thinned MCMC chains. Paramater name map is saved to ExoSystem.parnames, chains are saved to ExoSystem.samples,
        log likelihood values are saved to ExoSystem.log_likes.

        Args:
            name (str): Name of the previous run to load in.
        """

        z = pickle.load(open(self.direc+'/Output/'+name+'_samples.p', 'rb'))
        self.parnames = z['parnames']
        self.samples = z['samples']
        self.log_likes = z['log_like']

    
    def load_run_settings(self, name: str):
        """Loads in run settings from the run, specifically fit_transit, fit_rv, fit_star, fit_ld, use_priors, lc_supersample_size, and any parameters
        which were fixed. These are used to tell plotting functions what they need to plot.

        Args:
            name (str): Name of the previous run to load in.
        """
        
        run_settings = pickle.load(open(self.direc+'/Output/'+name+'_run_settings.p', 'rb'))
        self.fit_transit = run_settings['fit_transit']
        self.fit_rv = run_settings['fit_rv']
        self.fit_gaia = run_settings['fit_gaia']
        self.fit_star = run_settings['fit_star']
        self.fit_ld = run_settings['fit_ld']
        self.use_priors = run_settings['use_priors']
        self.lc_supersample_size = run_settings['lc_supersample_size']
        if self.use_priors:
            self.fixed = run_settings['fixed']
        self.fit_planets = self.fit_transit or self.fit_rv or self.fit_gaia


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

        if os.path.exists(self.direc+'Output/'+name+'_gaiam.p'):
            os.remove(self.direc+'Output/'+name+'_gaiam.p')

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

            if not ((self.fit_transit and self.is_transit[i]) or (self.fit_rv and self.is_rv[i]) or (self.fit_gaia and self.is_astro[i])):

                continue

            num2 += 1
            vargrid.append([])

            for z in ['log(P)','Tc','ror','log(a/rs)','cos(i)','log(K)','log(q)','secw','sesw','e','w','sico','siso','comega','somega','fp']:

                if z+' {0}'.format(i+1) in self.parnames:
                    vargrid[-1].append(z)

            num1 = max(num1, len(vargrid[-1]))

        if self.fit_gaia and not self.fit_star:

            num2 += 1
            vargrid.append([])

            for z in ['plx','mstar','log(rhostar)','rstar']:

                if z in self.parnames:
                    vargrid[-1].append(z)

            num1 = max(num1, len(vargrid[-1]))

        fig, ax = plt.subplots(num1, num2, figsize = (7*num2, 2*num1), sharex = True, layout = 'constrained')

        if num2 == 1:
            ax = np.array([ax.T]).T

        j = 0
        for i in range(self.n):

            if not ((self.fit_transit and self.is_transit[i]) or (self.fit_rv and self.is_rv[i]) or (self.fit_gaia and self.is_astro[i])):

                continue

            ax[0][j].set_title('Planet {0}'.format(i+1))

            for k in range(len(vargrid[j])):

                v = self.parnames[vargrid[j][k]+' {0}'.format(i+1)]
                ax[k][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
                ax[k][j].text(0.01, 0.99, vargrid[j][k], fontsize = 20, ha = 'left', va = 'top', transform = ax[k][j].transAxes, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

            j += 1

        if self.fit_gaia and not self.fit_star:

            ax[0][j].set_title('Star')

            for k in range(len(vargrid[j])):

                v = self.parnames[vargrid[j][k]+' {0}'.format(i+1)]
                ax[k][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
                ax[k][j].text(0.01, 0.99, vargrid[j][k], fontsize = 20, ha = 'left', va = 'top', transform = ax[k][j].transAxes, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

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

        fig, ax = plt.subplots(np.sum(self.detrend), 2, figsize = (10, 3*len(self.tt)), sharex = True, layout = 'constrained')

        if np.sum(self.detrend) == 1:
            ax = np.array([ax])

        for j in range(len(self.tt)):

            if not self.detrend[j]:
                continue

            i = np.sum(self.detrend[:j])

            ax[i][0].set_ylabel('{0}'.format(self.lcnames[j]))

            if 'log(rho_gp) {0}'.format(self.lcnames[j]) in self.parnames:
                v = self.parnames['log(rho_gp) {0}'.format(self.lcnames[j])]
                ax[i][0].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            if 'log(sigma_gp) {0}'.format(self.lcnames[j]) in self.parnames:
                v = self.parnames['log(sigma_gp) {0}'.format(self.lcnames[j])]
                ax[i][1].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

        ax[0][0].set_title('log(rho_gp)')
        ax[0][1].set_title('log(sigma_gp)')

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

        for i, z in enumerate(['eep','log10(age)','feh','plx','AV']):

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

            for z in ['log(P)','Tc','ror','log(a/rs)','cos(i)','log(K)','log(q)','secw','sesw','e','w','sico','siso','comega','somega','fp']:

                if z+' {0}'.format(i+1) in self.parnames:

                    labels.append(z+' {0}'.format(i+1))

            if self.fit_transit and self.is_transit[i] and self.fit_ttv[i]:

                for k in range(len(self.ttvi['{0}'.format(i+1)])):

                    if 'TT {0} {1}'.format(i+1, k+1) in self.parnames:
                        labels.append('TT {0} {1}'.format(i+1, k+1))

            if self.fit_gaia and not self.fit_star:

                for z in ['plx','mstar','log(rhostar)','rstar']:

                    if z in self.parnames:
                        labels.append(z)

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

        for z in ['eep','log10(age)','feh','plx','AV']:
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

            y['log(rhostar)'] = np.log(y['rhostar'])

            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                p = y['P {0}'.format(j+1)]

                if 'log(a/rs) {0}'.format(i+1) not in y:

                    y['log(a/rs) {0}'.format(i+1)] = np.log(y['a/rs {0}'.format(i+1)])

        pars = []
        tcs = []

        for j in range(self.n):

            if not self.is_transit[j]:
                continue
            
            p = y['P {0}'.format(j+1)]
            tc = y['Tc {0}'.format(j+1)]
            ror = y['ror {0}'.format(j+1)]
            ar = y['a/rs {0}'.format(j+1)]
            inc = y['i {0}'.format(j+1)]
            e = y['e {0}'.format(j+1)] if 'e {0}'.format(j+1) in y else np.full((n), 0)
            w = y['w {0}'.format(j+1)]*180/np.pi if 'w {0}'.format(j+1) in y else np.full((n), 90)
            
            pars.append(np.array([p,tc,ror,ar,inc,e,w]).T)

            tcs.append(np.median(tc))


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

                starpars = np.array([y['Tstar'], y['loggstar'], y['feh']]).T
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

                    rs = np.median(y['rstar'])
                    q = 0

                    if 'Mp {0}'.format(k+1) in y:
                        mp = (np.median(y['Mp {0}'.format(k+1)])*u.earthMass).to(u.Msun).value
                        ms = np.median(y['mstar'])
                        q = mp/ms

                    tc2 = calc_t_sec(medpar[0], medpar[1], medpar[5], medpar[6]*np.pi/180, medpar[3], medpar[4]*np.pi/180, rs, q)
                    eclphase.append(lightCurve(medpar, ttphase + tc2, np.median(ld, axis = 0) if ld.ndim > 1 else ld, phaseexp, phasess, eclipse = True, fp = np.median(y['fp {0}'.format(k+1)]), rstar = rs, q = q))


            fmphase_err = [[] for j in range(self.nt)]
            eclphase_err = [[] for j in range(np.sum(self.is_eclipse))]

            for j in tqdm(range(0,n,200)):

                for k in range(self.n):

                    if not self.is_transit[k]:
                        continue

                    l = np.sum(self.is_transit[:k])

                    fmphase_err[l].append(lightCurve(pars[l][j], ttphase + tcs[l], ld[j] if ld.ndim > 1 else ld, phaseexp, phasess))

                    if self.is_eclipse[k]:

                        rs = y['rstar'][j]
                        q = 0
    
                        if 'Mp {0}'.format(k+1) in y:
                            mp = (y['Mp {0}'.format(k+1)][j]*u.earthMass).to(u.Msun).value
                            ms = y['mstar'][j]
                            q = mp/ms

                        tc2 = calc_t_sec(pars[l][j][0], pars[l][j][1], pars[l][j][5], pars[l][j][6]*np.pi/180, pars[l][j][3], pars[l][j][4]*np.pi/180, rs, q)
                        le = np.sum(self.is_eclipse[:k])
                        eclphase_err[le].append(lightCurve(pars[l][j], ttphase + tc2, ld[j] if ld.ndim > 1 else ld, phaseexp, phasess, eclipse = True, fp = y['fp {0}'.format(k+1)][j], rstar = rs, q = q))

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

                starpars = np.array([y['Tstar'], y['loggstar'], y['feh']]).T
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

                        rs = np.median(y['rstar'])
                        q = 0
    
                        if 'Mp {0}'.format(k+1) in y:
                            mp = (np.median(y['Mp {0}'.format(k+1)])*u.earthMass).to(u.Msun).value
                            ms = np.median(y['mstar'])
                            q = mp/ms

                        fm0 = lightCurve(np.median(pars[l], axis = 0), self.tt[i], np.median(ld, axis = 0) if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i], eclipse = True, fp = np.median(y['fp {0}'.format(k+1)]), rstar = rs, q = q)

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

            for j in tqdm(range(0,n,200)):

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

                            rs = y['rstar'][j]                           
                            q = 0
        
                            if 'Mp {0}'.format(k+1) in y:
                                mp = (y['Mp {0}'.format(k+1)][j]*u.earthMass).to(u.Msun).value
                                ms = y['mstar'][j]
                                q = mp/ms

                            fm0 = lightCurve(pars[l][j], self.tt[i], ld[j] if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i], eclipse = True, fp = y['fp {0}'.format(k+1)][j], rstar = rs, q = q)

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

        lc_mod is a dict, with a key for the phase folded models as well as for each light curve data set using their nicknames.

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

        eclipse = np.any(self.is_eclipse)

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
                    inc = np.median(y['i {0}'.format(j+1)])*np.pi/180
                    rs = np.median(y['rstar'])

                    mp = 0
                    ms = np.median(y['mstar'])
                    if 'Mp {0}'.format(j+1) in y:
                        mp = (np.median(y['Mp {0}'.format(j+1)])*u.earthMass).to(u.Msun).value

                    q = mp/ms

                    tc2 = calc_t_sec(p, tc, e, w, ar, inc, rs, q)

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

            pars.append(get_rv_params(y, i+1, shape = (n)).T)
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

        for i in tqdm(range(0,n,20)):

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

        rv_mod is a dict,  with the following keys:

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

        gamma = np.median(y['gamma'])

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

            ax['a'].plot(trplot, rvmplot[j] + gamma, linewidth = 2, linestyle = '--', zorder = 1, label = 'Planet {0}'.format(i+1))

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


    def gen_Gaia_astrometry(self, name: str):
        """Generates Gaia astrometry models from the best fit parameters. Saves these to the pickle file Output/name_gaiam.p. These astrometry models
        are accessible through ExoSystem.gaia_mod. See the documentation on ExoSystem.gaia_mod for details on the formatting of the object.

        Args:
            name (str): Name of the run. Sets the name of the output pickle file to name_gaiam.p.
        """

        print('\nGenerating Gaia astrometry models for plots.')

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
        maxp = 0
        maxtc = 0

        for i in range(self.n):

            if not self.is_astro[i]:
                continue

            pars.append(get_Gaia_params(y, i+1, shape = (n)).T)

            if np.median(y['P {0}'.format(i+1)]) > maxp:
                maxp = np.median(y['P {0}'.format(i+1)])
                maxtc = np.median(y['Tc {0}'.format(i+1)])

        plx = np.median(y['plx'])

        gm = 0
        gmra = 0
        gmdec = 0
        gmskyra = 0
        gmskydec = 0
        gmorbra = 0
        gmorbdec = 0

        tgsky = np.linspace(np.min(self.tg)-50, np.max(self.tg)+50, 1000)
        tgorb = np.linspace(maxtc, maxtc+maxp, 1000)

        gmorbra_pl = []
        gmorbdec_pl = []

        xpts = []
        ypts = []

        for j in range(self.n):

            if not self.is_astro[j]:
                continue

            k = np.sum(self.is_astro[:j])

            medpars = np.median(pars[k], axis = 0)

            gm0 = Gaia_AL_Model(medpars, self.tg, self.psi)
            gm += gm0[2]
            gmra += gm0[0]
            gmdec += gm0[1]

            gmsky = Gaia_AL_Model(medpars, tgsky, np.zeros(tgsky.shape))
            gmskyra += gmsky[0]
            gmskydec += gmsky[1]

            gmorb = Gaia_AL_Model(medpars, tgorb, np.zeros(tgorb.shape))
            gmorbra += gmorb[0]
            gmorbdec += gmorb[1]

            if np.sum(self.is_astro) > 1:
                
                gmorbra_pl.append(gmorbra)
                gmorbdec_pl.append(gmorbdec)

            p = medpars[0]
            tp = np.median(y['Tp {0}'.format(j+1)])
            e = medpars[2]
            w = medpars[3]
            fasc = -w
            Easc = np.arctan2(np.sqrt(1-e**2) * np.sin(fasc), e + np.cos(fasc))
            tasc = p/(2*np.pi) * (Easc - e * np.sin(Easc)) + tp
            fdesc = -w + np.pi
            Edesc = np.arctan2(np.sqrt(1-e**2) * np.sin(fdesc), e + np.cos(fdesc))
            tdesc = p/(2*np.pi) * (Edesc - e * np.sin(Edesc)) + tp
            ts = np.array([tp, tp+p/2, tasc, tdesc])

            _, xts, yts = Gaia_AL_Model(medpars, ts, np.zeros(ts.shape))
            xpts.append(xts)
            ypts.append(yts)

        gclean = self.gal - gm - self.plxf * plx
        tgyr = (self.tg - self.tgref)/365.25

        M = np.zeros((len(self.tg),4))
        M[:,0] = np.sin(self.psi)
        M[:,1] = np.cos(self.psi)
        M[:,2] = tgyr*np.sin(self.psi)
        M[:,3] = tgyr*np.cos(self.psi)
    
        W = 1/self.galerr**2
        Mw = M * np.sqrt(W)[:, None]
        Yw = gclean * np.sqrt(W)
    
        X, _, _, _ = np.linalg.lstsq(Mw, Yw, rcond = None)

        bkg = M @ X + self.plxf * plx

        ra, dec = self.ra*np.pi/180, self.dec*np.pi/180
        pvec = np.array([-np.sin(ra), np.cos(ra), 0])
        qvec = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])

        tplx = Time(self.tg, format = 'jd', scale = 'tdb')
        earth = get_body_barycentric('earth', tplx)
        Rxyz = -np.array([earth.x.value, earth.y.value, earth.z.value]).T
        fra = Rxyz @ pvec
        fdec = Rxyz @ qvec
        raplx = plx * fra
        decplx = plx * fdec

        bkgra = X[2] * tgyr + raplx
        bkgdec = X[3] * tgyr + decplx

        tplxsky = Time(tgsky, format = 'jd', scale = 'tdb')
        earthsky = get_body_barycentric('earth', tplxsky)
        Rxyzsky = -np.array([earthsky.x.value, earthsky.y.value, earthsky.z.value]).T
        frasky = Rxyzsky @ pvec
        fdecsky = Rxyzsky @ qvec
        raplxsky = plx * frasky
        decplxsky = plx * fdecsky

        bkgskyra = X[2] * (tgsky - self.tgref)/365.25 + raplxsky
        bkgskydec = X[3] * (tgsky - self.tgref)/365.25 + decplxsky

        gm_err = []

        for i in tqdm(range(0,n,20)):

            gm0 = self.plxf * y['plx'][i] + (y['ra*_offset'][i] + y['pmra*'][i]*tgyr)*np.sin(self.psi) + (y['dec_offset'][i] + y['pmdec'][i]*tgyr)*np.cos(self.psi)

            for j in range(self.n):

                if not self.is_astro[j]:
                    continue

                k = np.sum(self.is_astro[:j])

                gm0 += Gaia_AL_Model(pars[k][i], self.tg, self.psi)

            gm_err.append(gm0)


        gmskyra_draws = []
        gmskydec_draws = []
        gmorbra_draws = []
        gmorbdec_draws = []

        idxs = np.random.choice(n, 100, replace = False)
        for i in idxs:

            gmskyra0 = 0
            gmskydec0 = 0
            gmorbra0 = 0
            gmorbdec0 = 0

            for j in range(self.n):

                if not self.is_astro[j]:
                    continue

                k = np.sum(self.is_astro[:j])

                gmsky = Gaia_AL_Model(np.median(pars[k], axis = 0), tgsky, np.zeros(tgsky.shape))
                gmskyra0 += gmsky[0]
                gmskydec0 += gmsky[1]

                gmorb = Gaia_AL_Model(np.median(pars[k], axis = 0), tgorb, np.zeros(tgorb.shape))
                gmorbra0 += gmorb[0]
                gmorbdec0 += gmorb[1]

            gmskyra_draws.append(gmskyra0)
            gmskydec_draws.append(gmskydec0)
            gmorbra_draws.append(gmorbra0)
            gmorbdec_draws.append(gmorbdec0)


        gm_err = np.percentile(gm_err, [16,84], axis = 0)

        self._gaiam = {'gm': np.array(gm), 'bkg': np.array(bkg), 'gm_err': np.array(gm_err), 'gmra': gmra, 'gmdec': gmdec, 'bkgra': np.array(bkgra), 'bkgdec': np.array(bkgdec), 'tgsky': tgsky, 'gmskyra': np.array(gmskyra), 'gmskydec': np.array(gmskydec), 'gmskyra_draws': np.array(gmskyra_draws), 'gmskydec_draws': np.array(gmskydec_draws), 'bkgskyra': np.array(bkgskyra), 'bkgskydec': np.array(bkgskydec), 'tgorb': tgorb, 'gmorbra': np.array(gmorbra), 'gmorbdec': np.array(gmorbdec), 'gmorbra_draws': np.array(gmorbra_draws), 'gmorbdec_draws': np.array(gmorbdec_draws), 'gmorbra_pl': np.array(gmorbra_pl), 'gmorbdec_pl': np.array(gmorbdec_pl), 'xpts': np.array(xpts), 'ypts': np.array(ypts)}

        pickle.dump(self._gaiam, open(self.direc+'Output/'+name+'_gaiam.p', 'wb'))


    @property
    def gaia_mod(self):
        """Stores the best fit Gaia astrometry models.

        gaia_mod is a dict,  with the following keys:

        - gm: A 1d numpy array of the best fit Gaia along scan offsets, only including the orbital motion, at the observation times. Must be
            combined with bkg to match the data.
        - bkg: A 1d numpy array of the values of the along scan offsets due to the proper motion, position offsets, and parallax motion, at the
            observation times.
        - gm_err: A 2d numpy array of the 1 sigma uncertainties on the full gm + bkg model at the observation times. In the 0th axis, the 0 index
            corresponds to a 1d array of the lower 1 sigma error and the 1 index corresponds to a 1d array of the upper 1 sigma error.
        - gmra: A 1d numpy array of the best fit Gaia model in the ra* direction, only including the orbital motion, at the observation times.
        - gmdec: A 1d numpy array of the best fit Gaia model in the dec direction, only including the orbital motion, at the observation times.
        - bkgra: A 1d numpy array of the ra* proper motion and ra* component of the parallax motion, at the observation times.
        - bkgdec: A 1d numpy array of the dec proper motion and dec component of the parallax motion, at the observation times.
        - tgsky: A 1d numpy array of more finely sampled times to make the plotted on-sky models look smooth, spanning the range of the observation
            times.
        - gmskyra: A 1d numpy array of the best fit Gaia model in the ra* direction, only including the orbital motion, at the times of tgsky.
        - gmskydec: A 1d numpy array of the best fit Gaia model in the dec direction, only including the orbital motion, at the times of tgsky.
        - gmskyra_draws: A 2d numpy array of 100 orbit models in the ra* direction randomly drawn from the chains, at the times of tgsky. The 0th
            axis corresponds to each different model, where each of these is a 1d array similar to gmskyra.
        - gmskydec_draws: A 2d numpy array of 100 orbit models in the dec direction randomly drawn from the chains, at the times of tgsky. The 0th
            axis corresponds to each different model, where each of these is a 1d array similar to gmskydec.
        - bkgskyra: A 1d numpy array of the ra* proper motion and ra* component of the parallax motion, at the times of tgsky.
        - bkgskydec: A 1d numpy array of the dec proper motion and dec component of the parallax motion, at the times of tgsky.
        - tgorb: A 1d numpy array of more finely sampled times to make the plotted 2d orbit models look smooth, spanning a full period of the
            longest period astrometric planet.
        - gmorbra: A 1d numpy array of the best fit Gaia model in the ra* direction, only including the orbital motion, at the times of tgorb.
        - gmorbdec: A 1d numpy array of the best fit Gaia model in the dec direction, only including the orbital motion, at the times of tgorb.
        - gmorbra_draws: A 2d numpy array of 100 orbit models in the ra* direction randomly drawn from the chains, at the times of tgorb. The 0th
            axis corresponds to each different model, where each of these is a 1d array similar to gmorbra.
        - gmorbdec_draws: A 2d numpy array of 100 orbit models in the dec direction randomly drawn from the chains, at the times of tgorb. The 0th
            axis corresponds to each different model, where each of these is a 1d array similar to gmorbdec.
        - gmorbra_pl: A 2d numpy array of the best fit Gaia model for each individual astrometric planet in the ra* direction, only including the
            orbital motion, at the times of tgorb. The 0th axis corresponds to each astrometric planet, where each of these is a 1d array similar to
            gmorbra. If there is only 1 astrometric planet, this is empty as gmorbra would be an identical model.
        - gmorbdec_pl: A 2d numpy array of the best fit Gaia model for each individual astrometric planet in the dec direction, only including the
            orbital motion, at the times of tgorb. The 0th axis corresponds to each astrometric planet, where each of these is a 1d array similar to
            gmorbdec. If there is only 1 astrometric planet, this is empty as gmorbdec would be an identical model.
        - xpts: A 2d numpy array of the relative ra* positions on the orbit of the periastron, apastron, ascending node, and descending node for each
            planet. The 0th axis corresponds to each planet.
        - ypts: A 2d numpy array of the relative dec positions on the orbit of the periastron, apastron, ascending node, and descending node for each
            planet. The 0th axis corresponds to each planet.
        """
        return self._gaiam


    def plot_Gaia_AL(self, name: str, show_plot = True):
        """Plots the best fit Gaia along scan model to the data. Saves to the folder in Plots set by name. The plot is named gaia_al.png. Shows the
        plot if show_plot is True.

        The top panel shows the full Gaia along scan offset time series, with the best fit model in green. The residuals are plotted below it.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        fig, ax = plt.subplot_mosaic([['a'],['a'],['b']], figsize = (14,8), layout = 'constrained', sharex = True)

        gm = self._gaiam['gm']
        bkg = self._gaiam['bkg']
        gmerr = self._gaiam['gm_err']

        ax['a'].errorbar(self.tg, self.gal, yerr = self.galerr, fmt = '.k', markersize = 10, elinewidth = 2, zorder = 2)
        ax['a'].plot(self.tg, gm+bkg, color = 'mediumseagreen', linewidth = 2, zorder = 1)
        ax['a'].fill_between(self.tg, gmerr[0], gmerr[1], color = 'mediumseagreen', zorder = 0, alpha = 0.5, edgecolor = 'none')

        ax['a'].tick_params(axis = 'both', labelsize = 15)
        ax['a'].set_ylabel('Gaia AL Offset [mas]', fontsize = 20)

        ax['b'].errorbar(self.tg, self.gal-gm-bkg, yerr = self.galerr, fmt = '.k', markersize = 10, elinewidth = 2, zorder = 2)
        ax['b'].axhline(0, c = 'red', lw = 1, zorder = 1)

        ax['b'].tick_params(axis = 'both', labelsize = 15)
        ax['b'].set_xlabel('Time [BJD-2450000]', fontsize = 20)
        ax['b'].set_ylabel('Resid. [mas]', fontsize = 20)

        fig.savefig(self.direc+'Plots/'+name+'/gaia_al.png')

        if show_plot:
            plt.show()

        else:
            plt.close()

    
    def plot_Gaia_onsky(self, name: str, show_plot = True):
        """Plots the best fit Gaia astrometic model to the data. Saves to the folder in Plots set by name. The plot is named gaia_onsky.png. Shows
        the plot if show_plot is True.

        The left panel shows the total motion on the sky, including the orbit, proper motion, and parallax motion. The right panel shows just the
        orbital motion.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
            
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """
        
        fig, ax = plt.subplots(1, 2, layout = 'constrained')

        y = self.res | self.dres
        
        if self.use_priors:

            y = y | self.fixed

        gmtot = self._gaiam['gm'] + self._gaiam['bkg']
        gmra, gmdec = self._gaiam['gmra'], self._gaiam['gmdec']
        bkgra, bkgdec = self._gaiam['bkgra'], self._gaiam['bkgdec']

        gmskyra, gmskydec = self._gaiam['gmskyra'], self._gaiam['gmskydec']
        bkgskyra, bkgskydec = self._gaiam['bkgskyra'], self._gaiam['bkgskydec']
        gmskyra_draws, gmskydec_draws = self._gaiam['gmskyra_draws'], self._gaiam['gmskydec_draws']

        pmra, pmdec = np.median(y['pmra*']), np.median(y['pmdec'])

        ax[0].scatter((self.gal-gmtot)*np.sin(self.psi) + gmra + bkgra, (self.gal-gmtot)*np.cos(self.psi) + gmdec + bkgdec, marker = '.', c = 'black', s = 100, zorder = 4)

        gmerrorbars = np.array([self.gal-gmtot-self.galerr, self.gal-gmtot+self.galerr])
        for i in range(len(self.tg)):
            ax[0].plot(gmerrorbars[i]*np.sin(self.psi[i]) + gmra[i] + bkgra[i], gmerrorbars[i]*np.cos(self.psi[i]) + gmdec[i] + bkgdec[i], c = 'black', linewidth = 2, zorder = 3)

        ax[0].plot(gmskyra + bkgskyra, gmskydec + bkgskydec, color = 'mediumseagreen', linewidth = 2, zorder = 2, label = 'Full Model')
        for i in range(len(gmskyra_draws)):
            ax[0].plot(gmskyra_draws[i] + bkgskyra, gmskydec_draws[i] + bkgskydec, color = 'darkseagreen', linewidth = 1, zorder = 1)
        
        ax[0].plot(bkgskyra, bkgskydec, color = 'darkred', linewidth = 2, zorder = 0, linestyle = '--', label = 'On-Sky Motion')

        xl, yl = ax[0].get_xlim(), ax[0].get_ylim()
        ax[0].set_xlim(xl)
        ax[0].set_ylim(yl)

        xrange, yrange = (xl[1]-xl[0]), (yl[1]-yl[0])
        xlen = xrange/10*pmra/np.sqrt(pmra**2+pmdec**2)
        ylen = yrange/10*pmdec/np.sqrt(pmra**2+pmdec**2)
        ya = yrange/8+xl[0]
        if pmra/pmdec >= 0:
            xa = xrange/8+xl[0]
        else:
            xa = xl[1]-xrange/8
        ax[0].annotate('', xytext = (xa-xlen, ya-ylen), xy = (xa+xlen, ya+ylen), arrowprops = dict(arrowstyle='->',color='black',linewidth=1))

        ax[0].set_box_aspect(1)
        ax[0].invert_xaxis()

        ax[0].legend(fontsize = 15)
        ax[0].tick_params(axis = 'both', labelsize = 15)
        ax[0].set_title('On-Sky Motion')
        ax[0].set_xlabel('$\\Delta \\alpha \\mathrm{cos}(\\delta)$ [mas]', fontsize = 20)
        ax[0].set_ylabel('$\\Delta \\delta$ [mas]')

        gmorbra, gmorbdec = self._gaiam['gmorbra'], self._gaiam['gmorbdec']
        gmorbra_draws, gmorbdec_draws = self._gaiam['gmorbra_draws'], self._gaiam['gmorbdec_draws']
        gmorbra_pl, gmorbdec_pl = self._gaiam['gmorbra_pl'], self._gaiam['gmorbdec_pl']
        xpts, ypts = self._gaiam['xpts'], self._gaiam['ypts']

        ax[1].scatter((self.gal-gmtot)*np.sin(self.psi) + gmra, (self.gal-gmtot)*np.cos(self.psi) + gmdec, marker = '.', c = 'black', s = 100, zorder = 4)

        for i in range(len(self.tg)):
            ax[1].plot(gmerrorbars[i]*np.sin(self.psi[i]) + gmra[i], gmerrorbars[i]*np.cos(self.psi[i]) + gmdec[i], c = 'black', linewidth = 2, zorder = 3)

        ax[1].plot(gmorbra, gmorbdec, color = 'mediumseagreen', linewidth = 2, zorder = 2, label = 'Full Model')
        for i in range(len(gmorbra_draws)):
            ax[1].plot(gmorbra_draws[i], gmorbdec_draws[i], color = 'darkseagreen', linewidth = 1, zorder = 1)

        ax[1].scatter(0, 0, c = 'black', marker = '*', s = 100, zorder = 3)

        for i in range(self.n):

            if not self.is_astro[i]:
                continue

            j = np.sum(self.is_astro[:i])

            if np.sum(self.is_astro) > 1:

                color = ['black','red','darkorange','green','blue','darkorchid'][j]

                ax[1].plot(gmorbra_pl[j], gmorbdec_pl[j], linewidth = 2, zorder = 2, label = 'Planet {0}'.format(i+1), color = color)

                ii = np.argmin(np.sqrt((gmorbra_pl[j]-xpts[j][1])**2 + (gmorbdec_pl[j]-ypts[j][1])**2))
                dx = gmorbra_pl[j][ii] - gmorbra_pl[j][ii-1]
                dy = gmorbdec_pl[j][ii] - gmorbdec_pl[j][ii-1]
                ax[1].annotate('', xytext = (xpts[j][1]-dx, ypts[j][1]-dy), xy = (xpts[j][1], ypts[j][1]), arrowprops = dict(arrowstyle = '->', color = color, lw = 0, mutation_scale = 15))

            else:
                ii = np.argmin(np.sqrt((gmorbra-xpts[j][1])**2 + (gmorbdec-ypts[j][1])**2))
                dx = gmorbra[ii] - gmorbra[ii-1]
                dy = gmorbdec[ii] - gmorbdec[ii-1]
                ax[1].annotate('', xytext = (xpts[j][1]-dx, ypts[j][1]-dy), xy = (xpts[j][1], ypts[j][1]), arrowprops = dict(arrowstyle = '->', color = 'mediumseagreen', lw = 0, mutation_scale = 15))

            ax[1].scatter(xpts[j][0], ypts[j][0], c = 'black', marker = '+', s = 100, zorder = 3, label = '$\\omega$' if j == 0 else None)
            ax[1].scatter(xpts[j][2], ypts[j][2], c = 'black', marker = 's', s = 100, zorder = 3, label = '\u260A' if j == 0 else None)
            ax[1].plot(xpts[j][2:], ypts[j][2:], c = 'gray', linestyle = '--', linewidth = 1, zorder = 2, label = 'Line of Nodes' if j == 0 else None)


        xl, yl = ax[1].get_xlim(), ax[1].get_ylim()
        limrange = max(xl[1]-xl[0], yl[1]-yl[0])
        xlm = (xl[1]+xl[0])/2
        ylm = (yl[1]+yl[0])/2
        ax[1].set_xlim(xlm-limrange/2, xlm+limrange/2)
        ax[1].set_ylim(ylm-limrange/2, ylm+limrange/2)
        ax[1].invert_xaxis()
        ax[1].set_aspect('equal')

        ax[1].legend(fontsize = 15, ncols = 2)
        ax[1].tick_params(axis = 'both', labelsize = 15)
        ax[1].set_title('Astrometric Orbit')
        ax[1].set_xlabel('$\\Delta \\alpha \\mathrm{cos}(\\delta)$ [mas]', fontsize = 20)
        ax[1].set_ylabel('$\\Delta \\delta$ [mas]')

        width, height = fig.get_size_inches()
        twidth = 14
        fig.set_figwidth(twidth)
        fig.set_figheight(twidth * height/width)

        fig.savefig(self.direc+'Plots/'+name+'/gaia_onsky.png')

        if show_plot:
            plt.show()

        else:
            plt.close()





    def gen_magfit(self, name: str):
        """Generates SED magnitudes and errors from the best fit parameters. Saves these to the pickle file Output/name_magfit.p. These model
        magnitudes are accessible through ExoSystem.magfit and ExoSystem.magfiterr. Also saves out the bands, so that the order can be matched.

        Args:
            name (str): Name of the run. Sets the name of the output pickle file to name_magfit.p.
        """

        y = self.res.copy()

        if self.use_priors:
            
            y = y | self.fixed

        mags = self.misti.interp_mag([y['eep'],y['log10(age)'],y['feh'],1000/y['plx'],y['AV']], list(self.magbands))[3]
        self.magfit = np.median(mags, axis = 0)
        self.magfiterr = np.diff(np.percentile(mags, [16,50,84], axis = 0), axis = 0)

        pickle.dump({'magfit': self.magfit, 'magfiterr': self.magfiterr, 'magbands': self.magbands}, open(self.direc+'Output/'+name+'_magfit.p', 'wb'))


    def plot_sed_fit(self, name: str, show_plot = True):
        """Plots the magnitudes of the best fit stellar parameters against the data. Residuals are shown below. Saves to the folder in Plots set
        by name. The plot is named sed.png. Shows the plot if show_plot is True.

        Args:
            name (str): Name of the run. Sets the folder in Plots to save this plot to.
           
            show_plot (bool, optional): Whether or not to show the plot. Default is True.
        """

        fig, ax = plt.subplot_mosaic([['a'],['a'],['b']], figsize = (14, 12), layout = 'constrained', sharex = True)

        ax['a'].errorbar(self.magwls, self.magfit, xerr = self.magwidths/2, yerr = self.magfiterr, fmt = '.r', label = 'Fit')
        ax['a'].errorbar(self.magwls, self.magobs, xerr = self.magwidths/2, yerr = self.magobserr, fmt = '.k', label = 'Data')
        ax['a'].invert_yaxis()
        ax['a'].set_ylabel('Magnitude', fontsize = 20)
        ax['a'].tick_params(axis = 'both', labelsize = 15)
        ax['a'].legend(fontsize = 15)

        ax['b'].axhline(0, c = 'red')
        ax['b'].errorbar(self.magwls, self.magobs-self.magfit, xerr = self.magwidths/2, yerr = np.sqrt(self.magobserr**2 + self.magfiterr**2), fmt = '.k')
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
        ExoSystem.calc_autocorr, ExoSystem.flatten_chains, ExoSystem.make_results, and ExoSystem.make_plots.

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

        if save_samples:

            self.save_samples(name)

        self.calc_autocorr()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('\nResults')
        self.restab.pprint_all()
        

    def burn_steps(self, name: str, num_steps: int, save_samples: bool = False, show_plots: bool = True):
        """Useful if the burn-in wasn't long enough. Cuts the first num_steps steps from samples, log_likes, and the lists of derived periods and tcs
        in a ttv fit. Then, reruns ExoSystem.calc_autocorr, ExoSystem.flatten_chains, ExoSystem.make_results, and ExoSystem.make_plots.

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

        if save_samples:

            self.save_samples(name)

        self.calc_autocorr()

        self.flatten_chains()

        self.make_results(name)

        self.make_plots(name, show_plots)

        print('\nResults')
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

        It can be useful to rescale errors if you believe they are under or overestimated, and a good place to start is looking at residual scatter.
        This is a simple alternative to using a jitter term in other fitting packages. Rescaling the errors can be very important when detrending light
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

        It can be useful to rescale errors if you believe they are under or overestimated, and a good place to start is looking at residual scatter.
        This is a simple alternative to using a jitter term in other fitting packages.
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


    def calc_gaia_err_scale(self):
        """Calculates the standard deviation of the residual of the best fit Gaia astrometry model and compares this to the median error on the Gaia
        AL data. Prints out the ratio of this standard deviation to the median errors. Prints out a new error scaling
        value to put into init_gaia, accounting for any error scaling already in effect.

        It can be useful to rescale errors if you believe they are under or overestimated, and a good place to start is looking at residual scatter.
        This is a simple alternative to using a jitter term in other fitting packages.
        """

        y = self.res.copy()

        if self.use_priors:

            y = y | self.fixed

        gm = self._gaiam['gm'] + self._gaiam['bkg']

        gal = self.gal.copy()

        resstd = np.std(gal - gm)

        mederr = np.median(self.galerr)

        print('resid. std', resstd, ', median err', mederr, ', resid. std / median err', resstd / mederr, ', new err scale', resstd/mederr*self.gaia_err_scale)


    def calc_autocorr(self):
        """Prints the integrated autocorrelation time (ACT) for each parameter in the MCMC fit, and how it compares to the chain lengths. Generally, if
        the chain length is 50 times the ACT for all parameters, then the MCMC has converged. This is also checked during fitting to stop the MCMC once
        it is converged. This function is for after the fit is finished.
        """

        print('\nIntegrated Autorcorrelation Times:')

        if self.sampler_type == 'emcee':

            taus = emcee.autocorr.integrated_time(self.samples, quiet=True)

        elif self.sampler_type == 'zeus':

            taus = zeus.AutoCorrTime(self.samples)

        n = len(self.samples)

        tab = Table({'Parameter': self.keys, 'Autocorr Time': taus, 'Len/Autocorr Time': n/taus})
        tab.pprint_all()


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

        y = self.res.copy()

        if self.use_priors:

            y = y | self.fixed

        rv = self.rv.copy()
        for i in np.unique(self.which_rv):

            j = np.where(self.which_rv == i)[0]

            if i > 0:

                rv[j] -= np.median(y['rv_offset {0}'.format(self.rvnames[i])])

        if use_residual:
            rvres = rv - np.sum(self._rvm['rvm'], axis = 0) - self._rvm['bkg']

        else:
            rvres = rv

        ls = LombScargle(self.tr, rvres, dy = self.rverr)

        if freq is not None:

            power = ls.power(freq)

        else:
             
            freq, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq)

        lsw = LombScargle(self.tr, np.ones_like(self.tr), fit_mean = False, center_data = False)
        powerw = lsw.power(freq)

        fig, ax = plt.subplots()

        if plot_periods:
            ax.plot(1/freq, power, c = 'black', zorder = 2)
            ax.plot(1/freq, powerw, c = 'gray', label = 'Window Fn.', zorder = 1, alpha = 0.5)
            ax.set_xscale('log')
            ax.set_xlabel('Period (days)', fontsize = 20)

        else:
            ax.plot(freq, power, c = 'black', zorder = 2)
            ax.plot(freq, powerw, c = 'gray', label = 'Window Fn.', zorder = 1, alpha = 0.5)
            ax.set_xlabel('Frequency (1/days)', fontsize = 20)

        ax.axhline(ls.false_alarm_level(0.01), c = 'red', label = '1% FA')
        ax.axhline(ls.false_alarm_level(0.05), c = 'orange', label = '5% FA')
        ax.legend(fontsize = 15)
        ax.tick_params(axis = 'both', labelsize = 15)
        plt.show()

        return freq, power, ls



class MiniExoSystem:
    """A class to store the data and options from an ExoSystem object used in the log likelihood function for fitting, without all of the methods.
    Critically, this does not contain self.sampler, allowing it to be used in a multiprocessing Pool.
    """

    def __init__(self, exs: ExoSystem):
        """Initializes the MiniExoSystem from an ExoSystem object. Copies the attributes from the ExoSystem which are necessary for fitting.
        """

        attrs = ('use_priors','fixed','n','fit_planets','fit_transit','is_transit','fit_ecc','fit_ld','fit_star','starmod','misti','fit_ttv',
                 'is_rv','fit_rv','is_astro','fit_gaia','ttvi','allpriors','lin_names','tt','lcnames','ld','filters','ldgrids','ttvsectors',
                 'exptimes','supersamples','is_eclipse','ms','rs','detrend','f','ferr','gps','tr','tr_ref','rv_bkg_order','which_rv','rvnames',
                 'rv','rverr','tg','tg_ref','gal','galerr','psi','plxf')

        for att in attrs:
            val = getattr(exs, att, None)
            if val is not None:
                setattr(self, att, val)



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


def lightCurve(par: np.typing.NDArray, t: np.typing.ArrayLike, ld: list, expt: float, ss: int, eclipse = False, fp: float = 0, rstar: float = 0, q: float = 0) -> np.typing.ArrayLike:
        """Generates a model transit light curve using batman.

        Args:
            par (ndarray): Array of transit parameters. Parameters must be specifically in this order:
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
            eclipse (bool): Whether or not to include a secondary eclipse in the light curve. The default is False.
            fp (float): The planet to star flux ratio, for the secondary eclipse. The default is 0.
            rstar (float): The stellar radius in solar radii. Used with a/Rs to get the secondary eclipse center time. The default is 0.
            q (float): The planet-to-star mass ratio. Used to get the secondary eclipse center time. The default is 0.

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

            tc2 = calc_t_sec(par[0], par[1], par[5], par[6]*np.pi/180, par[3], par[4]*np.pi/180, rstar, q)

            params.fp = fp
            params.t_secondary = tc2

            m2 = batman.TransitModel(params, t, supersample_factor = ss, exp_time = expt, transittype = 'secondary')

            flux += m2.light_curve(params) - 1 - fp

        return flux


def calc_t_sec(p: np.typing.ArrayLike, tc: np.typing.ArrayLike, e: np.typing.ArrayLike, w: np.typing.ArrayLike, a: np.typing.ArrayLike, i: np.typing.ArrayLike, r: np.typing.ArrayLike, q: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """Calculates the time of secondary eclipse center, including light travel delay.

    Args:
        p (ArrayLike): Orbital period in days.
        tc (ArrayLike): Time of conjunction in days.
        e (ArrayLike): Orbital eccentricity.
        w (ArrayLike): Argument of periastron in radians.
        a (ArrayLike): Semi-major axis to stellar radius ratio.
        i (ArrayLike): Orbital inclination in radians.
        r (ArrayLike): Stellar radius in solar radii.
        q (ArrayLike): Mass ratio, planet mass/stellar mass.

    Returns:
        ArrayLike: The time of secondary eclipse center.
    """

    f1 = np.pi/2 - w
    E1 = np.arctan2(np.sqrt(1-e**2)*np.sin(f1), e + np.cos(f1))
    M1 = E1 - e*np.sin(E1)

    f2 = 3*np.pi/2 - w
    E2 = np.arctan2(np.sqrt(1-e**2)*np.sin(f2), e + np.cos(f2))
    M2 = E2 - e*np.sin(E2)

    light_delay = (2 / constants.c * 1/(q+1) * a*r*u.Rsun * np.sin(i) * (1-e**2) / (1-(e*np.sin(w))**2)).to(u.day).value

    tc2 = p/(2*np.pi)*(M2-M1) + tc + light_delay

    return tc2


def rvModel(par: np.typing.NDArray, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """Generates model RVs using _rvModel. This is compiled with cython to run faster for large amounts of data.

    Args:
        par (Array): Array of RV parameters. Parameters must be specifically in this order:
            - Orbital period in same units as t.
            - Time of conjunction in same units as t. This is defined as the time when the true anomaly is pi/2 - w.
            - RV semi-amplitude. Returns model RV points in same units as this.
            - Orbital eccentricity.
            - Argument of periastron in radians.

        t (ArrayLike): Times at which to calculate the RV model.

    Returns:
        ArrayLike: The model RVs at each of the input times, centered at 0.
    """

    par = np.array(par, dtype = float)
    t = np.array(t, dtype = float)

    rv = _rvModel(par, t)

    return rv


def Gaia_AL_Model(par: np.typing.NDArray, t: np.typing.ArrayLike, psi: np.typing.ArrayLike) -> tuple[np.typing.ArrayLike,np.typing.ArrayLike,np.typing.ArrayLike]:
    """Generates model Gaia astrometry due to an orbit (not including proper motion), given the times and Gaia scan angles. This returns the
    ra*cos(dec) offsets, the dec offsets, and the along scan offsets. This is compiled with cython to run faster for large amounts of data.

    Args:
        par (Array): Array of astrometric parameters. Parameters must be specifically in this order:
            - Orbital period in same units as t.
            - Time of conjunction in same units as t. This is defined as the time when the true anomaly is pi/2 - w.
            - Orbital eccentricity.
            - Argument of periastron in radians.
            - Orbital inclination in radians.
            - Longitude of the ascending node in radians. The angle is defined Eastward of North, or from postive Dec towards positive RA.
            - The planet mass, in units of solar masses (for ease of calculation).
            - The stellar mass, in units of solar masses.
            - Parallax, in mas.

        t (ArrayLike): Times at which to calculate the Gaia AL offset model.
        psi (ArrayLike): Gaia scan angles, in radians, at each time in t.

    Returns:
        tuple[ArrayLike, ArrayLike, ArrayLike]: The model astrometric offsets, in mas, due to the orbit. The first item is the ra*cos(dec) offsets,
            the second is the dec offsets, and the third is the Gaia along scan offsets.
    """

    par = np.array(par, dtype = float)
    t = np.array(t, dtype = float)
    psi = np.array(psi, dtype = float)

    x, y, eta = _astrometryModel(par, t, psi)

    return x, y, eta


def get_transit_params(par: dict, i: int, shape: tuple = ()) -> np.typing.NDArray:
    """Generates the specific list of transit parameters needed for the lightCurve function from the parameter dict used in fitting.

    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        shape (tuple, optional): The shape of each parameter. Default is (), which is a scalar.
    
    Returns:
        ndarray: Array of transit parameters that can be input directly into lightCurve.
    """

    p = np.exp(par['log(P) {0}'.format(i)])
    tc = par['Tc {0}'.format(i)]
    ror = par['ror {0}'.format(i)]
    inc = np.arccos(par['cos(i) {0}'.format(i)]) * 180/np.pi

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) * 180/np.pi if 'secw {0}'.format(i) in par else np.full(shape, 90)

    if 'log(a/rs) {0}'.format(i) in par:
        ar = np.exp(par['log(a/rs) {0}'.format(i)])

    elif 'log(q) {0}'.format(i) in par:
        ms = (np.exp(par['log(rhostar)'])*u.g/u.cm**3 * 4/3*np.pi * (par['rstar']*u.Rsun)**3).to(u.Msun).value
        ar = ((ms*(1+np.exp(par['log(q) {0}'.format(i)])) * (p*u.day).to(u.yr).value**2)**(1/3)*u.AU).to(u.Rsun).value / par['rstar']
    
    elif 'log(K) {0}'.format(i) in par:
        ms = (np.exp(par['log(rhostar)'])*u.g/u.cm**3 * 4/3*np.pi * (par['rstar']*u.Rsun)**3).to(u.Msun).value
        mp = (calc_m_from_k(p, np.exp(par['log(K) {0}'.format(i)]), e, inc, ms)*u.earthMass).to(u.Msun).value
        ar = (((ms+mp) * (p*u.day).to(u.yr).value**2)**(1/3)*u.AU).to(u.Rsun).value / par['rstar']

    else:
        ar = (np.exp(par['log(rhostar)']) * p**2 / 0.018916375)**(1/3)

    return np.array([p, tc, ror, ar, inc, e, w])


def get_ttv_params(par: dict, i: int, ttvi: np.typing.NDArray[np.int64], shape: tuple = ()) -> np.typing.NDArray:
    """Generates the specific list of transit parameters needed for the lightCurve function from the parameter dict used in fitting, in the case
    when the planet is being fit for TTVs. Calculates the period and time of conjunction using linear regression on the transit times.

    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        ttvi (ndarray): Array of integers representing the number of periods that have occurred between the first observed transit and each
            observed transit that is being fit.
        shape (tuple, optional): The shape of each parameter. Default is (), which is a scalar.
        
    Returns:
        ndarray: Array of transit parameters that can be input directly into lightCurve.
    """

    ror = par['ror {0}'.format(i)]
    inc = np.arccos(par['cos(i) {0}'.format(i)]) * 180/np.pi

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) * 180/np.pi if 'secw {0}'.format(i) in par else np.full(shape, 90)

    tts = [par['TT {0} {1}'.format(i, j+1)] for j in range(len(ttvi))]

    res = linregress(ttvi, tts)
    p = res.slope
    tc = res.intercept

    if 'log(a/rs) {0}'.format(i) in par:
        ar = np.exp(par['log(a/rs) {0}'.format(i)])

    elif 'log(q) {0}'.format(i) in par:
        ms = (np.exp(par['log(rhostar)'])*u.g/u.cm**3 * 4/3*np.pi * (par['rstar']*u.Rsun)**3).to(u.Msun).value
        ar = ((ms*(1+np.exp(par['log(q) {0}'.format(i)])) * (p*u.day).to(u.yr).value**2)**(1/3)*u.AU).to(u.Rsun).value / par['rstar']

    elif 'log(K) {0}'.format(i) in par:
        ms = (np.exp(par['log(rhostar)'])*u.g/u.cm**3 * 4/3*np.pi * (par['rstar']*u.Rsun)**3).to(u.Msun).value
        mp = (calc_m_from_k(p, np.exp(par['log(K) {0}'.format(i)]), e, inc, ms)*u.earthMass).to(u.Msun).value
        ar = (((ms+mp) * (p*u.day).to(u.yr).value**2)**(1/3)*u.AU).to(u.Rsun).value / par['rstar']

    else:
        ar = (np.exp(par['log(rhostar)']) * p**2 / 0.018916375)**(1/3)

    return np.array([p, tc, ror, ar, inc, e, w])


def get_rv_params(par: dict, i: int, shape: tuple = ()) -> np.typing.NDArray:
    """Generates the specific list of RV parameters needed for the rvModel function from the parameter dict used in fitting.

    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        shape (tuple, optional): The shape of each parameter. Default is (), which is a scalar.
    
    Returns:
        ndarray: Array of RV parameters that can be input directly into rvModel.
    """

    p = np.exp(par['log(P) {0}'.format(i)])
    tc = par['Tc {0}'.format(i)]

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) if 'secw {0}'.format(i) in par else np.full(shape, np.pi/2)

    if 'log(K) {0}'.format(i) in par:
        k = np.exp(par['log(K) {0}'.format(i)])

    else:

        if 'sico {0}'.format(i) in par:
            inc = np.arccos(1-2*(par['sico {0}'.format(i)]**2 + par['siso {0}'.format(i)]**2))
        
        else:
            inc = np.arccos(par['cos(i) {0}'.format(i)])

        if ['mstar'] in par:
            ms = par['mstar']

        else:
            ms = (np.exp(par['log(rhostar)'])*u.g/u.cm**3 * 4/3*np.pi * (par['rstar']*u.Rsun)**3).to(u.Msun).value

        q = np.exp(par['log(q) {0}'.format(i)])
        k = 29785.326 * np.sin(inc) / np.sqrt(1-e**2) * (p*u.day).to(u.yr).value**(-1/3) * q/(1+q)**(2/3) * ms**(1/3)

    return np.array([p, tc, k, e, w])


def get_Gaia_params(par: dict, i: int, shape: tuple = ()):
    """Generates the specific list of gaia astrometric parameters needed for the Gaia_AL_Model function from the parameter dict used in fitting.
    
    Args:
        par (dict): Dictionary of fit parameters.
        i (int): Planet number.
        shape (tuple, optional): The shape of each parameter. Default is (), which is a scalar.
    
    Returns:
        ndarray: Array of Gaia astrometric parameters that can be input directly into Gaia_AL_Model.
    """

    p = np.exp(par['log(P) {0}'.format(i)])
    tc = par['Tc {0}'.format(i)]

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) if 'secw {0}'.format(i) in par else np.full(shape, np.pi/2)

    if 'sico {0}'.format(i) in par:

        inc = np.arccos(1-2*(par['sico {0}'.format(i)]**2 + par['siso {0}'.format(i)]**2))
        omega = np.arctan2(par['siso {0}'.format(i)], par['sico {0}'.format(i)])

    else:

        inc = np.arccos(par['cos(i) {0}'.format(i)])
        omega = np.arctan2(par['somega {0}'.format(i)], par['comega {0}'.format(i)])

    if ['mstar'] in par:
        ms = par['mstar']

    else:
        ms = (np.exp(par['log(rhostar)'])*u.g/u.cm**3 * 4/3*np.pi * (par['rstar']*u.Rsun)**3).to(u.Msun).value

    mp = np.exp(par['log(q) {0}'.format(i)]) * ms

    plx = par['plx']

    return [p, tc, e, w, inc, omega, mp, ms, plx]


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


def log_like_zeus(par: np.typing.NDArray, keys: np.typing.NDArray) -> float:
    """Wrapper function on log_like used for zeus sampler. Needed to turn the parameter array back into a dict.

    Args:
        par (ndarray): The fit parameter array.
        keys (ndarray): The names of each parameter, used to turn the array back into a dict.

    Returns:
        float: The log likelihood value from the log_like function.
    """

    pardict = dict(zip(keys, par))

    return log_like(pardict)


def check_emcee_conv(sampler: emcee.EnsembleSampler, it: int, last_tau: float, burnin: int, ntau: int = 50, deltatau: float = 0.01) -> tuple[bool, float]:
    """Checks whether or not the chains have converged, for the emcee sampler. The condition is that the length of the chain (with burn-in removed)
    must be greater than or equal to the maximum integrated autocorrelation time of the chains times ntau. Additionally, the fractional change in
    the maximum autocorrelation time from the last check must be less than deltatau.

    Args:
        sampler (emcee.EnsembleSampler): The emcee EnsembleSampler object.
        it (int): The current step number.
        last_tau (float): The previous maximum integrated autocorrelation time of the chains, from the last check.
        burnin (int): Number of burn-in steps to remove.
        ntau (int, optional): For convergence, the chain length must reach ntau times the maximum integrated autocorrelation time of the chains.
            Default is 50.
        deltatau (float, optional): For convergence, the rate of change of the maximum integrated autocorrelation time of the chains must drop below
            deltatau. Default is 0.01.

    Returns:
        tuple[bool, float]: Returns whether or not the chains have converged, as well as the maximum integrated autocorrelation time of the chains (to
            be passed into the next check as last_tau).
    """

    cleanlen = it-burnin

    clean_chain = sampler.get_chain(discard = burnin)

    try:
        tau = np.max(emcee.autocorr.integrated_time(clean_chain, quiet=True))
    except Exception:
        return False, last_tau

    conv = False

    if last_tau is not None:

        dtau = np.abs(tau-last_tau)/last_tau

        conv = cleanlen >= ntau*tau and dtau < deltatau

    return conv, tau


def check_zeus_conv(sampler: zeus.EnsembleSampler, it: int, last_tau: float, burnin: int, ntau: int = 50, deltatau: float = 0.01) -> tuple[bool, float]:
    """Checks whether or not the chains have converged, for the zeus sampler. The condition is that the length of the chain (with burn-in removed)
    must be greater than or equal to the maximum integrated autocorrelation time of the chains times ntau. Additionally, the fractional change in
    the maximum autocorrelation time from the last check must be less than deltatau.

    Args:
        sampler (zeus.EnsembleSampler): The zeus EnsembleSampler object.
        it (int): The current step number.
        last_tau (float): The previous maximum integrated autocorrelation time of the chains, from the last check.
        burnin (int): Number of burn-in steps to remove.
        ntau (int, optional): For convergence, the chain length must reach ntau times the maximum integrated autocorrelation time of the chains.
            Default is 50.
        deltatau (float, optional): For convergence, the rate of change of the maximum integrated autocorrelation time of the chains must drop below
            deltatau. Default is 0.01.

    Returns:
        tuple[bool, float]: Returns whether or not the chains have converged, as well as the maximum integrated autocorrelation time of the chains (to
            be passed into the next check as last_tau).
    """

    cleanlen = it-burnin

    clean_chain = sampler.get_chain(discard = burnin)

    try:
        tau = np.max(zeus.AutoCorrTime(clean_chain))
    except Exception:
        return False, last_tau

    conv = False

    if last_tau is not None:

        dtau = np.abs(tau-last_tau)/last_tau

        conv = cleanlen >= ntau*tau and dtau < deltatau

    return conv, tau


def log_like(par_in: dict) -> tuple[float,np.typing.NDArray]:
    """The log likelihood function for fitting.

    Checks certain parameters to make sure they are in bounds. Calculates likelihoods from any priors. Calculates log likelihood for stellar
    fitting using the isochrones MIST star model. Generates transit and RV models, then calculates log likelihood relative to data using lnNorm.

    Args:
        par_in (dict): Dictionary of fit parameters.

    Returns:
        float, ndrray: The log likelihood value is returned. If any parameter is out of bounds or any portion of the log likelihood
            is nan, returns negative infinity. Additionally, an array of the linear parameters (solved for using least squares) is returned.
    """

    par = par_in | miniexs.fixed

    for i in range(miniexs.n):

        if 'e {0}'.format(i+1) in par:

            par['secw {0}'.format(i+1)] = np.sqrt(par['e {0}'.format(i+1)]) * np.cos(par['w {0}'.format(i+1)])
            par['sesw {0}'.format(i+1)] = np.sqrt(par['e {0}'.format(i+1)]) * np.sin(par['w {0}'.format(i+1)])
 
    nlin = len(miniexs.lin_names)

    like = 0

    if miniexs.fit_star:

        starlike = miniexs.starmod.lnlike([par['eep'],par['log10(age)'],par['feh'],1000/par['plx'],par['AV']])
        starlike += miniexs.starmod.lnprior([par['eep'],par['log10(age)'],par['feh'],1000/par['plx'],par['AV']])

        if np.isnan(starlike):
            return -np.inf, np.zeros((nlin))
        
        like += starlike

        rstar, mstar, Tstar, loggstar, rhostar = miniexs.misti.interp_value([par['eep'],par['log10(age)'],par['feh']],['radius','mass','Teff','logg','density'])

        par['mstar'] = mstar
        par['rstar'] = rstar
        par['log(rhostar)'] = np.log(rhostar)

        if miniexs.fit_planets:

            if not 2300 <= Tstar <= 7800 or not 3 <= loggstar <= 6:
                return -np.inf, np.zeros((nlin))

            for i in range(miniexs.n):

                if not miniexs.is_transit[i]:
                    continue

                if miniexs.fit_ttv[i]:

                    p = get_ttv_params(par, i+1, miniexs.ttvi['{0}'.format(i+1)])[0]

                else:

                    p = np.exp(par['log(P) {0}'.format(i+1)])

                if miniexs.is_astro[i] and miniexs.fit_gaia:

                    q = np.exp(par['log(q) {0}'.format(i+1)])
                    mp = q * mstar

                elif miniexs.is_rv[i] and miniexs.fit_rv:

                    e = 0
                    if miniexs.fit_ecc[i]:
                        e = par['secw {0}'.format(i+1)]**2 + par['sesw {0}'.format(i+1)]**2

                    mp = (calc_m_from_k(p, np.exp(par['log(K) {0}'.format(i+1)]), e, np.arccos(par['cos(i) {0}'.format(i+1)]), mstar)*u.earthMass).to(u.Msun).value

                else:

                    mp = 0

                ar = (((mstar + mp) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                par['log(a/rs) {0}'.format(i+1)] = np.log(ar)

        
    lin_par = np.empty(0)
            
    tpars = []
    rpars = []
    gpars = []
    
    if miniexs.fit_planets:

        for i in range(miniexs.n):
            
            if miniexs.is_transit[i] and miniexs.fit_transit:

                if miniexs.fit_ttv[i]:

                    pars = get_ttv_params(par, i+1, miniexs.ttvi['{0}'.format(i+1)])
                    tpars.append(pars)

                    lin_par = np.append(lin_par, pars[:2])

                    par['log(P) {0}'.format(i+1)] = np.log(pars[0])
                    par['Tc {0}'.format(i+1)] = pars[1]

                else:

                    tpars.append(get_transit_params(par, i+1))

            if miniexs.is_rv[i] and miniexs.fit_rv:

                rpars.append(get_rv_params(par, i+1))

            if miniexs.is_astro[i] and miniexs.fit_gaia:

                gpars.append(get_Gaia_params(par, i+1))
            

    

    priorlike = miniexs.allpriors.apply(par)

    if np.isinf(priorlike):
        return -np.inf, np.zeros((nlin))
    
    else:
        like += priorlike


    if miniexs.fit_transit:

        for i in range(len(miniexs.tt)):

            fm = 0

            ld = miniexs.ld[miniexs.filters[i]]

            if miniexs.fit_ld:

                ld = [par['u1 {0}'.format(miniexs.filters[i])], par['u2 {0}'.format(miniexs.filters[i])]]

            if miniexs.fit_star:

                ld = [miniexs.ldgrids[miniexs.filters[i]][0]([Tstar, loggstar, par['feh']])[0], miniexs.ldgrids[miniexs.filters[i]][1]([Tstar, loggstar, par['feh']])[0]]

            for j in range(miniexs.n):

                if not miniexs.is_transit[j]:
                    continue

                k = np.sum(miniexs.is_transit[:j])

                if miniexs.fit_ttv[j]:

                    p = tpars[k][0]

                    for z in range(len(miniexs.ttvi['{0}'.format(j+1)])):

                        if miniexs.ttvsectors['{0} {1}'.format(j+1, z+1)] != i:
                            continue

                        ttime = par['TT {0} {1}'.format(j+1, z+1)]

                        tpars[k][1] = ttime

                        ind = np.where((miniexs.tt[i] >= ttime - p/4) & (miniexs.tt[i] <= ttime + p/4))

                        fm0 = np.zeros(len(miniexs.tt[i]))

                        fm0[ind] += lightCurve(tpars[k], miniexs.tt[i][ind], ld, miniexs.exptimes[i], miniexs.supersamples[i])

                        fm += fm0

                else:

                    if miniexs.is_eclipse[j]:

                        ms = par['mstar'] if 'mstar' in par else miniexs.ms
                        rs = par['rstar'] if 'rstar' in par else miniexs.rs
                        q = 0

                        if miniexs.is_astro[j] and miniexs.fit_gaia:

                            q = np.exp(par['log(q) {0}'.format(j+1)])

                        elif miniexs.is_rv[j] and miniexs.fit_rv:

                            kr = np.sum(miniexs.is_rv[:j])
                            
                            mp = (calc_m_from_k(tpars[k][0], rpars[kr][2], rpars[kr][3], tpars[k][4]*np.pi/180, ms)*u.earthMass).to(u.Msun).value

                            q = mp/ms

                        fm += lightCurve(tpars[k], miniexs.tt[i], ld, miniexs.exptimes[i], miniexs.supersamples[i], eclipse = True, fp = par['fp {0}'.format(j+1)], rstar = rs, q = q)

                    else:

                        fm += lightCurve(tpars[k], miniexs.tt[i], ld, miniexs.exptimes[i], miniexs.supersamples[i])

            if miniexs.detrend[i]:

                resid = miniexs.f[i] - fm
                
                ii = np.sum(miniexs.detrend[:i])

                try:
                    gp = set_gp_params(np.exp(par['log(rho_gp) {0}'.format(miniexs.lcnames[i])]), np.exp(par['log(sigma_gp) {0}'.format(miniexs.lcnames[i])]), miniexs.tt[i], miniexs.ferr[i], miniexs.gps[ii])

                except:
                    return -np.inf, np.zeros((nlin))

                if 'F0 {0}'.format(miniexs.lcnames[i]) in miniexs.fixed:

                    resid -= miniexs.fixed['F0 {0}'.format(miniexs.lcnames[i])]

                else:

                    M = np.ones((len(miniexs.tt[i]), 1))

                    invKM = gp.apply_inverse(M)
                    invKr = gp.apply_inverse(resid)

                    MTinvKM = M.T @ invKM
                    MTinvKr = M.T @ invKr

                    X = np.linalg.solve(MTinvKM, MTinvKr)
                    resid -= M @ X

                    _, logdet = np.linalg.slogdet(MTinvKM)
                    like += -0.5 * logdet

                    lin_par = np.append(lin_par, X)

                like += gp.log_likelihood(resid)

            else:

                if 'F0 {0}'.format(miniexs.lcnames[i]) in miniexs.fixed:

                    fm += miniexs.fixed['F0 {0}'.format(miniexs.lcnames[i])]

                    like += np.sum(lnNorm(miniexs.f[i], fm, miniexs.ferr[i]))

                else:

                    resid = miniexs.f[i] - fm

                    M = np.ones((len(miniexs.tt[i]), 1))

                    W = 1/miniexs.ferr[i]**2
                    Mw = M * np.sqrt(W)[:, None]
                    Yw = resid * np.sqrt(W)

                    X, chi2, _, _ = np.linalg.lstsq(Mw, Yw, rcond = None)
        
                    if len(chi2) > 0:
                        chi2min = chi2[0]
                    else:
                        chi2min = np.sum((Yw - Mw @ X)**2)

                    _, logdet = np.linalg.slogdet(Mw.T @ Mw)

                    like += -0.5 * (chi2min + logdet)

                    lin_par = np.append(lin_par, X)


    if miniexs.fit_rv:

        rvm = 0

        for i in range(miniexs.n):

            if not miniexs.is_rv[i]:
                continue

            j = np.sum(miniexs.is_rv[:i])

            rvm += rvModel(rpars[j], miniexs.tr)

        rvclean = miniexs.rv - rvm

        M = []

        if 'gamma' in miniexs.fixed:
            rvclean -= miniexs.fixed['gamma']

        else:
            M.append(np.ones((len(miniexs.tr))))

        if miniexs.rv_bkg_order > 0:

            if 'gamma_dot' in miniexs.fixed:
                rvclean -= miniexs.fixed['gamma_dot'] * (miniexs.tr - miniexs.tr_ref)

            else:
                M.append(miniexs.tr - miniexs.tr_ref)

        if miniexs.rv_bkg_order > 1:

            if 'gamma_ddot' in miniexs.fixed:
                rvclean -= miniexs.fixed['gamma_ddot'] * (miniexs.tr - miniexs.tr_ref)**2

            else:
                M.append((miniexs.tr - miniexs.tr_ref)**2)

        for i in np.unique(miniexs.which_rv)[1:]:

            if 'rv_offset {0}'.format(miniexs.rvnames[i]) in miniexs.fixed:

                j = np.where(miniexs.which_rv == i)[0]

                rvclean[j] -= miniexs.fixed['rv_offset {0}'.format(miniexs.rvnames[i])]

            else:
                M.append((miniexs.which_rv == i).astype(float))

        if len(M) > 0:

            M = np.column_stack(M)

            W = 1/miniexs.rverr**2
            Mw = M * np.sqrt(W)[:, None]
            Yw = rvclean * np.sqrt(W)

            X, chi2, _, _ = np.linalg.lstsq(Mw, Yw, rcond = None)

            if len(chi2) > 0:
                chi2min = chi2[0]
            else:
                chi2min = np.sum((Yw - Mw @ X)**2)

            _, logdet = np.linalg.slogdet(Mw.T @ Mw)

            like += -0.5 * (chi2min + logdet)

            lin_par = np.append(lin_par, X)

        else:

            like += np.sum(lnNorm(rvclean, np.zeros(rvclean.shape), miniexs.rverr))


    if miniexs.fit_gaia:

        galm = miniexs.plxf * par['plx']

        for i in range(miniexs.n):

            if not miniexs.is_astro[i]:
                continue

            j = np.sum(miniexs.is_astro[:i])

            galm += Gaia_AL_Model(gpars[j], miniexs.tg, miniexs.psi)[2]

        gclean = miniexs.gal - galm
        tgyr = (miniexs.tg - miniexs.tgref)/365.25

        M = []

        if 'ra*_offset' in miniexs.fixed:
            gclean -= miniexs.fixed['ra*_offset']*np.sin(miniexs.psi)

        else:
            M.append(np.sin(miniexs.psi))

        if 'dec_offset' in miniexs.fixed:
            gclean -= miniexs.fixed['dec_offset']*np.cos(miniexs.psi)

        else:
            M.append(np.cos(miniexs.psi))

        if 'pmra*' in miniexs.fixed:
            gclean -= miniexs.fixed['pmra*']*tgyr*np.sin(miniexs.psi)

        else:
            M.append(tgyr*np.sin(miniexs.psi))

        if 'pmdec' in miniexs.fixed:
            gclean -= miniexs.fixed['pmdec']*tgyr*np.cos(miniexs.psi)

        else:
            M.append(tgyr*np.cos(miniexs.psi))

        if len(M) > 0:

            M = np.column_stack(M)
    
            W = 1/miniexs.galerr**2
            Mw = M * np.sqrt(W)[:, None]
            Yw = gclean * np.sqrt(W)
        
            X, chi2, _, _ = np.linalg.lstsq(Mw, Yw, rcond = None)
        
            if len(chi2) > 0:
                chi2min = chi2[0]
            else:
                chi2min = np.sum((Yw - Mw @ X)**2)

            _, logdet = np.linalg.slogdet(Mw.T @ Mw)

            like += -0.5 * (chi2min + logdet)

            lin_par = np.append(lin_par, X)

        else:

            like += np.sum(lnNorm(gclean, np.zeros(gclean.shape), miniexs.galerr))


    linpriorlike = miniexs.allpriors.apply({miniexs.lin_names[i]: lin_par[i] for i in range(len(lin_par))})

    if np.isinf(linpriorlike):
        return -np.inf, np.zeros((nlin))
    
    else:
        like += linpriorlike


    return (like, lin_par) if not np.isnan(like) else (-np.inf, np.zeros((nlin)))


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

    starlike = exs.starmod.lnlike([par['eep'],par['log10(age)'],par['feh'],1000/par['plx'],par['AV']])
    starlike += exs.starmod.lnprior([par['eep'],par['log10(age)'],par['feh'],1000/par['plx'],par['AV']])

    if exs.use_priors:

        rstar, mstar, Tstar, loggstar, rhostar = miniexs.misti.interp_value([par['eep'],par['log10(age)'],par['feh']],['radius','mass','Teff','logg','density'])

        par['mstar'] = mstar
        par['rstar'] = rstar
        par['log(rhostar)'] = np.log(rhostar)

        starlike += exs.allpriors.apply(par)

    if np.isnan(starlike):
        return -np.inf
    
    return starlike