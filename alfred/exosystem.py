import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import emcee
import corner
import batman
from scipy.stats import linregress
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy import constants
from astropy.timeseries import LombScargle
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

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.use('TKAgg')
np.set_printoptions(legacy='1.25')


class ExoSystem:


    def __init__(self, direc: str, init_planets = 'init_planets.txt', init_star = 'init_star.txt', init_lcs = 'init_lcs.txt', init_rv = 'init_rv.txt', init_ld = 'init_ld.txt', init_priors = 'init_priors.txt', init_ttvs = 'init_ttvs.txt'):

        self.direc = direc
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

        self.magwls = [1.24,1.66,2.16,0.584,0.502,0.759,3.35,4.6,11.6]
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

            if name[-5:] == '.fits':

                with fits.open(self.direc+name) as hdul:

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

                dat = np.load(self.direc+name)

                tt0 = dat[tab_lcs['Time Col'][i]]
                f0 = dat[tab_lcs['Flux Col'][i]]
                ferr0 = dat[tab_lcs['Err Col'][i]]


            elif name[-4:] == '.dat':

                dat = Table.read(self.direc+name, format = 'ascii.no_header')

                tt0 = dat['col1']
                f0 = dat['col2']
                ferr0 = dat['col3']


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
            
                rvdata = Table.read(self.direc + name)
                
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

            self.tr_ref = (np.max(self.tr) + np.min(self.tr))/2
            self.tr_plot = np.linspace(np.min(self.tr)-5, np.max(self.tr+5), 1000)
            self.tr_phase = np.linspace(-0.5, 0.5, 1000)


    def fit(self, name, nburn, nrun, nwalk = 0, fit_transit = True, fit_rv = True, fit_star = False, fit_ld = False, use_priors = False,
            rv_bkg_order = 0, star_run = None, save_samples = False, sigma_clip = 5, lc_supersample_size = 600, show_plots = True, order_a = False):

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


        if not fit_transit and not fit_rv and not fit_star:
            print('You need to fit something!')
            return
        
        if not fit_transit and fit_rv and fit_star:
            print('To fit stellar parameters simultaneously with planet parameters, please use transit data (set fit_transit to True).')
            return
        
        if not os.path.isdir(self.direc+'Plots/'+name):
            os.mkdir(self.direc+'Plots/'+name)
        
        self.use_priors = use_priors
        if self.init_priors is None and use_priors:
            print('No init_priors file found. Check the name in the system initialization, or run Init_priors().create(). Running with no priors.')
            self.use_priors = False

        if not fit_transit and not fit_rv and fit_star:
            self.fit_star_only(name)
            return
        

        self.supersamples = np.array([max(1,int(x/lc_supersample_size)) for x in self.exptimes*24*60*60])


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
                self.starmod = apply_star_priors(self.init_priors.table, self.starmod)

            self.ldgrids = {}

            for filt in np.unique(self.filters):                    

                interpu1, interpu2 = load_ld_grid(filt)

                self.ldgrids[filt] = [interpu1, interpu2]



        if np.any(self.fit_ttv) and self.fit_transit:
            self.initialize_ttvs(name = self.init_ttvs_name)


        self.x0 = {}

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

            elif self.is_rv[i] and self.fit_rv:

                self.x0['log(P) {0}'.format(i+1)] = np.log(self.p[i])
                self.x0['Tc {0}'.format(i+1)] = self.tc[i]

            if self.is_rv[i] and self.fit_rv:

                j = np.sum(self.is_rv[:i])

                self.x0['log(K) {0}'.format(i+1)] = np.log(self.k[j])

            if self.fit_ecc[i]:

                self.x0['secw {0}'.format(i+1)] = self.secosw[i]
                self.x0['sesw {0}'.format(i+1)] = self.sesinw[i]

        
        if self.fit_star:

            if star_run is not None:

                self.load_results(star_run)

                try:

                    self.x0['eep'] = np.median(self.res['eep'])
                    self.x0['age'] = np.median(self.res['age'])
                    self.x0['feh'] = np.median(self.res['feh'])
                    self.x0['distance'] = np.median(self.res['distance'])
                    self.x0['AV'] = np.median(self.res['AV'])

                except:

                    self.x0['eep'] = np.median(self.dres['eep'])
                    self.x0['age'] = np.median(self.dres['age'])
                    self.x0['feh'] = np.median(self.dres['feh'])
                    self.x0['distance'] = np.median(self.dres['distance'])
                    self.x0['AV'] = np.median(self.dres['AV'])

            else:

                self.starmod.fit(overwrite = True, basename = self.direc+'multinest chains/'+name+'-')

                self.x0['eep'] = np.median(self.starmod.derived_samples['eep'])
                self.x0['age'] = np.median(self.starmod.derived_samples['age'])
                self.x0['feh'] = np.median(self.starmod.derived_samples['feh'])
                self.x0['distance'] = np.median(self.starmod.derived_samples['distance'])
                self.x0['AV'] = np.median(self.starmod.derived_samples['AV'])


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
            
            self.x0['trend 0'] = 0
            if self.rv_bkg_order > 0:
                self.x0['trend 1'] = 0
            if self.rv_bkg_order > 1:
                self.x0['trend 2'] = 0

            for i in np.unique(self.which_rv)[1:]:
                self.x0['offset {0}'.format(self.rvnames[i])] = 0

        keys = list(self.x0.keys())

        print('Initial parameters:')
        print(self.x0)

        if self.use_priors:

            self.allpriors = AllPriors(self.init_priors.table, self.x0, self.fit_ttv)


        if self.nwalk < len(keys) * 2:
            self.nwalk = len(keys) * 2

        if self.fit_transit:

            self.tt = copy.deepcopy(self.tt_orig)
            self.f = copy.deepcopy(self.f_orig)
            self.ferr = copy.deepcopy(self.ferr_orig)

            self.masks = [np.ones(len(self.tt[i]), dtype = bool) for i in range(len(self.tt))]
        
        if self.fit_transit:
            print('\nStarting sigma clipping of lightcurves.')


        clipped = [0]*len(self.tt)

        for i in range(10):

            res = minimize(lambda x, *args: -1 * log_like({k:v for k,v in zip(keys, x)}, *args)[0], [self.x0[k] for k in keys], method = 'Nelder-Mead', args = (self,))
            x = {k:v for k,v in zip(keys, res.x)}

            if not self.fit_transit:
                break

            if self.fit_star:

                rstar, mstar, Tstar, loggstar = self.misti.interp_value([x['eep'],x['age'],x['feh']],['radius','mass','Teff','logg'])

                arlist = []
                for j in range(self.n):

                    if not self.is_transit[j]:
                        arlist.append(np.nan)
                        continue

                    if self.fit_ttv[j]:

                        p = get_ttv_params(x, j+1, self.ttvi['{0}'.format(j+1)], ar = 1)[0]

                    else:

                        p = np.exp(x['log(P) {0}'.format(j+1)])

                    if self.is_rv[j] and self.fit_rv:

                        e = 0
                        if self.fit_ecc[j]:
                            e = x['secw {0}'.format(j+1)]**2 + x['sesw {0}'.format(j+1)]**2

                        mp = calc_m_from_k(p, np.exp(x['log(K) {0}'.format(j+1)]), e, np.arccos(x['cos(i) {0}'.format(j+1)]), mstar)

                    else:

                        mp = 0

                    ar = (((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                    arlist.append(ar)

            pars = []
            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                if self.fit_ttv[j]:

                    pars.append(get_ttv_params(x, j+1, self.ttvi['{0}'.format(j+1)], ar = arlist[j] if self.fit_star else None))

                else:

                    pars.append(get_transit_params(x, j+1, ar = arlist[j] if self.fit_star else None))


            lastclipped = 0

            for j in range(len(self.tt)):

                mean = x['F0 {0}'.format(self.lcnames[j])]

                fm = mean

                ld = self.ld[self.filters[j]]

                if self.fit_ld:

                    ld = [x['u1 {0}'.format(self.filters[j])], x['u2 {0}'.format(self.filters[j])]]

                if self.fit_star:

                    ld = [self.ldgrids[self.filters[j]][0]([Tstar, loggstar, x['feh']])[0], self.ldgrids[self.filters[j]][1]([Tstar, loggstar, x['feh']])[0]]

                for k in range(self.n):

                    if not self.is_transit[k]:
                        continue

                    l = np.sum(self.is_transit[:k])

                    if self.fit_ttv[k]:

                        p = pars[l][0]

                        for z in range(len(self.ttvi['{0}'.format(k+1)])):

                            if self.ttvsectors['{0} {1}'.format(k+1, z+1)] != j:
                                continue

                            ttime = x['TT {0} {1}'.format(k+1, z+1)]

                            pars[l][1] = ttime

                            ind = np.where((self.tt[j] >= ttime - p/4) & (self.tt[j] <= ttime + p/4))

                            fm0 = np.zeros(len(self.tt[j]))

                            lc = lightCurve(pars[l], self.tt[j][ind], ld, self.exptimes[j], self.supersamples[j])

                            fm0[ind] += lc

                            fm += fm0

                    else:

                        fm += lightCurve(pars[l], self.tt[j], ld, self.exptimes[j], self.supersamples[j])

                resid = self.f[j] - fm

                gpf = 0

                if self.detrend[j]:

                    gp = set_gp_params(np.exp(x['log(rho_gp) {0}'.format(self.lcnames[j])]), np.exp(x['log(sigma_gp) {0}'.format(self.lcnames[j])]), self.tt[j], self.ferr[j], self.gps[j])

                    gpf = gp.predict(resid)

                    resid = self.f[j] - fm - gpf

                rms = np.sqrt(np.median(resid**2))

                mask = abs(resid) < self.sigma_clip * rms

                c = np.sum(~mask)
                lastclipped += c
                clipped[j] += c

                fig, ax = plt.subplots(3 if self.detrend[j] else 2, sharex = True)

                z = 0

                if self.detrend[j]:

                    z = 1

                    ax[0].scatter(self.tt[j], self.f[j], c = 'black', marker = '.', zorder = 1)
                    ax[0].plot(self.tt[j], gpf + mean, c = 'mediumseagreen', zorder = 2)

                ax[0+z].scatter(self.tt[j], self.f[j] - gpf, c = 'black', marker = '.', zorder = 1)
                ax[0+z].plot(self.tt[j], fm, c = 'mediumseagreen', zorder = 2)

                ax[1+z].scatter(self.tt[j][mask], (self.f[j] - gpf - fm)[mask], c = 'black', marker = '.', zorder = 1)
                ax[1+z].scatter(self.tt[j][~mask], (self.f[j] - gpf - fm)[~mask], c = 'red', marker = 'x', zorder = 2)
                ax[1+z].axhline(0, c = 'mediumseagreen', zorder = 3)

                ax[0].set_title('{0} Clipped {1}'.format(self.lcnames[j], c))

                plt.show()

                k = np.where(self.masks[j])[0]
                self.masks[j][k[~mask]] = False

                self.tt[j] = self.tt[j][mask]
                self.f[j] = self.f[j][mask]
                self.ferr[j] = self.ferr[j][mask]



            if lastclipped < 10:
                break

        if self.fit_transit:
            print('\nTotal points clipped:')
            print('All light curves: {0}'.format(np.sum(clipped)))
            for i in range(len(self.tt)):
                print('{0}: {1}'.format(self.lcnames[i], clipped[i]))

        self.x = x
        print('\nInitial parameters after optimization:')
        print(x)

        if self.fit_transit:
            pickle.dump(self.masks, open(self.direc+'Masks/'+name+'_masks.p', 'wb'))

        self.parnames = {}

        pos = []

        for i, k in enumerate(keys):

            self.parnames[k] = i

            if 'log(P)' in k:
                pos.append(np.log(np.random.normal(np.exp(x[k]), 0.0001, self.nwalk)))

            if 'Tc' in k:
                pos.append(np.random.normal(x[k], 0.0001, self.nwalk))

            if 'TT' in k:
                pos.append(np.random.normal(x[k], 0.0001, self.nwalk))

            if 'ror' in k:
                pos.append(np.random.normal(x[k], 0.1*x[k], self.nwalk))

            if 'log(a/rs)' in k:
                pos.append(np.log(np.random.normal(np.exp(x[k]), 0.1, self.nwalk)))

            if 'cos(i)' in k:
                pos.append(np.abs(np.random.normal(x[k], 0.001, self.nwalk)))

            if 'log(K)' in k:
                pos.append(np.log(np.random.normal(np.exp(x[k]), 0.01, self.nwalk)))

            if 'secw' in k or 'sesw' in k:
                z = np.random.normal(x[k], 0.01, self.nwalk)
                j = np.where(np.abs(z) > 1)[0]
                z[j] = 0.9 * z[j] / np.abs(z[j])
                pos.append(z)

            if 'F0' in k:
                pos.append(np.random.normal(x[k], 0.001, self.nwalk))

            if 'trend' in k:
                if '0' in k:
                    pos.append(np.random.normal(x[k], 0.1, self.nwalk))
                if '1' in k:
                    pos.append(np.random.normal(x[k], 0.01, self.nwalk))
                if '2' in k:
                    pos.append(np.random.normal(x[k], 0.001, self.nwalk))

            if '_gp' in k:
                pos.append(np.log(np.random.normal(np.exp(x[k]), 0.01*np.exp(x[k]), self.nwalk)))

            if 'offset' in k:
                pos.append(np.random.normal(x[k], 0.1, self.nwalk))

            if k in ['u1','u2']:
                z = np.abs(np.random.normal(x[k], 0.01, self.nwalk))
                j = np.where(z > 1)[0]
                z[j] = 0.9 * z[j]
                pos.append(z)

            if k == 'eep':
                pos.append(np.random.normal(x[k], 1, self.nwalk))

            if k == 'age':
                pos.append(np.random.normal(x[k], 0.05, self.nwalk))

            if k == 'feh':
                z = np.random.normal(x[k], 0.01, self.nwalk)
                j = np.where(np.abs(z) > 0.5)[0]
                z[j] = 0.45 * np.sign(z[j])
                pos.append(z)

            if k == 'distance':
                pos.append(np.random.normal(x[k], 1, self.nwalk))

            if k == 'AV':
                pos.append(np.abs(np.random.normal(x[k], 0.01, self.nwalk)))


        pos = np.transpose(np.array(pos))


        sampler = emcee.EnsembleSampler(nwalkers = self.nwalk, ndim = len(x), log_prob_fn = log_like, args = (self,), parameter_names = self.parnames, blobs_dtype = [('ps', np.ndarray), ('tcs', np.ndarray)])

        print('\nRunning MCMC burn-in.')
        #burn in
        state = sampler.run_mcmc(pos, self.nburn, progress = True)
        sampler.reset()

        print('\nRunning MCMC sampling.')
        #run
        sampler.run_mcmc(state, self.nrun, progress = True)

        # print(sampler.get_autocorr_time(quiet = True))

        self.samples = sampler.get_chain()

        self.log_likes_full = sampler.get_log_prob()


        if save_samples:

            pickle.dump({'parnames': self.parnames, 'samples': self.samples, 'log_like': self.log_likes_full}, open(self.direc+'Output/'+name+'_samples.p', 'wb'))

        self.calc_gelman_rubin()


        self.flat_samples = sampler.get_chain(flat = True, thin = 20)
        self.log_likes = sampler.get_log_prob(flat = True, thin = 20)

        if self.fit_transit and np.any(self.fit_ttv):

            blobs = sampler.get_blobs(flat = True, thin = 20)

            ps = np.array([x for x in blobs['ps']]).T
            tcs = np.array([x for x in blobs['tcs']]).T


        self.res = {}

        for k, v in self.parnames.items():

            self.res[k] = self.flat_samples[:,v]

        self.res['log_like'] = self.log_likes

        pickle.dump(self.res, open(self.direc+'Output/'+name+'_res.p', 'wb'))

        self.dres = {}

        n = len(self.res['log_like'])

        if self.fit_star:

            rstar, mstar, T, logg = self.misti.interp_value([self.res['eep'],self.res['age'],self.res['feh']],['radius','mass','Teff','logg']).T
            self.dres['rstar'] = rstar
            self.dres['mstar'] = mstar
            self.dres['Tstar'] = T
            self.dres['loggstar'] = logg

            interp_input = np.array([T, logg, self.res['feh']]).T

            for filt in np.unique(self.filters):

                self.dres['u1 {0}'.format(filt)] = self.ldgrids[filt][0](interp_input)
                self.dres['u2 {0}'.format(filt)] = self.ldgrids[filt][1](interp_input)

        else:

            T = np.random.normal(self.Ts, self.Tserr, n)
            rstar = np.random.normal(self.rs, self.rserr, n)
            mstar = np.random.normal(self.ms, self.mserr, n)

        J = np.random.normal(self.Jmag, self.Jmagerr, n)
        einsol = 1*u.Lsun / (4 * np.pi * u.AU**2)

        for i in range(self.n):

            if self.is_transit[i] and self.fit_transit:

                if self.fit_ttv[i]:

                    k = np.sum(self.fit_ttv[:i])

                    p = np.array(ps[k])
                    self.dres['P {0}'.format(i+1)] = p

                    tc = np.array(tcs[k])
                    self.dres['Tc {0}'.format(i+1)] = tc

                else:

                    p = np.exp(self.res['log(P) {0}'.format(i+1)])
                    self.dres['P {0}'.format(i+1)] = p

                rp = self.res['ror {0}'.format(i+1)] * rstar * (1*u.Rsun).to(u.earthRad).value
                self.dres['Rp {0}'.format(i+1)] = rp

                if not self.fit_star:

                    ars = np.exp(self.res['log(a/rs) {0}'.format(i+1)])
                    self.dres['a/rs {0}'.format(i+1)] = ars

                    a = ars * rstar * (1*u.Rsun).to(u.AU).value
                    self.dres['a {0}'.format(i+1)] = a

                inc = np.arccos(self.res['cos(i) {0}'.format(i+1)]) * 180/np.pi
                self.dres['i {0}'.format(i+1)] = inc


            elif self.is_rv[i] and self.fit_rv:

                p = np.exp(self.res['log(P) {0}'.format(i+1)])
                self.dres['P {0}'.format(i+1)] = p

            if self.fit_ecc[i]:

                e = self.res['secw {0}'.format(i+1)]**2 + self.res['sesw {0}'.format(i+1)]**2
                self.dres['e {0}'.format(i+1)] = e

                w = np.arctan2(self.res['sesw {0}'.format(i+1)], self.res['secw {0}'.format(i+1)]) * 180/np.pi
                self.dres['w {0}'.format(i+1)] = w

            else:

                e = 0
                w = 90

            mp = 0

            if self.is_rv[i] and self.fit_rv:

                k = np.exp(self.res['log(K) {0}'.format(i+1)])
                self.dres['K {0}'.format(i+1)] = k

                if self.is_transit[i] and self.fit_transit:

                    mp = calc_m_from_k(p*(1*u.day).to(u.yr).value, k, e, inc*np.pi/180, mstar)
                    self.dres['Mp {0}'.format(i+1)] = mp

                    rhop = mp / (4/3 * np.pi * rp**3) * (1*u.earthMass/u.earthRad**3).to(u.g/u.cm**3).value
                    self.dres['rhop {0}'.format(i+1)] = rhop

                else:

                    mp = calc_m_from_k(p*(1*u.day).to(u.yr).value, k, e, np.pi/2, mstar)
                    self.dres['Mp sini {0}'.format(i+1)] = mp

            if self.fit_star or (self.is_rv[i] and self.fit_rv):

                a = ((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3)
                self.dres['a {0}'.format(i+1)] = a

                ars = (a*u.AU).to(u.Rsun).value / rstar
                self.dres['a/rs {0}'.format(i+1)] = ars

            teq = (1/4)**(1/4) * T * ars**(-1/2)
            self.dres['teq {0}'.format(i+1)] = teq

            sinc = (constants.sigma_sb * (T * u.K)**4 * ars**(-2) / einsol).to(u.dimensionless_unscaled)
            self.dres['sinc {0}'.format(i+1)] = sinc

            if self.is_transit[i] and self.fit_transit:

                b = ars * self.res['cos(i) {0}'.format(i+1)] * (1 - e**2) / (1 + e * np.sin(w * np.pi/180))
                self.dres['b {0}'.format(i+1)] = b

                dur = p / np.pi * np.arcsin(np.sqrt((1 + self.res['ror {0}'.format(i+1)])**2 - b**2) / (ars * np.sqrt(1 - self.res['cos(i) {0}'.format(i+1)]**2))) * (1*u.day).to(u.hr).value
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


        out = []
        for x in self.res:
            if x == 'log_like':
                continue
            out.append([x, np.nanmedian(self.res[x])]+list(np.diff(np.nanpercentile(self.res[x], [16,50,84]))))
        for x in self.dres:
            out.append([x, np.nanmedian(self.dres[x])]+list(np.diff(np.nanpercentile(self.dres[x], [16,50,84]))))


        tab = Table(rows = out, names = ['Parameter','Median','-Error','+Error'])
        tab.write(self.direc+'Results/'+name+'.txt', format = 'ascii.fixed_width_two_line', overwrite = True, delimiter = '|', delimiter_pad = ' ', bookend = True)

        
        self.plot_pl_chains(name, show_plot = show_plots)

        if self.nttv > 0 and self.fit_transit:

            self.plot_ttv_chains(name, show_plot = show_plots)

        if np.any(self.detrend) and self.fit_transit:

            self.plot_det_chains(name, show_plot = show_plots)

        if self.fit_star:

            self.plot_star_chains(name, show_plot = show_plots)

        self.plot_pl_corner(name, show_plot = show_plots)

        if self.fit_star:

            self.plot_star_corner(name, show_plot = show_plots)

        self.plot_big_corner(name)

        if self.fit_transit:

            self.gen_lcs(name, lc_supersample_size)

            self.plot_full_lc(name, show_plot = show_plots)

            self.plot_lc_phase(name, show_plot = show_plots)

        if self.fit_rv:

            self.gen_rv(name)

            self.plot_rv(name, show_plot = show_plots)

        if self.nttv > 0 and self.fit_transit:

            self.plot_ttvs(name, show_plot = show_plots)

        if self.fit_star:

            mags = self.misti.interp_mag([self.res['eep'],self.res['age'],self.res['feh'],self.res['distance'],self.res['AV']], ['J','H','K','G','BP','RP','W1','W2','W3'])[3]
            self.magfit = np.median(mags, axis = 0)
            self.magfiterr = np.diff(np.percentile(mags, [16,50,84], axis = 0), axis = 0)

            pickle.dump({'magfit': self.magfit, 'magfiterr': self.magfiterr}, open(self.direc+'Output/'+name+'_magfit.p', 'wb'))

            self.plot_sed_fit(name, show_plot = show_plots)


        self.print_results(name)


    def fit_star_only(self, name):

        misti = get_ichrone('mist')

        props = {'parallax': (self.plax, self.plaxerr), 'Teff': (self.Ts, self.Tserr),
        'J': (self.Jmag, self.Jmagerr), 'H': (self.Hmag, self.Hmagerr), 'K': (self.Kmag, self.Kmagerr),
        'W1': (self.W1mag, self.W1magerr), 'W2': (self.W2mag, self.W2magerr), 'W3': (self.W3mag, self.W3magerr),
        'G': (self.Gmag, self.Gmagerr), 'BP': (self.Bpmag, self.Bpmagerr), 'RP': (self.Rpmag, self.Rpmagerr)}

        if not np.isnan(self.logg):
            props['logg'] = (self.logg, self.loggerr)
        
        if not np.isnan(self.feh):
            props['feh'] = (self.feh, self.feherr)

        mod = SingleStarModel(misti, name=name, **props)

        if self.use_priors:
            mod = apply_star_priors(self.init_priors.table, mod)
        

        mod.fit(overwrite = True, basename = self.direc+'multinest chains/'+name+'-')


        self.dres = {}
        out = []
        for x in mod.derived_samples:
            self.dres[x] = np.array(mod.derived_samples[x])
            out.append([x, np.nanmedian(self.dres[x])]+list(np.diff(np.nanpercentile(self.dres[x], [16,50,84]))))

        pickle.dump(self.dres, open(self.direc+'Output/'+name+'_dres.p', 'wb'))

        tab = Table(rows = out, names = ['Parameter','Median','-Error','+Error'])
        tab.write(self.direc+'Results/'+name+'.txt', format = 'ascii.fixed_width', overwrite = True)


        fig = mod.corner_observed()
        
        fig.suptitle('Observed Params')

        plt.tight_layout()

        fig.savefig(self.direc+'Plots/'+name+'/corner_observed.png')

        plt.show()


        fig = mod.corner_physical()

        fig.suptitle('Physical Params')

        plt.tight_layout()

        fig.savefig(self.direc+'Plots/'+name+'/corner_physical.png')

        plt.show()

        maglist = ['J_mag','H_mag','K_mag','G_mag','BP_mag','RP_mag','W1_mag','W2_mag','W3_mag']
        self.magfit = np.array([np.median(self.dres[x]) for x in maglist])
        self.magfiterr = np.diff(np.percentile([self.dres[x] for x in maglist], [16,50,84], axis = 1), axis = 0)

        pickle.dump({'magfit': self.magfit, 'magfiterr': self.magfiterr}, open(self.direc+'Output/'+name+'_magfit.p', 'wb'))

        self.plot_sed_fit(name)


    def load_results(self, name):

        if os.path.exists(self.direc+'/Output/'+name+'_dres.p'):
            self.dres = pickle.load(open(self.direc+'/Output/'+name+'_dres.p', 'rb'))
        
        else:
            print('No run named {0}.'.format(name))
            return
    
        if os.path.exists(self.direc+'/Output/'+name+'_res.p'):
            self.res = pickle.load(open(self.direc+'/Output/'+name+'_res.p', 'rb'))

        if os.path.exists(self.direc+'/Masks/'+name+'_masks.p'):
            
            self.masks = pickle.load(open(self.direc+'/Masks/'+name+'_masks.p', 'rb'))

            self.tt = copy.deepcopy(self.tt_orig)
            self.f = copy.deepcopy(self.f_orig)
            self.ferr = copy.deepcopy(self.ferr_orig)

            for i in range(len(self.tt)):
                self.tt[i] = self.tt[i][self.masks[i]]
                self.f[i] = self.f[i][self.masks[i]]
                self.ferr[i] = self.ferr[i][self.masks[i]]

        
    def load_lcs(self, name):

        self.fm = pickle.load(open(self.direc+'Output/'+name+'_fm.p', 'rb'))


    def load_rv(self, name):

        self.rvm = pickle.load(open(self.direc+'Output/'+name+'_rvm.p', 'rb'))


    def load_magfit(self, name):

        mags = pickle.load(open(self.direc+'Resulsts/'+name+'_magfit.p', 'rb'))
        self.magfit = mags['magfit']
        self.magfiterr = mags['magfiterr']


    def load_samples(self, name):

            z = pickle.load(open(self.direc+'Output/'+name+'_samples.p', 'rb'))
            self.parnames = z['parnames']
            self.samples = z['samples']
            self.log_likes_full = z['log_like']


    def delete_run(self, name):

        if os.path.exists(self.direc+'Masks/'+name+'_masks.p'):
            os.remove(self.direc+'Masks/'+name+'_masks.p')

        if os.path.isdir(self.direc+'Plots/'+name):
            shutil.rmtree(self.direc+'Plots/'+name)

        if os.path.exists(self.direc+'Output/'+name+'_res.p'):
            os.remove(self.direc+'Output/'+name+'_res.p')

        if os.path.exists(self.direc+'Output/'+name+'_dres.p'):
            os.remove(self.direc+'Output/'+name+'_dres.p')

        if os.path.exists(self.direc+'Output/'+name+'_fm.p'):
            os.remove(self.direc+'Output/'+name+'_fm.p')

        if os.path.exists(self.direc+'Output/'+name+'_rvm.p'):
            os.remove(self.direc+'Output/'+name+'_rvm.p')

        if os.path.exists(self.direc+'Output/'+name+'_samples.p'):
            os.remove(self.direc+'Output/'+name+'_samples.p')

        if os.path.exists(self.direc+'Output/'+name+'_magfit.p'):
            os.remove(self.direc+'Output/'+name+'_magfit.p')

        if os.path.exists(self.direc+'Results/'+name+'.txt'):
            os.remove(self.direc+'Results/'+name+'.txt')

        if os.path.exists(self.direc+'multinest chains/'+name+'-.txt'):
            for filename in os.listdir(self.direc+'multinest chains/'):
                if len(filename) >= len(name) and filename[:len(name)] == name:
                    os.remove(self.direc+'multinest chains/'+filename)


    def initialize_ttvs(self, name):

        if not os.path.exists(self.direc+'/'+name):

            tab = Init_ttvs(self.direc).create().table

        else:

            tab = Init_ttvs(self.direc).from_file().table

        self.ttvs0 = {int(col): np.sort(np.array(tab[col])[~np.isnan(tab[col])]) for col in tab.columns}

        self.ttvsectors = {}
        self.ttvi = {}

        for i in range(self.n):

            if self.is_transit[i] and self.fit_ttv[i]:

                if i+1 not in self.ttvs0:

                    print('Need to input transit times in init_ttv file for planet {0}, which is set to fit_ttv = True. Either edit the file or run create_init_ttvs.')
                    break

                ttvi0 = np.round((np.array(self.ttvs0[i+1]) - self.ttvs0[i+1][0]) / self.p[i], 0)
                self.ttvi['{0}'.format(i+1)] = ttvi0

                for z in range(len(self.ttvs0[i+1])):

                    for l in range(len(self.tt)):

                        if np.min(self.tt[l]) <= self.ttvs0[i+1][z] <= np.max(self.tt[l]):

                            self.ttvsectors['{0} {1}'.format(i+1, z+1)] = l
                            break


    def plot_pl_chains(self, name, show_plot = True):

        fit_rv = False

        if 'trend 0' in self.res:

            fit_rv = True

        fit_transit = False

        if 'F0 {0}'.format(self.lcnames[0]) in self.res:

            fit_transit = True

        fit_star = False

        if 'eep' in self.res:

            fit_star = True

        num = (self.n if fit_rv else self.nt) if fit_transit else self.nr

        fig, ax = plt.subplots(4 + (3 if fit_transit else 0) + (1 if fit_rv else 0) - (1 if fit_star else 0), num, figsize = (7*self.n, 18), sharex = True, layout = 'constrained')

        if num == 1:
            ax = np.array([ax.T]).T

        for i in range(self.n):

            if not ((fit_transit and self.is_transit[i]) or (fit_rv and self.is_rv[i])):

                continue

            if fit_transit:

                if fit_rv:

                    j = i

                else:

                    j = np.sum(self.is_transit[:i])
            
            else:

                j = np.sum(self.is_rv[:i])

            ax[0][j].set_title('Planet {0}'.format(i+1))

            if self.is_transit[i] and fit_transit:

                if not self.fit_ttv[i]:

                    v = self.parnames['log(P) {0}'.format(i+1)]
                    ax[0][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
                    
                    v = self.parnames['Tc {0}'.format(i+1)]
                    ax[1][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

                v = self.parnames['ror {0}'.format(i+1)]
                ax[2][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

                if not fit_star:
                    v = self.parnames['log(a/rs) {0}'.format(i+1)]
                    ax[3][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

                v = self.parnames['cos(i) {0}'.format(i+1)]
                ax[4 - (1 if fit_star else 0)][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            elif self.is_rv[i] and fit_rv:

                v = self.parnames['log(P) {0}'.format(i+1)]
                ax[0][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
                
                v = self.parnames['Tc {0}'.format(i+1)]
                ax[1][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            if self.is_rv[i] and fit_rv:

                v = self.parnames['log(K) {0}'.format(i+1)]
                ax[2 + (3 if fit_transit else 0) - (1 if fit_star else 0)][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            if self.fit_ecc[i] and ((self.is_transit[i] and fit_transit) or (self.is_rv[i] and fit_rv)):

                v = self.parnames['secw {0}'.format(i+1)]
                ax[2 + (3 if fit_transit else 0) + (1 if fit_rv else 0) - (1 if fit_star else 0)][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

                v = self.parnames['sesw {0}'.format(i+1)]
                ax[3 + (3 if fit_transit else 0) + (1 if fit_rv else 0) - (1 if fit_star else 0)][j].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

        ax[0][0].set_ylabel('log(P)')
        ax[1][0].set_ylabel('Tc')
        if fit_transit:
            ax[2][0].set_ylabel('ror')
            if not fit_star:
                ax[3][0].set_ylabel('log(a/rs)')
            ax[4 - (1 if fit_star else 0)][0].set_ylabel('cos(i)')
        if fit_rv:
            ax[2 + (3 if fit_transit else 0) - (1 if fit_star else 0)][0].set_ylabel('log(K)')
        ax[2 + (3 if fit_transit else 0) + (1 if fit_rv else 0) - (1 if fit_star else 0)][0].set_ylabel('secw')
        ax[3 + (3 if fit_transit else 0) + (1 if fit_rv else 0) - (1 if fit_star else 0)][0].set_ylabel('sesw')

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/pl_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_ttv_chains(self, name, show_plot = True):

        n = 0
        for x in self.ttvi:
            if len(self.ttvi[x]) > n:
                n = len(self.ttvi[x])

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

                v = self.parnames['TT {0} {1}'.format(i+1, z+1)]
                ax[z][k].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)


        fig.supxlabel('N Steps')
        fig.supylabel('Transit Times')

        fig.savefig(self.direc+'Plots/'+name+'/tt_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_det_chains(self, name, show_plot = True):

        fig, ax = plt.subplots(np.sum(self.detrend), 3, figsize = (15, 3*len(self.tt)), sharex = True, layout = 'constrained')

        if len(self.tt) == 1:
            ax = np.array([ax])

        for j in range(len(self.tt)):

            if not self.detrend[j]:
                continue

            i = np.sum(self.detrend[:j])

            ax[i][0].set_ylabel('{0}'.format(self.lcnames[i]))

            v = self.parnames['F0 {0}'.format(self.lcnames[i])]
            ax[i][0].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            v = self.parnames['log(rho_gp) {0}'.format(self.lcnames[i])]
            ax[i][1].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

            v = self.parnames['log(sigma_gp) {0}'.format(self.lcnames[i])]
            ax[i][2].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)

        ax[0][0].set_title('F0')
        ax[0][1].set_title('log(rho_gp)')
        ax[0][2].set_title('log(sigma_gp)')

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/sec_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_star_chains(self, name, show_plot = True):

        fig, ax = plt.subplots(5, figsize = (7, 12), sharex = True, layout = 'constrained')

        for i, z in enumerate(['eep','age','feh','distance','AV']):

            v = self.parnames[z]
            ax[i].plot(self.samples[:,:,v], color = 'black', alpha = 0.2)
            ax[i].set_ylabel(z)

        fig.supxlabel('N Steps')

        fig.savefig(self.direc+'Plots/'+name+'/star_chains.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_pl_corner(self, name, show_plot = True):

        fit_rv = False

        if 'trend 0' in self.res:

            fit_rv = True

        fit_transit = False

        if 'F0 {0}'.format(self.lcnames[0]) in self.res:

            fit_transit = True

        fit_star = False

        if 'eep' in self.res:

            fit_star = True

        for i in range(self.n):

            labels = []

            if self.is_transit[i] and fit_transit:

                labels += ['ror {0}'.format(i+1), 'cos(i) {0}'.format(i+1)]

                if not fit_star:

                    labels.append('log(a/rs) {0}'.format(i+1))

                if self.fit_ttv[i]:

                    for k in range(len(self.ttvi['{0}'.format(i+1)])):

                        labels.append('TT {0} {1}'.format(i+1, k+1))

                else:

                    labels += ['log(P) {0}'.format(i+1), 'Tc {0}'.format(i+1)]

            elif self.is_rv[i] and fit_rv:

                labels += ['log(P) {0}'.format(i+1), 'Tc {0}'.format(i+1)]

            if self.is_rv[i] and fit_rv:

                labels.append('log(K) {0}'.format(i+1))

            if self.fit_ecc[i] and ((self.is_transit[i] and fit_transit) or (self.is_rv[i] and fit_rv)):

                labels += ['secw {0}'.format(i+1), 'sesw {0}'.format(i+1)]

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


    def plot_star_corner(self, name, show_plot = True):

        labels = ['eep','age','feh','distance','AV']
        j = [self.parnames[x] for x in labels]

        fig = corner.corner(self.flat_samples[:,j], labels = labels)

        fig.suptitle('Star')

        plt.tight_layout()

        fig.savefig(self.direc+'Plots/'+name+'/star_corner.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_big_corner(self, name, show_plot = False):

        keys = list(self.res.keys())
        keys.remove('log_like')

        samps = np.array([self.res[keys[i]] for i in range(len(keys))]).T

        fig = corner.corner(samps, labels = keys)

        fig.savefig(self.direc+'Plots/'+name+'/bigcorner.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def gen_lcs(self, name, lc_supersample_size = 600):

        print('\nGenerating light curves for plots.')

        self.supersamples = np.array([max(1,int(x/lc_supersample_size)) for x in self.exptimes*24*60*60])

        self.fm = {}

        if np.any(self.fit_ttv):
            self.initialize_ttvs(name = self.init_ttvs_name)

        fit_rv = False

        if 'trend 0' in self.res:

            fit_rv = True

        fit_ld = False

        if 'u1 {0}'.format(self.filters[0]) in self.res:

            fit_ld = True

        fit_star = False

        if 'eep' in self.res:

            fit_star = True

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

            rstar, mstar, Tstar, loggstar = self.misti.interp_value([self.res['eep'],self.res['age'],self.res['feh']],['radius','mass','Teff','logg']).T

            arlist = []
            for j in range(self.n):

                if not self.is_transit[j]:
                    arlist.append(np.nan)
                    continue

                p = self.dres['P {0}'.format(j+1)]

                if self.is_rv[j] and fit_rv:

                    e = 0
                    if self.fit_ecc[j]:
                        e = self.res['secw {0}'.format(j+1)]**2 + self.res['sesw {0}'.format(j+1)]**2

                    mp = calc_m_from_k(p, np.exp(self.res['log(K) {0}'.format(j+1)]), e, np.arccos(self.res['cos(i) {0}'.format(j+1)]), mstar)

                else:

                    mp = 0

                ar = (((mstar + (mp*u.earthMass).to(u.Msun).value) * (p*u.day).to(u.yr).value**2)**(1/3) * u.AU).to(u.Rsun).value / rstar

                arlist.append(ar)

        for i in range(len(self.lcnames)):

            sec = self.lcnames[i]

            print(sec+':')

            ld = np.array(self.ld[self.filters[i]])

            if fit_ld:

                ld = np.array([self.res['u1 {0}'.format(self.filters[i])], self.res['u2 {0}'.format(self.filters[i])]]).T

            if fit_star:

                starpars = np.array([Tstar, loggstar, self.res['feh']]).T
                ld = np.array([self.ldgrids[self.filters[i]][0](starpars)[0], self.ldgrids[self.filters[i]][1](starpars)[0]]).T

            detrend = self.detrend[i]

            n = len(self.res['ror 1'])

            pars = []
            tcs = []

            for j in range(self.n):

                if not self.is_transit[j]:
                    continue
                
                if self.fit_ttv[j]:

                    p = self.dres['P {0}'.format(j+1)]
                    tc = self.dres['Tc {0}'.format(j+1)]
                    ror = self.res['ror {0}'.format(j+1)]
                    if fit_star:
                        ar = arlist[j]
                    else:
                        ar = np.exp(self.res['log(a/rs) {0}'.format(j+1)])
                    inc = np.arccos(self.res['cos(i) {0}'.format(j+1)]) * 180/np.pi
                    e = self.res['sqrt(e)cos(w) {0}'.format(j+1)]**2 + self.res['sqrt(e)sin(w) {0}'.format(j+1)]**2 if 'sqrt(e)cos(w) {0}'.format(j+1) in self.res else np.zeros(np.shape(self.res['ror {0}'.format(j+1)]))
                    w = np.arctan2(self.res['sqrt(e)sin(w) {0}'.format(j+1)], self.res['sqrt(e)cos(w) {0}'.format(j+1)]) * 180/np.pi if 'sqrt(e)cos(w) {0}'.format(j+1) in self.res else np.full(np.shape(self.res['ror {0}'.format(j+1)]), 90)
                    
                    pars.append(np.array([p,tc,ror,ar,inc,e,w]).T)

                    tcs.append(np.median(tc))

                else:

                    pars.append(get_transit_params(self.res, j+1, ar = arlist[j] if fit_star else None).T)
                    tcs.append(np.median(self.res['Tc {0}'.format(j+1)]))


            mean = self.res['F0 {0}'.format(sec)]

            ttphase = np.linspace(-0.5, 0.5, 1000)
            phaseexp = 1/1000
            phasess = 1

            if detrend:

                rhogp = np.exp(self.res['log(rho_gp) {0}'.format(sec)])
                sigmagp = np.exp(self.res['log(sigma_gp) {0}'.format(sec)])

                kernel = terms.SHOTerm(rho = rhogp[0], sigma = sigmagp[0], Q = 1/np.sqrt(2))
                gp = GaussianProcess(kernel = kernel)

                gpf_err = []

            fm = []
            fmphase = []

            fmsum = 0

            for k in range(self.n):

                if not self.is_transit[k]:
                    continue

                l = np.sum(self.is_transit[:k])

                fm0 = np.zeros(len(self.tt[i]))

                fmphase.append(lightCurve(np.median(pars[l], axis = 0), ttphase + tcs[l], np.median(ld, axis = 0) if ld.ndim > 1 else ld, phaseexp, phasess) + np.median(mean))

                if self.fit_ttv[k]:

                    p = np.median(pars[l][:,0])

                    for z in range(len(self.ttvi['{0}'.format(k+1)])):

                        if self.ttvsectors['{0} {1}'.format(k+1, z+1)] != i:
                            continue

                        ttime = np.median(self.res['TT {0} {1}'.format(k+1, z+1)])

                        ind = np.where((self.tt[i] >= ttime - p/4) & (self.tt[i] <= ttime + p/4))

                        parstemp = np.median(pars[l], axis = 0)
                        parstemp[1] = ttime

                        fm0[ind] += lightCurve(parstemp, self.tt[i][ind], np.median(ld, axis = 0) if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])


                else:

                    fm0 = lightCurve(np.median(pars[l], axis = 0), self.tt[i], np.median(ld, axis = 0) if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])
                    
                fmsum += fm0
                fm.append(fm0)

            if detrend:

                gp = set_gp_params(np.median(rhogp), np.median(sigmagp), self.tt[i], self.ferr[i], gp)
                gpf0 = gp.predict(self.f[i] - fmsum - np.median(mean))
                gpf = gpf0


            z = {'fm': np.array(fm), 'fmphase': np.array(fmphase), 'ttphase': ttphase}

            if detrend:

                z['gpf'] = np.array(gpf)


            fm_err = [[] for j in range(self.nt)]
            fmphase_err = [[] for j in range(self.nt)]

            for j in tqdm(range(0,n,10)):

                fmsum = 0

                for k in range(self.n):

                    if not self.is_transit[k]:
                        continue

                    l = np.sum(self.is_transit[:k])

                    fm0 = np.zeros(len(self.tt[i]))

                    fmphase_err[l].append(lightCurve(pars[l][j], ttphase + tcs[l], ld[j] if ld.ndim > 1 else ld, phaseexp, phasess) + mean[j])

                    if self.fit_ttv[k]:

                        p = pars[l][j][0]

                        for ii in range(len(self.ttvi['{0}'.format(k+1)])):

                            if self.ttvsectors['{0} {1}'.format(k+1, ii+1)] != i:
                                continue

                            ttime = self.res['TT {0} {1}'.format(k+1, ii+1)][j]

                            pars[l][j][1] = ttime

                            ind = np.where((self.tt[i] >= ttime - p/4) & (self.tt[i] <= ttime + p/4))

                            fm0[ind] += lightCurve(pars[l][j], self.tt[i][ind], ld[j] if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])


                    else:

                        fm0 = lightCurve(pars[l][j], self.tt[i], ld[j] if ld.ndim > 1 else ld, self.exptimes[i], self.supersamples[i])
                        
                    fmsum += fm0
                    fm_err[l].append(fm0)

                if detrend:

                    gp = set_gp_params(rhogp[j], sigmagp[j], self.tt[i], self.ferr[i], gp)
                    gpf0 = gp.predict(self.f[i] - fmsum - mean[j])
                    gpf_err.append(gpf0)


            fm_err = np.percentile(fm_err, [16,84], axis = 1)
            fmphase_err = np.percentile(fmphase_err, [16,84], axis = 1)

            z['fm_err'] = fm_err
            z['fmphase_err'] = fmphase_err

            if detrend:

                gpf_err = np.percentile(gpf_err, [16,84], axis = 0)
                z['gpf_err'] = gpf_err

            self.fm[sec] = z

        pickle.dump(self.fm, open(self.direc+'Output/'+name+'_fm.p', 'wb'))


    def plot_full_lc(self, name, show_plot = True):
        
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

            fm = self.fm[sec]['fm']
            fm_err = self.fm[sec]['fm_err']
            mean = np.median(self.res['F0 {0}'.format(sec)])

            if self.detrend[i]:

                gpf = self.fm[sec]['gpf']
                gpf_err = self.fm[sec]['gpf_err']

                ax['a'].errorbar(self.tt[i], self.f[i], yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
                ax['a'].plot(self.tt[i], gpf + mean, color = 'green', label = 'GP Model', zorder = 3, linewidth = 2)
                ax['a'].fill_between(self.tt[i], gpf_err[0]+mean, gpf_err[1]+mean, color = 'green', edgecolor = 'none', alpha = 0.5, zorder = 2)

                ax['a'].text(0.01, 0.99, 'Raw', fontsize = 20, ha = 'left', va = 'top', transform = ax['a'].transAxes)

                ax['a'].tick_params(axis = 'both', labelsize = 15)
                ax['a'].set_yticks(ticks = ax['a'].get_yticks(), labels = np.round((np.array(ax['a'].get_yticks())-1)*1000, 1))
                ax['a'].set_title(str(sec), fontsize = 20)

            ax['b'].errorbar(self.tt[i], self.f[i] - (gpf if self.detrend[i] else 0), yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)

            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                k = np.sum(self.is_transit[:j])

                ax['b'].plot(self.tt[i], fm[k]+mean, label = 'Planet {0}'.format(j+1), zorder = 3, linewidth = 2)
                ax['b'].fill_between(self.tt[i], fm_err[0][k]+mean, fm_err[1][k]+mean, zorder = 2, alpha = 0.5, edgecolor = 'none')
            
            ax['b'].text(0.01, 0.99, 'Corrected', fontsize = 20, ha = 'left', va = 'top', transform = ax['b'].transAxes)

            ax['b'].tick_params(axis = 'both', labelsize = 15)
            ax['b'].legend(fontsize = 15)

            if not self.detrend[i]:
                ax['b'].set_yticks(ticks = ax['b'].get_yticks(), labels = np.round((np.array(ax['b'].get_yticks())-1)*1000, 1))
                ax['b'].set_title(str(sec), fontsize = 20)

            mod = np.sum(fm, axis = 0) + mean + (gpf if self.detrend[i] else 0)
            ax['c'].errorbar(self.tt[i], self.f[i] - mod, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
            ax['c'].axhline(0, color = 'red', lw = 1, zorder = 2)

            ax['c'].text(0.01, 0.99, 'Residuals', fontsize = 20, ha = 'left', va = 'top', transform = ax['c'].transAxes)

            ax['c'].tick_params(axis = 'both', labelsize = 15)
            ax['c'].set_yticks(ticks = ax['c'].get_yticks(), labels = np.round(np.array(ax['c'].get_yticks())*1000, 1))
            ax['b'].sharex(ax['c'])
            plt.setp(ax['b'].get_xticklabels(), visible = False)
            ax['c'].set_xlabel('Time [BJD-2450000]', fontsize = 20)

            if self.detrend[i]:
                ax['b'].sharey(ax['a'])
                ax['a'].sharex(ax['c'])
                plt.setp(ax['a'].get_xticklabels(), visible = False)

            fig.supylabel('Relative Flux [ppt]', fontsize = 20)

            for a in ax:
                ax[a].set_rasterized(True)

            fig.savefig(self.direc+'Plots/'+name+'/transit_det_{0}.png'.format(sec))

            if show_plot:
                plt.show()

            else:
                plt.close()


    def plot_lc_phase(self, name, show_plot = True):

        if np.any(self.fit_ttv):
            self.initialize_ttvs(name = self.init_ttvs_name)

        for i in range(len(self.lcnames)):

            sec = self.lcnames[i]

            detrend = False
            if 'log(rho_gp) {0}'.format(sec) in self.res:
                detrend = True

            alpha = 0.1
            if self.exptimes[i] == 1800/60/60/24:
                alpha = 0.3

            mos = []
            for j in range(self.nt):
                mos.append(['a{0}'.format(j)])
                mos.append(['a{0}'.format(j)])
                mos.append(['b{0}'.format(j)])

            fig, ax = plt.subplot_mosaic(mos, figsize = (14,6*self.nt), sharex = True, layout = 'constrained')

            mean = np.median(self.res['F0 {0}'.format(sec)])

            for j in range(self.n):

                if not self.is_transit[j]:
                    continue

                k = np.sum(self.is_transit[:j])

                p = np.median(self.dres['P {0}'.format(j+1)])

                if self.fit_ttv[j]:
                    tc = np.median(self.dres['Tc {0}'.format(j+1)])
                else:
                    tc = np.median(self.res['Tc {0}'.format(j+1)])

                newt = self.tt[i].copy()

                if self.fit_ttv[j]:

                    for z in range(len(self.ttvi['{0}'.format(j+1)])):

                        if self.ttvsectors['{0} {1}'.format(j+1, z+1)] != i:
                            continue

                        ttime = np.median(self.res['TT {0} {1}'.format(j+1, z+1)])

                        ind = np.where((self.tt[i] >= ttime - 0.5) & (self.tt[i] <= ttime + 0.5))

                        newt[ind] -= ttime - (tc + p * self.ttvi['{0}'.format(j+1)][z])



                other = np.sum(self.fm[sec]['fm'], axis = 0) - self.fm[sec]['fm'][k] + (self.fm[sec]['gpf'] if detrend else 0)
                xfold = (newt - tc + 0.5 * p) % p - 0.5 * p
                ttphase = self.fm[sec]['ttphase']
                fmphase = self.fm[sec]['fmphase'][k]
                fmphase_err = self.fm[sec]['fmphase_err'][:,k]
                fm = self.fm[sec]['fm'][k]

                ax['a{0}'.format(j)].errorbar(xfold, self.f[i] - other, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)

                exp = self.exptimes[i]*60*60*24
                bins = np.linspace(-0.5, 0.5, 15 if exp == 1800 else (30 if exp == 600 else 45))
                denom, _ = np.histogram(xfold, bins)
                num, _ = np.histogram(xfold, bins, weights = self.f[i] - other)
                denom[num == 0] = 1.0
                ax['a{0}'.format(j)].scatter(0.5 * (bins[1:] + bins[:-1]), num / denom, color='mediumseagreen', zorder = 2)

                ax['a{0}'.format(j)].plot(ttphase, fmphase, zorder = 4, linewidth = 2, color = 'dodgerblue')
                ax['a{0}'.format(j)].fill_between(ttphase, fmphase_err[0], fmphase_err[1], zorder = 3, alpha = 0.5, color = 'dodgerblue', edgecolor = 'none')

                ax['a{0}'.format(j)].text(0.01, 0.99, 'Planet {0}'.format(j+1), fontsize = 20, ha = 'left', va = 'top', transform = ax['a{0}'.format(j)].transAxes)

                l = np.where((xfold >= -0.5) & (xfold <= 0.5))[0]
                if len(l) > 0:
                    high = np.max(self.f[i][l] - other[l] + self.ferr[i][l])
                    low = np.min(self.f[i][l] - other[l] - self.ferr[i][l])

                else:
                    k = np.where((ttphase >= -0.5) & (ttphase <= 0.5))[0]
                    high = np.max(fmphase[2])
                    low = np.min(fmphase[0])

                ax['a{0}'.format(j)].set_ylim((low - 0.1*(high-low), high + 0.1*(high-low)))

                ax['a{0}'.format(j)].set_yticks(ticks = ax['a{0}'.format(j)].get_yticks()[1:-1], labels = np.round((np.array(ax['a{0}'.format(j)].get_yticks()[1:-1])-1)*1000, 1))
                ax['a{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

                ax['b{0}'.format(j)].errorbar(xfold, self.f[i] - other - fm - mean, yerr = self.ferr[i], fmt = '.k', zorder = 1, alpha = alpha, markersize = 5, markeredgewidth = 0, elinewidth = 1)
                ax['b{0}'.format(j)].axhline(0, c = 'red', lw = 1, zorder = 2)

                ax['b{0}'.format(j)].text(0.01, 0.99, 'Residuals', fontsize = 20, ha = 'left', va = 'top', transform = ax['b{0}'.format(j)].transAxes)

                ax['b{0}'.format(j)].set_yticks(ticks = ax['b{0}'.format(j)].get_yticks(), labels = np.round(np.array(ax['b{0}'.format(j)].get_yticks())*1000, 1))
                ax['b{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

            ax['a0'].set_xlim(-0.5,0.5)
            ax['a0'].set_title(str(sec), fontsize = 20)
            ax['b{0}'.format(self.nt-1)].set_xlabel('Time since $T_{C}$ [days]', fontsize = 20)
            fig.supylabel('Relative Flux [ppt]', fontsize = 20)

            for a in ax:
                ax[a].set_rasterized(True)

            fig.savefig(self.direc+'/Plots/'+name+'/transits_{0}.png'.format(sec))

            if show_plot:
                plt.show()

            else:
                plt.close()


    def gen_rv(self, name):

        print('\nGenerating RV models for plots.')

        n = len(self.res['log_like'])

        pars = []
        ps = []
        tcs = []

        for i in range(self.n):

            if not self.is_rv[i]:
                continue

            if 'log(P) {0}'.format(i+1) not in self.res:

                pars.append(get_rv_params(self.res, i+1, self.dres['P {0}'.format(i+1)], self.dres['Tc {0}'.format(i+1)]).T)
                ps.append(np.median(self.dres['P {0}'.format(i+1)]))
                tcs.append(np.median(self.dres['Tc {0}'.format(i+1)]))

            else:

                pars.append(get_rv_params(self.res, i+1).T)
                ps.append(np.median(self.dres['P {0}'.format(i+1)]))
                tcs.append(np.median(self.res['Tc {0}'.format(i+1)]))
        

        bkg_order = 0

        trend0 = self.res['trend 0']

        if 'trend 1' in self.res:
            trend1 = self.res['trend 1']
            bkg_order = 1

        if 'trend 2' in self.res:
            trend2 = self.res['trend 2']
            bkg_order = 2

        rvm = []
        rvmplot = []
        trplot = np.linspace(np.min(self.tr)-5, np.max(self.tr)+5, 1000)
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

        self.rvm = {'rvm': rvm, 'bkg': bkg, 'rvallplot': rvallplot, 'rvallplot_err': rvallplot_err, 'rvmplot': rvmplot, 'bkgplot': bkgplot, 'rvmphase': rvmphase, 'rvmphase_err': rvmphase_err, 'trplot': trplot, 'trphase': trphase}

        pickle.dump(self.rvm, open(self.direc+'Output/'+name+'_rvm.p', 'wb'))

    
    def plot_rv(self, name, show_plot = True):

        mos = [['a' for i in range(self.nr)], ['a' for i in range(self.nr)], ['b' for i in range(self.nr)], ['c{0}'.format(i) for i in range(self.nr)], ['c{0}'.format(i) for i in range(self.nr)], ['d{0}'.format(i) for i in range(self.nr)]]

        fig, ax = plt.subplot_mosaic(mos, figsize = (14*self.nr,12), layout = 'constrained')

        trplot = self.rvm['trplot']
        rvallplot = self.rvm['rvallplot']
        rvallplot_err = self.rvm['rvallplot_err']
        rvmplot = self.rvm['rvmplot']
        bkgplot = self.rvm['bkgplot']
        rvm = self.rvm['rvm']
        bkg = self.rvm['bkg']
        rvmphase = self.rvm['rvmphase']
        rvmphase_err = self.rvm['rvmphase_err']
        trphase = self.rvm['trphase']

        rv = self.rv.copy()
        for i, k in enumerate(np.unique(self.which_rv)):

            j = np.where(self.which_rv == k)[0]

            if i > 1:

                rv[j] -= np.median(self.res['offset {0}'.format(self.rvnames[k])])

            ax['a'].errorbar(self.tr[j], rv[j], yerr = self.rverr[j], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][i], label = self.rvnames[k], zorder = 3, markersize = 10, elinewidth = 2)


        ax['a'].plot(trplot, rvallplot, color = 'mediumseagreen', linewidth = 2, zorder = 2)
        ax['a'].fill_between(trplot, rvallplot_err[0], rvallplot_err[1], color = 'mediumseagreen', zorder = 1, alpha = 0.5, edgecolor = 'none')

        for i in range(self.n):

            if not self.is_rv[i]:
                continue

            j = np.sum(self.is_rv[:i])

            ax['a'].plot(trplot, rvmplot[j], linewidth = 2, linestyle = '--', zorder = 1, label = 'Planet {0}'.format(i+1))

        ax['a'].plot(trplot, bkgplot, color = 'black', linewidth = 2, linestyle = ':', zorder = 0, label = 'Background')

        ax['a'].legend(fontsize = 15, loc = 'upper left')
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

        for i in range(self.n):

            if not self.is_rv[i]:
                continue

            j = np.sum(self.is_rv[:i])

            p = np.median(self.dres['P {0}'.format(i+1)])
            
            if 'Tc {0}'.format(i+1) in self.res:
                tc = np.median(self.res['Tc {0}'.format(i+1)])
            else:
                tc = np.median(self.dres['Tc {0}'.format(i+1)])

            other = np.sum(rvm, axis = 0) - rvm[j]

            xfold = ((self.tr - tc + 0.5 * p) % p - 0.5 * p)/p

            for ii, kk in enumerate(np.unique(self.which_rv)):

                jj = np.where(self.which_rv == kk)[0]

                ax['c{0}'.format(j)].errorbar(xfold[jj], rv[jj] - other[jj] - bkg[jj], yerr = self.rverr[jj], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][ii], zorder = 3, markersize = 10, elinewidth = 2)

            ax['c{0}'.format(j)].plot(trphase, rvmphase[j], linewidth = 2, zorder = 2)
            ax['c{0}'.format(j)].fill_between(trphase, rvmphase_err[0][j], rvmphase_err[1][j], alpha = 0.5, edgecolor = 'none', zorder = 1)

            ax['c{0}'.format(j)].text(0.01, 0.99, 'Planet {0}'.format(i+1), fontsize = 20, ha = 'left', va = 'top', transform = ax['c{0}'.format(j)].transAxes)

            ax['c{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)

            if j == 0:
                ax['c{0}'.format(j)].set_ylabel('RV [m/s]', fontsize = 20)

            else:
                ax['c{0}'.format(j)].sharey(ax['c0'])
                plt.setp(ax['c{0}'.format(j)].get_yticklabels(), visible = False)


            for ii, kk in enumerate(np.unique(self.which_rv)):

                jj = np.where(self.which_rv == kk)[0]

                ax['d{0}'.format(j)].errorbar(xfold[jj], rv[jj] - np.sum(rvm, axis = 0)[jj] - bkg[jj], yerr = self.rverr[jj], fmt = '.', color = ['black','red','darkorange','green','blue','darkorchid'][ii], zorder = 3, markersize = 10, elinewidth = 2)


            ax['d{0}'.format(j)].axhline(0, c = 'red', lw = 1, zorder = 1)

            ax['d{0}'.format(j)].sharex(ax['c{0}'.format(j)])
            plt.setp(ax['c{0}'.format(j)].get_xticklabels(), visible = False)
            ax['d{0}'.format(j)].tick_params(axis = 'both', labelsize = 15)
            ax['d{0}'.format(j)].set_xlabel('Orbital Phase', fontsize = 20)
            ax['d{0}'.format(j)].set_xlim([-0.5,0.5])

            if j == 0:
                ax['d{0}'.format(j)].set_ylabel('Resid. [m/s]', fontsize = 20)

            else:
                ax['d{0}'.format(j)].sharey(ax['d0'])
                plt.setp(ax['d{0}'.format(j)].get_yticklabels(), visible = False)

        
        fig.savefig(self.direc+'Plots/'+name+'/rv.png')

        if show_plot:
            plt.show()

        else:
            plt.close()


    def plot_sed_fit(self, name, show_plot = True):

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


    def plot_ttvs(self, name, show_plot = True):

        self.initialize_ttvs(name = self.init_ttvs_name)

        fig, ax = plt.subplots(self.nttv, figsize = (14, 6*self.nttv), sharex = True, layout = 'constrained')

        if self.nttv == 1:
            ax = np.array([ax])

        for i in range(self.n):

            if not self.fit_ttv[i]:
                continue

            j = np.sum(self.fit_ttv[:i])

            p = self.dres['P {0}'.format(i+1)]
            tc = self.dres['Tc {0}'.format(i+1)]

            ax[j].axhline(0, c = 'red')

            for k in range(len(self.ttvi['{0}'.format(i+1)])):

                tt = self.res['TT {0} {1}'.format(i+1,k+1)]
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


    def print_results(self, name):

        tab = Table.read(self.direc+'Results/'+name+'.txt', format = 'ascii.fixed_width_two_line', delimiter = '|', header_rows = ['name'])
        print(tab)


    def calc_rv_bic(self):

        k = len(self.res)-1
        n = len(self.tr.ravel())

        rvm = np.sum(self.rvm['rvm'], axis = 0) + self.rvm['bkg']

        for i in np.unique(self.which_rv)[1:]:

            j = np.where(self.which_rv == i)[0]

            rvm[j] += np.median(self.res['offset {0}'.format(self.rvnames[i])])

        L = np.sum(lnGauss(self.rv, rvm, self.rverr))

        BIC = k * np.log(n) - 2 * L

        return BIC


    def calc_transit_err_scale(self):

        for i in range(len(self.tt)):

            sec = self.lcnames[i]

            fm = np.sum(self.fm[sec]['fm'], axis = 0) + np.median(self.res['F0 {0}'.format(sec)])

            if 'gpf' in self.fm[sec]:
                fm += self.fm[sec]['gpf']
        
            resstd = np.std(self.f[i] - fm)

            mederr = np.median(self.ferr[i])

            print(sec, 'std', resstd, 'median err', mederr, 'std / median err', resstd / mederr, 'new err scale', resstd/mederr*self.lc_err_scale[i])


    def calc_rv_err_scale(self):

        rvm = np.sum(self.rvm['rvm'], axis = 0) + self.rvm['bkg']

        rv = self.rv.copy()
        for i in np.unique(self.which_rv):

            j = np.where(self.which_rv == i)[0]

            if i > 0:

                rv[j] -= np.median(self.res['offset {0}'.format(self.rvnames[i])])

            resstd = np.std(rv[j] - rvm[j])

            mederr = np.median(self.rverr[j])

            print(self.rvnames[i], 'std', resstd, 'median err', mederr, 'std / median err', resstd / mederr, 'new err scale', resstd/mederr*self.rv_err_scale[i])


    def calc_gelman_rubin(self):

        print('Gelman-Rubin Statistics:')

        for k, v in self.parnames.items():

            s = self.samples[:,:,v]

            L, J = s.shape

            xj = np.mean(s, axis = 0)
            xs = np.mean(xj)
            B = L/(J-1)*np.sum((xj - xs)**2)
            W = np.mean([1/(L-1)*np.sum((s[:,j]-xj[j])**2) for j in range(J)])
            R = ((L-1)/L*W + B/L)/W
            
            print(k, R)


    def rv_lomb_scargle(self, min_freq = None, max_freq = None, freq = None, plot_periods = False, residual = True):

        if residual:
            rvres = self.rv - np.sum(self.rvm['rvm'], axis = 0) - self.rvm['bkg']

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

        else:
            ax.plot(freq, power, c = 'black')

        ax.axhline(ls.false_alarm_level(0.01), c = 'red', label = '1% FA')
        ax.axhline(ls.false_alarm_level(0.05), c = 'orange', label = '5% FA')
        ax.legend()
        plt.show()

        return freq, power





def calc_m_from_k(p, k, e, inc, mstar):

        sini = np.sin(inc)
        
        b = (1*u.earthMass).to(u.Msun).value / mstar
        c = (k * np.sqrt(1-e**2) / 0.08946 / sini)**(-3/2) * p**(-1/2) / mstar

        m = b**2 / (3*c**2) + (2*b**6 + 18*b**3*c**2 + 3*np.sqrt(3)*np.sqrt(4*b**3*c**6 + 27*c**8) + 27*c**4)**(1/3) / (3*2**(1/3)*c**2) - 2**(1/3)*(-b**4 - 6*b*c**2) / (3*c**2*(2*b**6 + 18*b**3*c**2 + 3*np.sqrt(3)*np.sqrt(4*b**3*c**6 + 27*c**8) + 27*c**4)**(1/3))  

        return m


def lightCurve(par, t, ld, expt, ss):
        
        #par = [p, tc, ror, a/rs, i, e, w]
                
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
        
        flux = m.light_curve(params)

        return flux-1


def rvModel(par, t):

    #par = [p, tc, k, e, w]

    par = np.array(par, dtype = float)
    t = np.array(t, dtype = float)

    rv = _rvModel(par, t)

    return rv


def get_transit_params(par, i, ar = None):

    p = np.exp(par['log(P) {0}'.format(i)])
    tc = par['Tc {0}'.format(i)]
    ror = par['ror {0}'.format(i)]
    if ar is None:
        ar = np.exp(par['log(a/rs) {0}'.format(i)])
    inc = np.arccos(par['cos(i) {0}'.format(i)]) * 180/np.pi
    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(np.shape(par['log(P) {0}'.format(i)]))
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) * 180/np.pi if 'secw {0}'.format(i) in par else np.full(np.shape(par['log(P) {0}'.format(i)]), 90)

    return np.array([p, tc, ror, ar, inc, e, w])


def get_ttv_params(par, i, ttvi, ar = None):

    ror = par['ror {0}'.format(i)]
    if ar is None:
        ar = np.exp(par['log(a/rs) {0}'.format(i)])
    inc = np.arccos(par['cos(i) {0}'.format(i)]) * 180/np.pi
    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(np.shape(par['ror {0}'.format(i)]))
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) * 180/np.pi if 'secw {0}'.format(i) in par else np.full(np.shape(par['ror {0}'.format(i)]), 90)

    tts = [par['TT {0} {1}'.format(i, j+1)] for j in range(len(ttvi))]

    res = linregress(ttvi, tts)
    p = res.slope
    tc = res.intercept

    return np.array([p, tc, ror, ar, inc, e, w])


def get_rv_params(par, i, p = None, tc = None):

    if p is None:
        p = np.exp(par['log(P) {0}'.format(i)])
        tc = par['Tc {0}'.format(i)]

    shape = np.shape(par['log(K) {0}'.format(i)])

    e = par['secw {0}'.format(i)]**2 + par['sesw {0}'.format(i)]**2 if 'secw {0}'.format(i) in par else np.zeros(shape)
    w = np.arctan2(par['sesw {0}'.format(i)], par['secw {0}'.format(i)]) if 'secw {0}'.format(i) in par else np.full(shape, np.pi/2)

    k = np.exp(par['log(K) {0}'.format(i)])

    return np.array([p, tc, k, e, w])


def set_gp_params(rho, sigma, t, ferr, gp: GaussianProcess):

    gp.kernel = terms.SHOTerm(rho = rho, sigma = sigma, Q = 1/np.sqrt(2))
    gp.compute(t, yerr = ferr)

    return gp


def lnGauss(data, model, err):
    
    return - 0.5 * ( (model - data) / err)**2 - np.log( np.sqrt(2 * np.pi) * err)


def log_like(par: dict, exs: ExoSystem):

    if exs.order_a:
        logalist = []

    #check params
    for i in range(exs.n):
    
        if exs.is_transit[i] and exs.fit_transit:

            #check cosi
            if not 0 <= par['cos(i) {0}'.format(i+1)] <= 1:
                return -np.inf, [], []
            
            #check ror
            if not 0 <= par['ror {0}'.format(i+1)] <= 1:
                return -np.inf, [], []
            
            if exs.order_a:
                logalist.append(par['log(a/rs) {0}'.format(i+1)])
            
            
        if exs.fit_ecc[i]:

            #check e
            if par['secw {0}'.format(i+1)]**2 + par['sesw {0}'.format(i+1)]**2 > 0.9:
                return -np.inf, [], []
            

    #check a order
    if exs.fit_transit and exs.order_a:
        logadiff = np.diff(np.array(logalist)[exs.transitsortorder])
        if np.any(logadiff <= 0):
            return -np.inf, [], []


    #check ld
    if exs.fit_transit and exs.fit_ld:
        if not 0 <= par['u1'] <= 1 or not 0 <= par['u2'] <= 1:
            return -np.inf, [], []
        


    like = 0

    if exs.fit_star:

        #check feh
        if not -0.5 <= par['feh'] <= 0.5:
            return -np.inf, [], []
        
        #check AV
        if par['AV'] < 0:
            return -np.inf, [], []

        starlike = exs.starmod.lnlike([par['eep'],par['age'],par['feh'],par['distance'],par['AV']])
        starlike += exs.starmod.lnprior([par['eep'],par['age'],par['feh'],par['distance'],par['AV']])

        if np.isnan(starlike):
            return -np.inf, [], []
        
        like += starlike

        rstar, mstar, Tstar, loggstar = exs.misti.interp_value([par['eep'],par['age'],par['feh']],['radius','mass','Teff','logg'])

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

                    fm += lightCurve(tpars[k], exs.tt[i], ld, exs.exptimes[i], exs.supersamples[i])

            if exs.detrend[i]:

                resid = exs.f[i] - fm

                try:

                    gp = set_gp_params(np.exp(par['log(rho_gp) {0}'.format(exs.lcnames[i])]), np.exp(par['log(sigma_gp) {0}'.format(exs.lcnames[i])]), exs.tt[i], exs.ferr[i], exs.gps[i])

                except:

                    return -np.inf, [], []

                like += gp.log_likelihood(resid)

            else:

                like += np.sum(lnGauss(exs.f[i], fm, exs.ferr[i]))

    if exs.fit_rv:

        rvm = np.array([par['trend 0']]*len(exs.tr)) + (par['trend 1'] * (exs.tr - exs.tr_ref) if exs.rv_bkg_order > 0 else 0) + (par['trend 2'] * (exs.tr - exs.tr_ref)**2 if exs.rv_bkg_order > 1 else 0)

        for i in np.unique(exs.which_rv)[1:]:

            j = np.where(exs.which_rv == i)[0]

            rvm[j] += par['offset {0}'.format(exs.rvnames[i])]

        for i in range(exs.n):

            if not exs.is_rv[i]:
                continue

            j = np.sum(exs.is_rv[:i])

            rvm += rvModel(rpars[j], exs.tr)


        like += np.sum(lnGauss(exs.rv, rvm, exs.rverr))



    return like if not np.isnan(like) else -np.inf, np.array(ps), np.array(tcs)


def log_like_staronly(par: dict, exs: ExoSystem):

    #check feh
    if not -0.5 <= par['feh'] <= 0.5:
        return -np.inf
    
    #check AV
    if par['AV'] < 0:
        return -np.inf

    starlike = exs.starmod.lnlike([par['eep'],par['age'],par['feh'],par['distance'],par['AV']])
    starlike += exs.starmod.lnprior([par['eep'],par['age'],par['feh'],par['distance'],par['AV']])

    if np.isnan(starlike):
        return -np.inf
    
    return starlike