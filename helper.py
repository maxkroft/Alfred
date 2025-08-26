import numpy as np
from astropy import units as u
import batman
import os
from scipy.stats import linregress
from celerite2 import GaussianProcess, terms

from alfred import ExoSystem

from ctypes import c_double, c_int, CDLL
rv_func = CDLL(os.path.dirname(os.path.realpath(__file__))+'/rv_func.so')
arr5 = c_double * 5


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

    n = len(t)
    n2 = c_int(n)

    par2 = arr5()
    for i in range(5):
        par2[i] = par[i]

    arrn = c_double * n
    t2 = arrn()
    for i in range(n):
        t2[i] = t[i]

    rv = arrn()

    rv_func.rvModel(par2, t2, n2, rv)

    return np.array(rv)


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

            fm = par['mean {0}'.format(exs.lcnames[i])]

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