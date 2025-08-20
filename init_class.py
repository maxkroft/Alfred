import numpy as np
from astropy.table import Table
import shutil
import os

class InitFile:

    def __init__(self, direc, name):

        self.direc = direc
        self.name = name
    
    def from_file(self):

        self.table = Table.read(self.direc + '/' + self.name, format = 'ascii.fixed_width_two_line', delimiter = '|',
                                header_rows = self.header_rows, converters = {'*': [int, float, bool, str]})

    def save(self):

        self.table.write(self.direc +'/' + self.name, format = 'ascii.fixed_width_two_line', overwrite = True, delimiter = '|',
                         delimiter_pad = ' ', bookend = True, header_rows = self.header_rows)
        
    def rename(self, newname):

        self.name = newname



class Init_lcs(InitFile):

    def __init__(self, direc, name = 'init_lcs.txt'):

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']


    def create(self, empty = False):

        self.table = Table(names = ['File','Nickname','Time Col','Flux Col','Err Col','Quality Col','Time Offset','Err Scale','Exp Time','Filter'], 
                           units = [None,None,None,None,None,None,'BJD',None,'s',None],
                           dtype = [str,str,str,str,str,str,int,float,float,str])

        if not empty:

            input('Creating light curve initialization file {0} in {1}. If this was a mistake, press esc. Otherwise, enter to continue.'.format(self.name, self.direc))

            while True:

                self.add_lc_file()

                x = input('More files? y/n ')
                
                if x.lower() != 'y':
                    break
        
        self.save()


    def add_lc_file(self):

        row = []

        fpath = input('Absolute path to lightcurve file to move here: ')
        fname = fpath[fpath.rfind('/')+1:]
        row.append(fname)
        shutil.copyfile(fpath, dir+'/'+fname)

        nickname = input('Nickname for the data set (e.g. "TESS S57"): ')
        row.append(nickname)

        timecol = input('Column header for the time data: ')
        row.append(timecol)

        fcol = input('Column header for the flux data: ')
        row.append(fcol)

        ferrcol = input('Column header for the flux error data: ')
        row.append(ferrcol)

        qcol = input('Column header for the quality flag data (or None): ')
        row.append(qcol)

        toffset = input('Time offset from BJD (in days), for example 2457000 is common for TESS: ')
        row.append(toffset)

        row.append(1.0)

        exptime = input('Exposure time for this data set (aka its cadence) in seconds: ')
        row.append(exptime)

        filter = input('Filter or bandpass for this data (e.g. TESS, Kepler, V): ')
        row.append(filter)

        self.table.add_row(row)


class Init_rv(InitFile):

    def __init__(self, direc, name = 'init_rv.txt'):

        super().__init__(direc, name)

        self.header_rows = ['name']


    def create(self, empty = False):

        self.table = Table(names = ['File','Nickname','Time Col','RV Col','Err Col','Err Scale','m/s or km/s'],
                           dtype = [str,str,str,str,str,float,str])

        if not empty:

            input('Creating rv initialization file {0} in {1}. If this was a mistake, press esc. Otherwise, enter to continue.'.format(self.name, self.direc))

            while True:

                self.add_rv_file()

                x = input('More files? y/n ')
                
                if x.lower() != 'y':
                    break
        
        self.save()


    def add_rv_file(self):

        row = []

        fpath = input('Absolute path to RV file to move here: ')
        fname = fpath[fpath.rfind('/')+1:]
        row.append(fname)
        shutil.copyfile(fpath, dir+'/'+fname)

        nickname = input('Nickname for the data set (e.g. "NEID"): ')
        row.append(nickname)

        timecol = input('Column header for the time data: ')
        row.append(timecol)

        rvcol = input('Column header for the RV data: ')
        row.append(rvcol)

        rverrcol = input('Column header for the RV error data: ')
        row.append(rverrcol)

        row.append(1.0)

        mskms = input('Are the RV and RV error data in "m/s" or "km/s": ')
        while mskms not in ['m/s','km/s']:
            mskms = input('Please type "m/s" or "km/s" exactly: ')
        row.append(mskms)

        self.table.add_row(row)


class Init_star(InitFile):

    def __init__(self, direc, name = 'init_star.txt'):

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']


    def create(self):

        input('Creating star initialization file {0} in {1}. If this was a mistake, press esc. Otherwise, enter to continue.'.format(self.name, self.direc))

        rows = [['Value'],['Error']]

        r = input('Stellar radius and error (in Rsun), separated by a space: ').split()
        rows[0].append(r[0])
        rows[1].append(r[1])

        m = input('Stellar mass and err (in Msun), separated by a space: ').split()
        rows[0].append(m[0])
        rows[1].append(m[1])

        T = input('Stellar temperature and err (in K), separated by a space: ').split()
        rows[0].append(T[0])
        rows[1].append(T[1])

        P = input('Gaia parallax and error (in mas), separated by a space: ').split()
        rows[0].append(P[0])
        rows[1].append(P[1])

        logg = input('Stellar log(g) and err (in cgs), separated by a space (or "nan nan" if unavailable): ').split()
        rows[0].append(logg[0])
        rows[1].append(logg[1])

        feh = input('Stellar Fe/H and err, separated by a space (or "nan nan" if unavailable): ').split()
        rows[0].append(feh[0])
        rows[1].append(feh[1])

        J = input('2MASS J band magnitude and err, separated by a space: ').split()
        rows[0].append(J[0])
        rows[1].append(J[1])

        H = input('2MASS H band magnitude and err, separated by a space: ').split()
        rows[0].append(H[0])
        rows[1].append(H[1])

        K = input('2MASS K band magnitude and err, separated by a space: ').split()
        rows[0].append(K[0])
        rows[1].append(K[1])

        G = input('Gaia G band magnitude and err, separated by a space: ').split()
        rows[0].append(G[0])
        rows[1].append(G[1])

        Bp = input('Gaia Bp band magnitude and err, separated by a space: ').split()
        rows[0].append(Bp[0])
        rows[1].append(Bp[1])

        Rp = input('Gaia Rp band magnitude and err, separated by a space: ').split()
        rows[0].append(Rp[0])
        rows[1].append(Rp[1])

        W1 = input('WISE W1 band magnitude and err, separated by a space: ').split()
        rows[0].append(W1[0])
        rows[1].append(W1[1])

        W2 = input('WISE W2 band magnitude and err, separated by a space: ').split()
        rows[0].append(W2[0])
        rows[1].append(W2[1])

        W3 = input('WISE W3 band magnitude and err, separated by a space: ').split()
        rows[0].append(W3[0])
        rows[1].append(W3[1])

        self.table = Table(rows = rows,
                names = ['Val/Err','Radius','Mass','Teff','Parallax','log(g)','Fe/H','J','H','K','G','Bp','Rp','W1','W2','W3'],
                units = [None,'Rsun','Msun','K','mas','cgs','dex','mag','mag','mag','mag','mag','mag','mag','mag','mag'],
                dtype = [str,float,float,int,float,float,float,float,float,float,float,float,float,float,float,float])
        
        self.save()

    
class Init_ld(InitFile):

    def __init__(self, direc, name = 'init_ld.txt'):

        super().__init__(direc, name)

        self.header_rows = ['name']


    def create(self, empty = False):

        self.table = Table(names = ['u1','u2','Filter'], dtype = [float,float,str])

        if not empty:

            input('Creating limb darkening initialization file {0} in {1}. If this was a mistake, press esc. Otherwise, enter to continue.'.format(self.name, self.direc))

            while True:

                self.add_ld_params()

                x = input('More filters? y/n ')
                
                if x.lower() != 'y':
                    break
        
        self.save()


    def add_ld_params(self):

        row = []

        filt = input('Filter to define limb darkening parameters for (e.g. TESS, Kepler, V): ')

        u1 = input('First quadratic limb darkening parameter for {0} (linear term): '.format(filt))
        row.append(u1)

        u2 = input('Second quadratic limb darkening parameter for {0} (non-linear term): '.format(filt))
        row.append(u2)

        row.append(filt)


class Init_planets(InitFile):

    def __init__(self, direc, name = 'init_planets.txt'):

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']


    def create(self):

        self.table = Table(names = ['Transiting', 'RV Signal', 'Fit TTVs', 'Fit Ecc', 'Period', 'Tc', 'Rp/Rs', 'a/Rs', 'cos(i)', 'K', 'sqrt(e)cos(w)', 'sqrt(e)sin(w)'],
            units = [None, None, None, None, 'days', 'BJD-2450000', None, None, None, 'm/s', None, None],
            dtype = [bool, bool, bool, bool, float, float, float, float, float, float, float, float])

        input('Creating planet initialization file {0} in {1}. If this was a mistake, press esc. Otherwise, enter to continue.'.format(self.name, self.direc))

        while True:

            self.add_planet()

            x = input('More planets? y/n ')
                
            if x.lower() != 'y':
                break

        self.save()


    def add_planet(self):

        row = []

        transit = input('Is the planet transitting? True or False ').lower() == 'true'
        row.append(transit)

        rv = input('Does the planet have an RV signal? True or False ').lower() == 'true'
        row.append(rv)

        if transit:

            ttv = input('Fit the planet for ttvs? True or False ').lower() == 'true'
            row.append(ttv)

        else:

            row.append(False)

        ecc = input('Fit the planet for eccentricity? True or False ').lower() == 'true'
        row.append(ecc)

        p = input('Planet period initial guess (in days): ')
        row.append(p)

        tc = input('Planet tc initial guess (in BJD - 2450000): ')
        row.append(tc)

        if transit:

            rp = input('Planet to star radius ratio initial guess: ')
            row.append(rp)

            a = input('Planet semimajor axis to stellar radius ratio initial guess: ')
            row.append(a)

            cosi = input('Planet cos(inclination) initial guess: ')
            row.append(cosi)

        else:

            row.append(None)
            row.append(None)
            row.append(None)

        if rv:

            k = input('Planet rv semi-amplitude initial guess (in m/s): ')
            row.append(k)

        else:

            row.append(None)

        if rv and ecc:

            secw = input('Planet sqrt(e)cos(w) initial guess: ')
            row.append(secw)

            sesw = input('Planet sqrt(e)sin(w) initial guess: ')
            row.append(sesw)

        else:

            row.append(0.01)
            row.append(0.01)

        self.table.add_row(row)


class Init_ttvs(InitFile):

    def __init__(self, direc, name = 'init_ttvs.txt'):

        super().__init__(direc, name)

        self.header_rows = ['name']


    def create(self):

        self.table = Table()

        input('Creating light curve initialization file {0} in {1}. If this was a mistake, press esc. Otherwise, enter to continue.'.format(self.name, self.direc))

        while True:

            self.add_transit_times()

            x = input('More TTV planets? y/n ')
            
            if x.lower() != 'y':
                break
        
        self.save()


    def add_transit_times(self):

        col = []

        num = input('Planet number: ')

        n = int(input('Number of transit times to input for planet {0}: '.format(num)))

        for i in range(1, n+1):

            tt = input('Transit time {0} for planet {1} (BJD-2450000): '.format(i, num))
            col.append(tt)

        if n > len(self.table):

            self.table.add_rows([[np.nan]*len(self.table.columns)]*(n-len(self.table)))

        elif n < len(self.table):

            col += [np.nan]*(len(self.table)-n)

        self.table.add_column(col, name = num)



def create_folder(direc: str):
    """Creates a new folder for an exoplanet system to fit with this code. Helps you make the initialization file.

    ### Parameters
    1. dir : str
        - The absolute path to the new directory to create the folder for this system.
    """

    if os.path.exists(direc):

        x = input('Folder already exists. Continue anyway? y/n ')

        if x.lower() != 'y':
            return
        
    else:

        os.mkdir(direc)


    if not os.path.exists(direc+'/Masks'):
        os.mkdir(direc+'/Masks')

    if not os.path.exists(direc+'/Plots'):
        os.mkdir(direc+'/Plots')

    if not os.path.exists(direc+'/Results'):
        os.mkdir(direc+'/Results')

    if not os.path.exists(direc+'/Output'):
        os.mkdir(direc+'/Output')


    alt_name = input('What is the HD ID or TIC ID of the star? ')
    with open(direc+'/'+alt_name+'.txt', 'w') as file: pass

    x = input('Do you have the light curve files already? y/n ')

    if x.lower() == 'y':

        Init_lcs(direc).create()

        Init_ld(direc).create()

    else:

        Init_lcs(direc).create(empty = True)

        Init_ld(direc).create(empty = True)

        _ = input('Once you have these, run the function create_init_lcs and then create_init_ld. Press any key to continue.')


    x = input('Do you have RV files already? y/n ')

    if x.lower() == 'y':

        Init_rv(direc).create()

    else:

        Init_rv(direc).create(empty = True)

        _ = input('Once you have these, run the function create_init_rv. Press any key to continue.')

    
    Init_star(direc).create()

    Init_planets(direc).create()

