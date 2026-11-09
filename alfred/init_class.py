import numpy as np
from astropy.table import Table, Column
import shutil
import os
from astroquery.vizier import Vizier
from astropy import units as u
from astropy.io.ascii import InconsistentTableError
from typing import Self

from alfred.ld_grids import calc_ld, ld_grid_list
from alfred.init_gui import InitPlanetsGUI, InitLcsGUI, InitRVGUI, InitPriorsGUI, InitStarGUI, InitLDGUI, InitTTVsGUI, InitGaiaGUI


class InitFile:
    """Super class for the other init file classes.
    """

    def __init__(self, direc: str, name: str):
        """Initializes an InitFile object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str): The name of the init file. If it does not end in .txt, this is appended to the name.
        """

        self.direc = direc

        if len(name) < 4 or name[-4:] != '.txt':
            name += '.txt'
        self.name = name
    
    def from_file(self) -> Self:
        """Loads an an InitFile table from a previously created file. This is stored in .table.

        Returns:
            InitFile: The whole InitFile object is returned by this function.
        """

        self.table = Table.read(self.direc + '/' + self.name, format = 'ascii.fixed_width_two_line', delimiter = '|',
                                header_rows = self.header_rows, converters = self.converters)
        return self

    def save(self):
        """Saves the current table out to the directory and name of this object. Will overwrite an existing file of the same name.
        """

        self.table.write(self.direc +'/' + self.name, format = 'ascii.fixed_width_two_line', overwrite = True, delimiter = '|',
                         delimiter_pad = ' ', bookend = True, header_rows = self.header_rows)
        
    def rename(self, newname: str):
        """Renames this object to the new name. Appends .txt if it is not there.

        Args:
            newname (str): The new name for this object and its file.
        """

        if len(newname) < 4 or newname[-4:] != '.txt':
            newname += '.txt'
        self.name = newname
        self.save()

    @property
    def table(self):
        """The table storing the init file information.
        """
        return self._table
    
    @table.setter
    def table(self, tab: Table):
        self._table = tab



class Init_lcs(InitFile):
    """Class for creating, loading, and editing light curve initialization files. Inherits from InitFile. Contains information for loading light curve
    data, as well as information about each data set such as the filter or exposure time. Currently supports data in .fits, .npz, .csv, .dat, and .txt files,
    with the latter two assumed to be ascii tables with a header row. Each data set is given a nickname by the user for differentiating them during
    and after fitting.
    """

    def __init__(self, direc: str, name: str = 'init_lcs.txt'):
        """Initializes an Init_lcs object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_lcs.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']

        self.converters = {'File':str,'Nickname':str,'Time Col':str,'Flux Col':str,'Err Col':str,'Quality Col':str,'Time Offset':float,'Err Scale':float,'Exp Time':float,'Filter':str,'Detrend':bool}

        self.table = Table(rows = [['']*6 + [np.nan,1,np.nan] + [''] + [False]],
                           names = ['File','Nickname','Time Col','Flux Col','Err Col','Quality Col','Time Offset','Err Scale','Exp Time','Filter','Detrend'], 
                           units = [None,None,None,None,None,None,'BJD',None,'s',None,None],
                           dtype = [str,str,str,str,str,str,float,float,float,str,bool])

    
    def __call__(self):
        """Starts the InitLcsGUI.
        """

        app = InitLcsGUI(self)
        app.mainloop()


class Init_rv(InitFile):
    """Class for creating, loading, and editing RV data initialization files. Inherits from InitFile. Contains information for loading RV data.
    Currently supports data in .csv files or ascii tables with headers. Each data set is given a nickname by the user for differentiating them during
    and after fitting.
    """

    def __init__(self, direc: str, name: str = 'init_rv.txt'):
        """Initializes an Init_rv object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_rv.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']

        self.converters = {'File':str,'Nickname':str,'Time Col':str,'RV Col':str,'Err Col':str,'Time Offset':float,'Err Scale':float,'m/s or km/s':str}

        self.table = Table(rows = [['']*5 + [np.nan,1] + ['']],
                           names = ['File','Nickname','Time Col','RV Col','Err Col','Time Offset','Err Scale','m/s or km/s'],
                           units = [None, None, None, None, None, 'BJD', None, None],
                           dtype = [str,str,str,str,str,float,float,str])


    def __call__(self):
        """Starts the InitRVGUI.
        """

        app = InitRVGUI(self)
        app.mainloop()


class Init_gaia(InitFile):
    """Class for creating, loading, and editing Gaia data initialization files. Inherits from InitFile. Contains information for loading Gaia astrometric
    data. Currently supports data in the default Gaia DR4 epoch astrometry format, or custom formatted .csv files or ascii tables with headers.
    """

    def __init__(self, direc: str, name: str = 'init_gaia.txt'):
        """Initializes an Init_gaia object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_gaia.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']

        self.converters = {'File':str,'Format':str,'Source ID':int,'Time Col':str,'AL Col':str,'Err Col':str,'Psi Col':str,'Plx Fac Col':str,'Err Scale':float,'RA':float,'Dec':float,'Ref Time':float}

        self.table = Table(rows = [['']*2 + [0] + ['']*5 + [1] + [np.nan]*3],
                           names = ['File','Format','Source ID','Time Col','AL Col','Err Col','Psi Col','Plx Fac Col','Err Scale','RA','Dec','Ref Time'],
                           units = [None, None, None, None, None, None, None, None, None, 'deg', 'deg', 'BJD'],
                           dtype = [str,str,int,str,str,str,str,str,float,float,float,float])


    def __call__(self):
        """Starts the InitGaiaGUI.
        """

        app = InitGaiaGUI(self)
        app.mainloop()


class Init_star(InitFile):
    """Class for creating, loading, and editing stellar parameter initialization files. Inherits from InitFile. Contains physical stellar parameters,
    such as the effective temperature, and measured parameters, such as the parallax or various photometric values. All of these are also supplied with
    uncertainties.
    """

    def __init__(self, direc: str, name: str = 'init_star.txt'):
        """Initializes an Init_star object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_star.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name']

        self.converters = {'Parameter':str,'Units':str,'Value':float,'Error':float}

        blank_data = {'Parameter': ['Radius','Mass','Teff','log(g)','Fe/H','Parallax'],
                      'Units': ['Rsun','Msun','K','cgs','dex','mas'],
                      'Value': [np.nan]*6,
                      'Error': [np.nan]*6}

        self.table = Table(blank_data, dtype = [str, str, float, float])


    def __call__(self):
        """Starts the InitStarGUI.
        """

        app = InitStarGUI(self)
        app.mainloop()


    def from_file(self) -> Self:
        """Loads an an Init_star table from a previously created file. This is stored in .table. If the load fails, trys again using the deprecated
        formatting and converts it to the current format.

        Returns:
            Init_star: The whole Init_star object is returned by this function.
        """

        try:
            self.table = Table.read(self.direc + '/' + self.name, format = 'ascii.fixed_width_two_line', delimiter = '|',
                                    header_rows = self.header_rows, converters = {'*': [float, bool, str]})

        except InconsistentTableError:

            table_old = Table.read(self.direc + '/' + self.name, format = 'ascii.fixed_width_two_line', delimiter = '|',
                                    header_rows = ['name','units'], converters = {'*': [float, bool, str]})

            self.table['Value'][0] = table_old['Radius'][0]
            self.table['Error'][0] = table_old['Radius'][1]

            self.table['Value'][1] = table_old['Mass'][0]
            self.table['Error'][1] = table_old['Mass'][1]

            self.table['Value'][2] = table_old['Teff'][0]
            self.table['Error'][2] = table_old['Teff'][1]

            self.table['Value'][3] = table_old['log(g)'][0]
            self.table['Error'][3] = table_old['log(g)'][1]

            self.table['Value'][4] = table_old['Fe/H'][0]
            self.table['Error'][4] = table_old['Fe/H'][1]

            self.table['Value'][5] = table_old['Parallax'][0]
            self.table['Error'][5] = table_old['Parallax'][1]

            for colname in table_old.colnames:

                if colname in ['G','Bp','Rp','J','H','K','W1','W2','W3']:

                    if colname == 'Bp':

                        self.table.add_row(['BP','mag',table_old['Bp'][0],table_old['Bp'][1]])

                    elif colname == 'Rp':

                        self.table.add_row(['RP','mag',table_old['Rp'][0],table_old['Rp'][1]])

                    else:

                        self.table.add_row([colname,'mag',table_old[colname][0],table_old[colname][1]])

            self.save()

        return self

    
class Init_ld(InitFile):
    """Class for creating, loading, and editing limb darkening initialization files. Inherits from InitFile. Currently only supports quadratic limb
    darkening. Contains the two quadratic limb darkening parameters for this star in each assigned filter. These are only used when the limb darkening
    parameters are not being fit, and the stellar parameters are not being fit.
    """

    def __init__(self, direc: str, name: str = 'init_ld.txt'):
        """Initializes an Init_ld object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_ld.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name']
        
        self.converters = {'Filter':str,'u1':float,'u2':float}

        self.table = Table(rows = [[''] + [np.nan]*2], names = ['Filter','u1','u2'], dtype = [str,float,float])


    def __call__(self):
        """Starts the InitLDGUI.
        """

        app = InitLDGUI(self)
        app.mainloop()


class Init_planets(InitFile):
    """Class for creating, loading, and editing planet parameter initialization files. Inherits from InitFile. Contains information on what kinds of
    parameters to fit for the planet, as well as initial guess for the parameters. Planets are labeled during fitting as 1-indexed numbers from the
    ordering in this file.
    """

    def __init__(self, direc: str, name: str = 'init_planets.txt'):
        """Initializes an Init_planets object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_planets.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name', 'unit']

        self.converters = {'Transiting':bool, 'RV Signal':bool, 'Fit Gaia': bool, 'Fit Ecc':bool, 'Fit TTVs':bool, 'Fit Eclipse':bool, 'Period':float, 'Tc':float, 'Rp/Rs':float, 'a/Rs':float, 'cos(i)':float, 'K':float, 'sqrt(e)cos(w)':float, 'sqrt(e)sin(w)':float, 'Mp/Ms':float, 'Omega':float, 'fp':float}

        self.table = Table(rows = [[False]*6 + [np.nan]*11],
            names = ['Transiting', 'RV Signal', 'Fit Gaia', 'Fit Ecc', 'Fit TTVs', 'Fit Eclipse', 'Period', 'Tc', 'Rp/Rs', 'a/Rs', 'cos(i)', 'K', 'sqrt(e)cos(w)', 'sqrt(e)sin(w)', 'Mp/Ms', 'omega', 'fp'],
            units = [None, None, None, None, None, None, 'days', 'BJD-2450000', None, None, None, 'm/s', None, None, None, 'rad', None],
            dtype = [bool, bool, bool, bool, bool, bool, float, float, float, float, float, float, float, float, float, float, float])


    def __call__(self):
        """Starts the InitPlanetsGUI.
        """

        app = InitPlanetsGUI(self)
        app.mainloop()


class Init_ttvs(InitFile):
    """Class for creating, loading, and editing TTV initialization files. Inherits from InitFile. Contains initial estimates for all of the transit
    times for each planet which is being fit for TTVs. These will only be utilized if the planet has fit TTVs set to true in the Init_planets file.
    """

    def __init__(self, direc: str, name: str = 'init_ttvs.txt'):
        """Initializes an Init_ttvs object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_ttvs.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name','unit']

        self.converters = {'*':float}

        self.table = Table({'X': [np.nan]}, units = ['BJD-2450000'], dtype = [float])


    def __call__(self):
        """Starts the InitTTVsGUI.
        """

        app = InitTTVsGUI(self)
        app.mainloop()


    def from_file(self) -> Self:
        """Loads an an Init_ttvs table from a previously created file. This is stored in .table. If the load fails, trys again using the deprecated
        formatting and converts it to the current format.

        Returns:
            Init_star: The whole Init_ttvs object is returned by this function.
        """

        try:
            self.table = Table.read(self.direc + '/' + self.name, format = 'ascii.fixed_width_two_line', delimiter = '|',
                                    header_rows = self.header_rows, converters = {'*': [float, bool, str]})

        except InconsistentTableError:

            table_old = Table.read(self.direc + '/' + self.name, format = 'ascii.fixed_width_two_line', delimiter = '|',
                                    header_rows = ['name'], converters = {'*': [float, bool, str]})

            for col in table_old.colnames:
                table_old[col].unit = 'BJD-2450000'

            self.table = table_old

            self.save()

        return self


class Init_priors(InitFile):
    """Class for creating, loading, and editing prior initialization files. Inherits from InitFile. Contains any priors the user wishes to apply to the
    fit. Currently supported priors are Gaussian priors, uniform priors (essentially just hard upper and lower boundaries), fixed priors (fixing a
    parameter at a specific value so it is not fit), log uniform priors (uninformative prior independent of scale), and modified Jeffrey's priors
    (uniform below the knee value, log uniform above). Priors can be set for a specific parameter (e.g. P 1, the period of planet 1) or for all instances
    of a type of parameter in the fit (e.g. P x, the periods of all planets).
    """

    def __init__(self, direc: str, name: str = 'init_priors.txt'):
        """Initializes an Init_priors object with a directory and a name.

        Args:
            direc (str): A directory for alfred fitting in which to store the init file.
            name (str, optional): The name of the init file. If it does not end in .txt, this is appended to the name. The default is 'init_priors.txt'.
        """

        super().__init__(direc, name)

        self.header_rows = ['name']

        self.converters = {'Variable':str,'Prior Type':str,'Param 1':float,'Param 2':float}

        self.allowed_vars = ['log(P)', 'P', 'Tc', 'ror', 'log(a/rs)', 'a/rs', 'cos(i)', 'i', 'log(K)', 'K', 'secw', 'sesw', 'e', 'w', 'TT', 'fp',
                             'log(q)', 'q', 'sico', 'siso', 'comega', 'somega', 'omega',
                             'F0', 'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp', 'u1', 'u2', 'gamma', 'gamma_dot', 'gamma_ddot', 'rv_offset',
                             'ra*_offset', 'dec_offset', 'pmra*', 'pmdec',
                             'eep', 'log10(age)', 'age', 'feh', 'plx', 'distance', 'AV', 'mstar', 'rstar', 'log(rhostar)', 'rhostar']
        
        
        self.var_descriptions = ['natural log of the planet period in days',
                                 'planet period in days',
                                 'planet time of conjunction in BJD - 2450000',
                                 'planet to star radius ratio',
                                 'natural log of planet semi-major axis to stellar radius ratio',
                                 'planet semi-major axis to stellar radius ratio',
                                 'cosine of the planet orbital inclination',
                                 'planet orbital inclination in radians',
                                 'natural log of the planet RV semi-amplitude in m/s',
                                 'planet RV semi-amplitude in m/s',
                                 'square root of planet eccentricity times cosine of planet argument of periastron',
                                 'square root of planet eccentricity times sine of planet argument of periastron',
                                 'planet eccentricity',
                                 'planet argument of periastron in radians',
                                 'individual planet transit time (for fitting TTVs) in BJD - 2450000',
                                 'planet to star flux ratio, only used for secondary eclipse fitting',
                                 'natural log of the planet to star mass ratio',
                                 'planet to star mass ratio',
                                 'sine of inclination/2 times cosine of the longitude of the ascending node',
                                 'sine of inclination/2 times sine of the longitude of the ascending node',
                                 'cosine of the longitude of the ascending node',
                                 'sine of the longitude of the ascending node',
                                 'longitude of the ascending node in radians',
                                 'lightcurve baseline flux value',
                                 'natural log of the light curve gaussian process period in days',
                                 'light curve gaussian process period in days',
                                 'natural log of the light curve gaussian process standard deviation',
                                 'light curve gaussian process standard deviation',
                                 'quadratic limb darkening linear term',
                                 'quadratic limb darkening non-linear term',
                                 'RV systemic velocity in m/s',
                                 '1st derivative of RV systemic velocity in m/s per day, relative to time of first RV point',
                                 '2nd derivative of RV systemic velocity in m/s per day^2, relative to time of first RV point',
                                 'RV offset in m/s, for additional data sets beyond the first',
                                 'offset from reference value of right ascension times cosine of declination in mas',
                                 'offset from reference value of declination in mas',
                                 'proper motion of right ascension times cosine of declination in mas/yr',
                                 'proper motion of declination in mas/yr',
                                 'equivalent evolutionary phase, used in fitting the star with isochrones',
                                 'log10 of stellar age in yr. This is used directly in fitting',
                                 'stellar age in yr.',
                                 'stellar metallicity in dex',
                                 'parallax in mas',
                                 'distance to the system in pc',
                                 'V band extinction in mags',
                                 'stellar mass in solar masses',
                                 'stellar radius in solar radii',
                                 'natural log of the mean stellar density in g/cm^3',
                                 'mean stellar density in g/cm^3']
        

        self.table = Table(rows = [['']*2 + [np.nan]*2],
            names = ['Variable', 'Prior Type', 'Param 1', 'Param 2'],
            dtype = [str, str, float, float])
        

    def __call__(self):
        """Starts the InitPriorsGUI.
        """

        app = InitPriorsGUI(self)
        app.mainloop()
        
    
    def var_help(self):
        """Prints out all parameters which can have priors set on them, as well as descriptions of what they are and what units they should be in.
        """

        for i in range(len(self.var_descriptions)):

            print(self.allowed_vars[i], '-', self.var_descriptions[i])



def create_folder(direc: str):
    """Creates a new folder for an exoplanet system to fit with alfred. Helps you make the initialization files.

    Args:
        dir (str): The absolute path to the new directory to create the folder for this system.
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

    initlcs = Init_lcs(direc)
    initlcs()
    initlcs.save()

    initld = Init_ld(direc)
    initld()
    initld.save()

    initrv = Init_rv(direc)
    initrv()
    initrv.save()

    initgaia = Init_gaia(direc)
    initgaia()
    initgaia.save()

    initstar = Init_star(direc)
    initstar()
    initstar.save()

    initplanets = Init_planets(direc)
    initplanets()
    initplanets.save()

