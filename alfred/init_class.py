import numpy as np
from astropy.table import Table, Column
import shutil
import os
from astroquery.vizier import Vizier
from astropy import units as u
from astropy.io.ascii import InconsistentTableError
from typing import Self

from alfred.ld_grids import calc_ld, ld_grid_list
from alfred.init_gui import InitPlanetsGUI, InitLcsGUI, InitRVGUI, InitPriorsGUI, InitStarGUI, InitLDGUI, InitTTVsGUI


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
                                header_rows = self.header_rows, converters = self.converters, fill_values = [('', '0', '*'),('', 'NaN', '*'),('', '', '*')])
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


    # def create(self, empty: bool = False) -> Self:
    #     """Creates an Init_lcs table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output directory
    #     and file.

    #     Args:
    #         empty (bool, optional): If true, creates the table with correct columns and one empty row filled with None and nans. Can be filled in later
    #             using Init_lcs.add_lc_file, or manually with either astropy Table methods or in the txt file. Default is False.

    #     Returns:
    #         Init_lcs: The whole Init_lcs object after creating the table.
    #     """

    #     self.table = Table(names = ['File','Nickname','Time Col','Flux Col','Err Col','Quality Col','Time Offset','Err Scale','Exp Time','Filter','Detrend'], 
    #                        units = [None,None,None,None,None,None,'BJD',None,'s',None,None],
    #                        dtype = [str,str,str,str,str,str,int,float,float,str,bool])

    #     if not empty:

    #         x = input('Creating light curve initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #         if x == 'stop':
    #             return

    #         while True:

    #             self.add_lc_file()

    #             x = input('More files? y/n ')
                
    #             if x.lower() != 'y':
    #                 break
        
    #     else:

    #         self.table.add_row(['None','None','None','None','None','None',0,np.nan,np.nan,'None',False])

    #     self.save()

    #     return self


    # def add_lc_file(self):
    #     """Adds a new row to the Init_lcs table for a new light curve file. Prompts the user to fill it in. Easier than adding rows manually.
    #     """

    #     row = []

    #     found = False
    #     while not found:

    #         fpath = input('Absolute path to light curve file to move here, or type "skip" if you dont want to move a file now: ')

    #         if fpath.lower() == 'skip':
    #             fname = input('Name of light curve file for later: ')
    #             found = True

    #         else:
                
    #             fname = fpath[fpath.rfind('/')+1:]

    #             try:
    #                 shutil.copyfile(fpath, self.direc+'/'+fname)
    #                 found = True

    #             except shutil.SameFileError:
    #                 print('File already there.')

    #             except FileNotFoundError:
    #                 print('File not found. Try again.')

    #     row.append(fname)

    #     nickname = input('Nickname for the data set (e.g. "TESS S57"): ')
    #     row.append(nickname)

    #     timecol = input('Column header for the time data: ')
    #     row.append(timecol)

    #     fcol = input('Column header for the flux data: ')
    #     row.append(fcol)

    #     ferrcol = input('Column header for the flux error data: ')
    #     row.append(ferrcol)

    #     qcol = input('Column header for the quality flag data (or None): ')
    #     row.append(qcol)

    #     toffset = input('Time offset from BJD (in days), for example 2457000 is common for TESS: ')
    #     row.append(toffset)

    #     row.append(1.0)

    #     exptime = input('Exposure time for this data set (aka its cadence) in seconds: ')
    #     row.append(exptime)

    #     filter = input('Filter or bandpass for this data (e.g. TESS, Kepler, V): ')
    #     row.append(filter)

    #     detrend = input('Detrend this lightcurve? True or False ').lower() == 'true'
    #     row.append(detrend)

    #     self.table.add_row(row)

    #     self.save()



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


    # def create(self, empty: bool = False) -> Self:
    #     """Creates an Init_rv table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output directory
    #     and file.

    #     Args:
    #         empty (bool, optional): If true, creates the table with correct columns and one empty row filled with None and nans. Can be filled in later
    #             using Init_rv.add_rv_file, or manually with either astropy Table methods or in the txt file. Default is False.

    #     Returns:
    #         Init_rv: The whole Init_rv object after creating the table.
    #     """

    #     self.table = Table(names = ['File','Nickname','Time Col','RV Col','Err Col','Time Offset','Err Scale','m/s or km/s'],
    #                        units = [None, None, None, None, None, 'BJD', None, None],
    #                        dtype = [str,str,str,str,str,float,float,str])

    #     if not empty:

    #         x = input('Creating rv initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #         if x == 'stop':
    #             return

    #         while True:

    #             self.add_rv_file()

    #             x = input('More files? y/n ')
                
    #             if x.lower() != 'y':
    #                 break

    #     else:

    #         self.table.add_row(['None','None','None','None','None',np.nan,np.nan,'None'])
        
    #     self.save()

    #     return self


    # def add_rv_file(self):
    #     """Adds a new row to the Init_rv table for a new RV data file. Prompts the user to fill it in. Easier than adding rows manually.
    #     """

    #     row = []

    #     found = False
    #     while not found:

    #         fpath = input('Absolute path to RV file to move here, or type "skip" if you dont want to move a file now: ')

    #         if fpath.lower() == 'skip':
    #             fname = input('Name of RV file for later: ')
    #             found = True

    #         else:

    #             fname = fpath[fpath.rfind('/')+1:]

    #             try:
    #                 shutil.copyfile(fpath, self.direc+'/'+fname)
    #                 found = True

    #             except shutil.SameFileError:
    #                 print('File already there.')

    #             except FileNotFoundError:
    #                 print('File not found. Try again.')

    #     row.append(fname)

    #     nickname = input('Nickname for the data set (e.g. "NEID"): ')
    #     row.append(nickname)

    #     timecol = input('Column header for the time data: ')
    #     row.append(timecol)

    #     rvcol = input('Column header for the RV data: ')
    #     row.append(rvcol)

    #     rverrcol = input('Column header for the RV error data: ')
    #     row.append(rverrcol)

    #     toffset = input('Time offset from BJD (in days), for example 2457000 is common for TESS: ')
    #     row.append(toffset)

    #     row.append(1.0)

    #     mskms = input('Are the RV and RV error data in "m/s" or "km/s": ')
    #     while mskms not in ['m/s','km/s']:
    #         mskms = input('Please type "m/s" or "km/s" exactly: ')
    #     row.append(mskms)

    #     self.table.add_row(row)

    #     self.save()


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


    # def create(self) -> Self:
    #     """Creates an Init_star table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output directory
    #     and file. There is a prompt option to get some parameters automatically with astroquery, assuming vizier isn't down.

    #     Returns:
    #         Init_star: The whole Init_star object after creating the table.
    #     """

    #     x = input('Creating star initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #     if x == 'stop':
    #         return

    #     rows = [['Value'],['Error']]

    #     r = input('Stellar radius and error (in Rsun), separated by a space: ').split()
    #     rows[0].append(r[0])
    #     rows[1].append(r[1])

    #     m = input('Stellar mass and err (in Msun), separated by a space: ').split()
    #     rows[0].append(m[0])
    #     rows[1].append(m[1])

    #     T = input('Stellar temperature and err (in K), separated by a space: ').split()
    #     rows[0].append(T[0])
    #     rows[1].append(T[1])

    #     logg = input('Stellar log(g) and err (in cgs), separated by a space (or "nan nan" if unavailable): ').split()
    #     rows[0].append(logg[0])
    #     rows[1].append(logg[1])

    #     feh = input('Stellar Fe/H and err, separated by a space (or "nan nan" if unavailable): ').split()
    #     rows[0].append(feh[0])
    #     rows[1].append(feh[1])

    #     auto = input('Get parallax and magnitudes using astroquery? y/n').lower()

    #     if auto == 'y':

    #         try:

    #             cid = input('Catalog ID (TIC, Gaia, HIP, etc): ')

    #             vizier = Vizier(columns = ['**','+_r'])

    #             gaia = vizier.query_object(object_name = cid, catalog = 'I/355/gaiadr3', radius = 1*u.arcmin)[0][0]

    #             twomass = vizier.query_object(object_name = cid, catalog = 'II/246/out', radius = 1*u.arcmin)[0][0]

    #             wise = vizier.query_object(object_name = cid, catalog = 'II/328/allwise', radius = 1*u.arcmin)[0][0]

    #         except:

    #             auto == 'n'
    #             print('Issue accessing astroquery, switching back to manual input.')

    #     if auto == 'y':

    #         rows[0].append(gaia['Plx'])
    #         rows[1].append(gaia['e_Plx'])

    #         rows[0].append(twomass['Jmag'])
    #         rows[1].append(twomass['e_Jmag'])

    #         rows[0].append(twomass['Hmag'])
    #         rows[1].append(twomass['e_Hmag'])

    #         rows[0].append(twomass['Kmag'])
    #         rows[1].append(twomass['e_Kmag'])

    #         rows[0].append(gaia['Gmag'])
    #         rows[1].append(gaia['e_Gmag'])

    #         rows[0].append(gaia['BPmag'])
    #         rows[1].append(gaia['e_BPmag'])

    #         rows[0].append(gaia['RPmag'])
    #         rows[1].append(gaia['e_RPmag'])

    #         rows[0].append(wise['W1mag'])
    #         rows[1].append(wise['e_W1mag'])

    #         rows[0].append(wise['W2mag'])
    #         rows[1].append(wise['e_W2mag'])

    #         rows[0].append(wise['W3mag'])
    #         rows[1].append(wise['e_W3mag'])

    #     else:

    #         P = input('Gaia parallax and error (in mas), separated by a space: ').split()
    #         rows[0].append(P[0])
    #         rows[1].append(P[1])

    #         J = input('2MASS J band magnitude and err, separated by a space: ').split()
    #         rows[0].append(J[0])
    #         rows[1].append(J[1])

    #         H = input('2MASS H band magnitude and err, separated by a space: ').split()
    #         rows[0].append(H[0])
    #         rows[1].append(H[1])

    #         K = input('2MASS K band magnitude and err, separated by a space: ').split()
    #         rows[0].append(K[0])
    #         rows[1].append(K[1])

    #         G = input('Gaia G band magnitude and err, separated by a space: ').split()
    #         rows[0].append(G[0])
    #         rows[1].append(G[1])

    #         Bp = input('Gaia Bp band magnitude and err, separated by a space: ').split()
    #         rows[0].append(Bp[0])
    #         rows[1].append(Bp[1])

    #         Rp = input('Gaia Rp band magnitude and err, separated by a space: ').split()
    #         rows[0].append(Rp[0])
    #         rows[1].append(Rp[1])

    #         W1 = input('WISE W1 band magnitude and err, separated by a space: ').split()
    #         rows[0].append(W1[0])
    #         rows[1].append(W1[1])

    #         W2 = input('WISE W2 band magnitude and err, separated by a space: ').split()
    #         rows[0].append(W2[0])
    #         rows[1].append(W2[1])

    #         W3 = input('WISE W3 band magnitude and err, separated by a space: ').split()
    #         rows[0].append(W3[0])
    #         rows[1].append(W3[1])


    #     self.table = Table(rows = rows,
    #             names = ['Val/Err','Radius','Mass','Teff','log(g)','Fe/H','Parallax','J','H','K','G','Bp','Rp','W1','W2','W3'],
    #             units = [None,'Rsun','Msun','K','cgs','dex','mas','mag','mag','mag','mag','mag','mag','mag','mag','mag'],
    #             dtype = [str,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float])
        
    #     if auto == 'y':

    #         tabformat = {'Parallax': '%.4f','J': '%.3f','H': '%.3f','K': '%.3f','G': '%.6f','Bp': '%.6f','Rp': '%.6f','W1': '%.3f','W2': '%.3f','W3': '%.3f'}

    #         for x in tabformat:
    #             self.table[x].format = tabformat[x]
        
    #     self.save()

    #     return self

    
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


    # def create(self, empty: bool = False) -> Self:
    #     """Creates an Init_ld table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output directory
    #     and file.

    #     Args:
    #         empty (bool, optional): If true, creates the table with correct columns and one empty row filled with None and nans. Can be filled in later
    #             using Init_ld.add_ld_params, or manually with either astropy Table methods or in the txt file. Default is False.

    #     Returns:
    #         Init_ld: The whole Init_ld object after creating the table.
    #     """

    #     self.table = Table(names = ['Filter','u1','u2'], dtype = [str,float,float])

    #     if not empty:

    #         x = input('Creating limb darkening initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #         if x == 'stop':
    #             return

    #         while True:

    #             self.add_ld_params()

    #             x = input('More filters? y/n ')
                
    #             if x.lower() != 'y':
    #                 break

    #     else:

    #         self.table.add_row([np.nan,np.nan,'None'])
        
    #     self.save()

    #     return self


    # def add_ld_params(self):
    #     """Adds a set of limb darkening parameters to the Init_ld table for a new filter. Prompts the user to fill it in. Easier than adding rows
    #     manually. There is a prompt to automatically generate the parameters from existing grids given the stellar parameters. By default, the only
    #     grids included are for the TESS filter and the Kepler filter, but more grids can be generated with alfred.generate_ld_grid if the user has
    #     exoctk installed.
    #     """

    #     autogen = input('Auto-generate quadratic limb darkening parameters from existing grids? y/n').lower()

    #     if autogen == 'y':

    #         while True:

    #             filt = input('Filter to generate the limb darkening parameter for? Type "help" to see available filters. Type "stop" to exit.')

    #             if filt.lower() == 'help':

    #                 print(ld_grid_list)

    #             elif filt.lower() == 'stop':
    #                 return

    #             elif filt not in ld_grid_list:

    #                 z = input('Filter is not recognized. If you have exoctk installed, you can make a grid for this filter using alfred.generate_ld_grid. Enter to try again. Type "stop" to exit.')
    #                 if z.lower() == 'stop':
    #                     return

    #             else:
                    
    #                 break

    #         T = float(input('Stellar temperature in K:'))
    #         logg = float(input('Stellar log g in cgs units:'))
    #         feh = float(input('Stellar [Fe/H]:'))

    #         u1, u2 = calc_ld(filt, T, logg, feh)

    #         row = [u1, u2, filt]

    #         self.table.add_row(row)

    #     else:

    #         row = []

    #         filt = input('Filter to define limb darkening parameters for (e.g. TESS, Kepler, V): ')

    #         u1 = input('First quadratic limb darkening parameter for {0} (linear term): '.format(filt))
    #         row.append(u1)

    #         u2 = input('Second quadratic limb darkening parameter for {0} (non-linear term): '.format(filt))
    #         row.append(u2)

    #         row.append(filt)

    #         self.table.add_row(row)

    #     self.save()


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

        self.converters = {'Transiting':bool, 'RV Signal':bool,  'Fit Ecc':bool, 'Fit TTVs':bool, 'Fit Eclipse':bool, 'Period':float, 'Tc':float, 'Rp/Rs':float, 'a/Rs':float, 'cos(i)':float, 'K':float, 'sqrt(e)cos(w)':float, 'sqrt(e)sin(w)':float, 'fp':float}

        self.table = Table(rows = [[False]*5 + [np.nan]*9],
            names = ['Transiting', 'RV Signal',  'Fit Ecc', 'Fit TTVs', 'Fit Eclipse', 'Period', 'Tc', 'Rp/Rs', 'a/Rs', 'cos(i)', 'K', 'sqrt(e)cos(w)', 'sqrt(e)sin(w)', 'fp'],
            units = [None, None, None, None, None, 'days', 'BJD-2450000', None, None, None, 'm/s', None, None, None],
            dtype = [bool, bool, bool, bool, bool, float, float, float, float, float, float, float, float, float])


    def __call__(self):
        """Starts the InitPlanetsGUI.
        """

        app = InitPlanetsGUI(self)
        app.mainloop()


    # def create(self) -> Self:
    #     """Creates an Init_planets table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output
    #     directory and file.

    #     Returns:
    #         Init_planets: The whole Init_planets object after creating the table.
    #     """

    #     self.table = Table(names = ['Transiting', 'RV Signal', 'Fit TTVs', 'Fit Eclipse', 'Fit Ecc', 'Period', 'Tc', 'Rp/Rs', 'a/Rs', 'cos(i)', 'K', 'sqrt(e)cos(w)', 'sqrt(e)sin(w)', 'fp'],
    #         units = [None, None, None, None, None, 'days', 'BJD-2450000', None, None, None, 'm/s', None, None, None],
    #         dtype = [bool, bool, bool, bool, bool, float, float, float, float, float, float, float, float, float])

    #     x = input('Creating planet initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #     if x == 'stop':
    #         return

    #     while True:

    #         self.add_planet()

    #         x = input('More planets? y/n ')
                
    #         if x.lower() != 'y':
    #             break

    #     self.save()

    #     return self


    # def add_planet(self):
    #     """Adds a new row to the Init_planets table for a new planet. Prompts the user to fill it in. Easier than adding rows manually.
    #     """

    #     row = []

    #     transit = input('Is the planet transiting? True or False ').lower() == 'true'
    #     row.append(transit)

    #     rv = input('Does the planet have an RV signal? True or False ').lower() == 'true'
    #     row.append(rv)

    #     if transit:

    #         ttv = input('Fit the planet for ttvs? True or False ').lower() == 'true'
    #         row.append(ttv)

    #         ecl = input('Fit the planet for a secondary eclipse? True or False ').lower() == 'true'
    #         row.append(ecl)

    #     else:

    #         row.append(False)

    #     ecc = input('Fit the planet for eccentricity? True or False ').lower() == 'true'
    #     row.append(ecc)

    #     p = input('Planet period initial guess (in days): ')
    #     row.append(p)

    #     tc = input('Planet tc initial guess (in BJD - 2450000): ')
    #     row.append(tc)

    #     if transit:

    #         rp = input('Planet to star radius ratio initial guess: ')
    #         row.append(rp)

    #         a = input('Planet semimajor axis to stellar radius ratio initial guess: ')
    #         row.append(a)

    #         cosi = input('Planet cos(inclination) initial guess: ')
    #         row.append(cosi)

    #     else:

    #         row.append(None)
    #         row.append(None)
    #         row.append(None)

    #     if rv:

    #         k = input('Planet rv semi-amplitude initial guess (in m/s): ')
    #         row.append(k)

    #     else:

    #         row.append(None)

    #     if rv and ecc:

    #         secw = input('Planet sqrt(e)cos(w) initial guess: ')
    #         row.append(secw)

    #         sesw = input('Planet sqrt(e)sin(w) initial guess: ')
    #         row.append(sesw)

    #     else:

    #         row.append(0.01)
    #         row.append(0.01)

    #     if ecl:

    #         fp = input('Planet to star flux ratio initial guess: ')
    #         row.append(fp)

    #     else:

    #         row.append(None)

    #     self.table.add_row(row)

    #     self.save()



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


    # def create(self) -> Self:
    #     """Creates an Init_ttvs table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output
    #     directory and file.

    #     Returns:
    #         Init_ttvs: The whole Init_ttvs object after creating the table.
    #     """

    #     self.table = Table()

    #     x = input('Creating ttv initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #     if x == 'stop':
    #         return

    #     while True:

    #         self.add_transit_times()

    #         x = input('More TTV planets? y/n ')
            
    #         if x.lower() != 'y':
    #             break
        
    #     self.save()
        
    #     return self


    # def add_transit_times(self):
    #     """Adds a new column to the Init_ttvs table for a new planet's TTVs. Prompts the user to fill it in. Recommended to only use this function
    #     rather than adding columns manually, because any existing columns must all be adjusted to the same length.
    #     """

    #     col = []

    #     num = input('Planet number: ')

    #     n = int(input('Number of transit times to input for planet {0}: '.format(num)))

    #     for i in range(1, n+1):

    #         tt = input('Transit time {0} for planet {1} (BJD-2450000): '.format(i, num))
    #         col.append(tt)

    #     if len(self.table.columns) > 0:

    #         if n > len(self.table):
                
    #             for i in range(n-len(self.table)):
    #                 self.table.add_row([np.nan]*len(self.table.columns))

    #         elif n < len(self.table):

    #             col += [np.nan]*(len(self.table)-n)

    #     col = Column(col, dtype = float)
    #     self.table.add_column(col, name = num)

    #     self.save()


class Init_priors(InitFile):
    """Class for creating, loading, and editing prior initialization files. Inherits from InitFile. Contains any priors the user wishes to apply to the
    fit. Currently supported priors are Gaussian priors, uniform priors (essentially just hard upper and lower boundaries), fixed priors (fixing a
    parameter at a specific value so it is not fit), Jeffrey's priors (uninformative prior independent of scale), and modified Jeffrey's priors
    (becomes log uniform below the knee value). Priors can be set for a specific parameter (e.g. P 1, the period of planet 1) or for all instances
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

        self.allowed_vars = ['log(P)', 'P', 'Tc', 'ror', 'log(a/rs)', 'a/rs', 'rhos', 'cos(i)', 'i', 'log(K)', 'K', 'secw', 'sesw', 'e', 'w', 'TT', 'fp',
                             'F0', 'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp', 'u1', 'u2', 'gamma', 'gamma_dot', 'gamma_ddot', 'rv_offset',
                             'eep', 'log10(age)', 'age', 'feh', 'distance', 'AV', 'mstar', 'rstar', 'rhostar']
        
        self.planet_vars = ['log(P)', 'P', 'Tc', 'ror', 'log(a/rs)', 'a/rs', 'rhos', 'cos(i)', 'i', 'log(K)', 'K', 'secw', 'sesw', 'e', 'w', 'TT', 'fp']
        self.lc_vars = ['F0', 'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp']
        self.ld_vars = ['u1', 'u2']
        
        self.var_descriptions = ['natural log of the planet period in days',
                                 'planet period in days',
                                 'planet time of conjunction in BJD - 2450000',
                                 'planet radius to stellar radius ratio',
                                 'natural log of planet semi-major axis to stellar radius ratio',
                                 'planet semi-major axis to stellar radius ratio',
                                 'implied stellar density from planet transit in g/cm^3',
                                 'cosine of the planet orbital inclination',
                                 'planet orbital inclination in radians',
                                 'natural log of the planet RV semi-amplitude in m/s',
                                 'planet RV semi-amplitude in m/s',
                                 'square root of planet eccentricity times cosine of planet argument of periastron',
                                 'square root of planet eccentricity times sine of planet argument of periastron',
                                 'planet eccentricity',
                                 'planet argument of periastron in radians',
                                 'individual planet transit time (for fitting TTVs) in BJD - 2450000',
                                 'planet to star flux ratio',
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
                                 'equivalent evolutionary phase, used in fitting the star with isochrones',
                                 'log10 of stellar age in yr. This is used directly in fitting',
                                 'stellar age in yr.',
                                 'stellar metallicity in dex',
                                 'distance to the system in pc',
                                 'V band extinction in mags',
                                 'stellar mass in solar masses',
                                 'stellar radius in solar radii',
                                 'mean stellar density in g/cm^3 in stellar fitting']
        

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

    
    # def create(self) -> Self:
    #     """Creates an Init_priors table with correct formatting and columns. Prompts the user to help fill it in. Saves the table to the output directory
    #     and file.

    #     Returns:
    #         Init_priors: The whole Init_priors object after creating the table.
    #     """

    #     self.table = Table(names = ['Variable', 'Prior Type', 'Param 1', 'Param 2'],
    #         dtype = [str, str, float, float])

    #     x = input('Creating prior initialization file {0} in {1}. If this was a mistake, type "stop". Otherwise, enter to continue.'.format(self.name, self.direc)).lower()

    #     if x == 'stop':
    #         return

    #     while True:

    #         self.add_prior()

    #         x = input('More priors? y/n ')
                
    #         if x.lower() != 'y':
    #             break

    #     self.save()

    #     return self
    

    # def add_prior(self):
    #     """Adds a new row to the Init_priors table for a new prior. Prompts the user to fill it in. Easier than adding rows manually. Currently
    #     supported priors are Gaussian priors, uniform priors (essentially just hard upper and lower boundaries), fixed priors (fixing a parameter at
    #     a specific value so it is not fit), Jeffrey's priors (uninformative prior independent of scale), and modified Jeffrey's priors (becomes
    #     log uniform below the knee value). Priors can be set for a specific parameter (e.g. P 1, the period of planet 1) or for all instances of a
    #     type of parameter in the fit (e.g. P x, the periods of all planets).
    #     """

    #     row = []

    #     while True:

    #         var = input('Variable name. Type "help" to see a list of variables. Type "stop" to exit.')

    #         if var.lower() == 'help':

    #             self.var_help()

    #         elif var.lower() == 'stop':
    #             return

    #         elif var not in self.allowed_vars:

    #             z = input('Variable is not recognized. Enter to try again. Type "stop" to exit.')
    #             if z.lower() == 'stop':
    #                 return

    #         else:
                
    #             break

    #     origvar = var
        
    #     if var in self.planet_vars:

    #         num = input('Planet number to apply to apply this prior to (planets are 1-indexed starting at top of init_planets file).' + ('' if var == 'TT' else ' Enter x to apply this prior to all planets.'))

    #         if var == 'TT':

    #             tnum = input('Transit number of planet {0} to apply this prior to (1-indexed in time order).'.format(num))

    #             num += ' '
    #             num += tnum

    #         var += ' '
    #         var += num

    #     elif var in self.lc_vars:

    #         lc = input('Light curve nickname to apply this prior to. Enter x to apply this prior to all light curves.')

    #         var += ' '
    #         var += lc

    #     elif var in self.ld_vars:

    #         filt = input('Filter to apply this prior to. Enter x to apply this prior to all filters.')

    #         var += ' '
    #         var += filt

    #     elif var == 'rv_offset':

    #         rv = input('RV dataset nickname to apply this prior to. Enter x to apply this prior to all RV datasets.')

    #         var += ' '
    #         var += rv

    #     row.append(var)

    #     while True:
            
    #         priortype = input("Prior type to use for {0}. U for uniform, G for Gaussian, F for fixed, J for Jeffrey's, or MJ for modified Jeffrey's.".format(var)).upper()

    #         if priortype not in ['U', 'G', 'F', 'J', 'MJ']:

    #             z = input('Prior type is not recognized. Enter to try again. Type "stop" to exit.')
    #             if z.lower() == 'stop':
    #                 return
                
    #         elif priortype == 'F' and origvar in ['rhos','u1','u2','mstar','rstar','rhostar']:

    #             z = input('Cannot set {0} to fixed. Enter to try again. Type "stop" to exit.'.format(origvar))
    #             if z.lower() == 'stop':
    #                 return

    #         else:

    #             break

    #     row.append(priortype)

    #     if priortype == 'U':

    #         p1 = input('Lower bound of uniform prior for {0}.'.format(var))
    #         p2 = input('Upper bound of uniform prior for {0}.'.format(var))

    #     elif priortype == 'G':

    #         p1 = input('Mean of Gaussian prior for {0}.'.format(var))
    #         p2 = input('Standard deviation of Gaussian prior for {0}.'.format(var))

    #     elif priortype == 'F':

    #         p1 = input('Value at which to fix {0}.'.format(var))
    #         p2 = 0

    #     elif priortype == 'J':

    #         p1 = input("Lower bound of Jeffrey's prior for {0}.".format(var))
    #         p2 = input("Upper bound of Jeffrey's prior for {0}.".format(var))

    #     elif priortype == 'MJ':

    #         p1 = input("Upper bound of modified Jeffrey's prior for {0}.".format(var))
    #         p2 = input("Knee value of modified Jeffrey's prior for {0}.".format(var))

    #     row.append(p1)
    #     row.append(p2)

    #     self.table.add_row(row)

    #     self.save()




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

    initstar = Init_star(direc)
    initstar()
    initstar.save()

    initplanets = Init_planets(direc)
    initplanets()
    initplanets.save()

