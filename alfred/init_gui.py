from __future__ import annotations  # MUST be line 1

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alfred.init_class import InitFile

import customtkinter as ctk
ctk.set_appearance_mode('dark')
ctk.DrawEngine.preferred_drawing_method = "circle_shapes"

from astropy.table import Table, Row, vstack
import numpy as np


class InitGUI(ctk.CTk):
    def __init__(self, initfile: InitFile):
        super().__init__()

        self.tk.createcommand("bgerror", lambda *args: None)

        self.initfile = initfile
        self.table = self.initfile.table

        self.title(self.initfile.name)
        
        self.grid_columnconfigure((0,1), weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        self.add = ctk.CTkButton(self, text = 'Add', command = self.add_cmd, font = ('Arial', 18, 'bold'))
        self.add.grid(row = 1, column = 0, padx = 100, pady = (10, 0), sticky = 'ew', columnspan = 2)

        self.load = ctk.CTkButton(self, text = 'Load', command = self.load_cmd, font = ('Arial', 18, 'bold'))
        self.load.grid(row = 2, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', command = self.save_cmd, font = ('Arial', 18, 'bold'))
        self.save.grid(row = 2, column = 1, padx = 10, pady = (10, 0), sticky = 'ew')

        self.rename = ctk.CTkButton(self, text = 'Rename', command = self.rename_cmd, font = ('Arial', 18, 'bold'))
        self.rename.grid(row = 3, column = 0, padx = 10, pady = (10,10), sticky = 'ew')

        self.directory = ctk.CTkButton(self, text = 'Directory', command = self.direc_cmd, font = ('Arial', 18, 'bold'))
        self.directory.grid(row = 3, column  = 1, padx = 10, pady = (10,10), sticky = 'ew')


    def add_cmd(self):
        return

    def load_cmd(self):

        self.initfile.from_file()
        self.table = self.initfile.table

        self.items_frame.grid_forget()
        self.setup_items_frame()

    def setup_items_frame(self):
        return


    def save_cmd(self):

        self.initfile.save()

        self.destroy()

    def rename_cmd(self):

        rename_prompt = RenamePrompt(self.initfile.name)

        self.wait_window(rename_prompt)

        newname = rename_prompt.newname

        self.initfile.name = newname
        self.title(newname)

    
    def direc_cmd(self):

        direc_prompt = DirectoryPrompt(self.initfile.direc)

        self.wait_window(direc_prompt)

        newpath = direc_prompt.newpath

        self.initfile.direc = newpath


class DeletePrompt(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()

        self.title('Delete?')
        self.geometry('300x100')
        self.grid_columnconfigure((0,1), weight = 1)
        self.grid_rowconfigure((0,1), weight = 1)

        self.answer = False

        self.label = ctk.CTkLabel(self, text = 'Delete?', font = ('Arial', 18, 'bold'))
        self.label.grid(row = 0, column = 0, padx = 10, sticky = 'ew', columnspan = 2)

        self.yes = ctk.CTkButton(self, text = 'Yes', command = self.yes_cmd, font = ('Arial', 18, 'bold'))
        self.yes.grid(row = 1, column  = 0, padx = 10, sticky = 'ew')

        self.no = ctk.CTkButton(self, text = 'No', command = self.no_cmd, font = ('Arial', 18, 'bold'))
        self.no.grid(row = 1, column = 1, padx = 10, sticky = 'ew')

    def yes_cmd(self):

        self.answer = True

        self.destroy()

    def no_cmd(self):

        self.destroy()


class RenamePrompt(ctk.CTkToplevel):
    def __init__(self, orig_name):
        super().__init__()

        self.title('New File Name')
        self.geometry('300x100')
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure((0,1), weight = 1)

        self.newname = orig_name

        self.entry_var = ctk.StringVar(value = orig_name)
        self.entry = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = self.entry_var)
        self.entry.grid(row = 0, column = 0, padx = 10, pady = (10, 0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', font = ('Arial', 18, 'bold'), command = self.save_cmd)
        self.save.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

    def save_cmd(self):

        self.newname = self.entry_var.get().strip()
        
        if self.newname[-4:] != '.txt':
            self.newname += '.txt'

        self.destroy()


class DirectoryPrompt(ctk.CTkToplevel):
    def __init__(self, orig_path):
        super().__init__()

        self.title('Directory')
        self.geometry('300x100')
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure((0,1), weight = 1)

        self.newpath = orig_path

        self.entry_var = ctk.StringVar(value = orig_path)
        self.entry = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = self.entry_var)
        self.entry.grid(row = 0, column = 0, padx = 10, pady = (10, 0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', font = ('Arial', 18, 'bold'), command = self.save_cmd)
        self.save.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

    def save_cmd(self):

        self.newpath = self.entry_var.get().strip()
        
        if self.newpath[-1] != '/':
            self.newpath += '/'

        self.destroy()


class FloatEntryFrame(ctk.CTkFrame):
    def __init__(self, master, label, row, col, min_val = None, max_val = None, **kwargs):
        super().__init__(master, fg_color = 'transparent', **kwargs)

        self.row = row
        self.col = col
        self.min_val = min_val
        self.max_val = max_val

        self.grid_columnconfigure(0, weight = 1)

        self.label = ctk.CTkLabel(self, text = label, font = ('Arial',18,'bold'))
        self.entry = FloatEntry(self, min_val = self.min_val, max_val = self.max_val, font = ('Arial', 18), justify = 'center')

        self.label.grid(row = 0, column = 0, pady = (10,0))
        self.entry.grid(row = 1, column = 0, sticky = "ew", padx = 10)


class FloatEntry(ctk.CTkEntry):
    def __init__(self, master, min_val = None, max_val = None, **kwargs):
        super().__init__(master, **kwargs)

        self.min_val = min_val
        self.max_val = max_val

        validate_float_cmd = self.register(self.validate_float_input)

        self.configure(validate = 'key', validatecommand = (validate_float_cmd, '%P'))
        

    def validate_float_input(self, proposed_text):

        if proposed_text == "-":
            if self.min_val is not None and float(self.min_val) >= 0:
                return False
            return True

        if proposed_text in ("", ".", "-."):
            return True

        try:
            val = float(proposed_text)
            
            if self.min_val is not None and val < float(self.min_val):
                return False
                
            if self.max_val is not None and val > float(self.max_val):
                return False
                
            return True
        
        except ValueError:
            return False

    def return_float(self):
        try:
            return float(self.get().strip())
        except ValueError:
            return np.nan

    
class ToggleCheckbox(ctk.CTkCheckBox):
    def __init__(self, master, label, startval=0, toggles=[]):

        self.label = label
        self.val = ctk.IntVar(value = startval)
        self.toggles = toggles

        super().__init__(master, text = self.label, font = ('Arial', 18, 'bold'), variable = self.val, command = self.toggle_entry)
        self.toggle_entry()


    def toggle_entry(self):
        
        if self.val.get():

            for x in self.toggles:
                x.grid(row = x.row, column = x.col, sticky = "ew", padx = 10)

        else:

            for x in self.toggles:
                x.grid_forget()

    def return_bool(self):
        return self.val.get() == 1
    

######################
#####Init Planets#####
######################

class PlanetGUI(ctk.CTkToplevel):
    def __init__(self, data: Row, idx: int, close_cmd):
        super().__init__()

        self.data = data
        self.close_cmd = close_cmd

        self.title('Planet {0}'.format(idx))
        self.geometry('500x600')
        self.grid_columnconfigure((0,1), weight = 1)

        self.p_entry = FloatEntryFrame(self, "P (days)", 1, 0, min_val = 0)
        self.p_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['Period']) else self.data['Period']))
        self.p_entry.grid(row = self.p_entry.row, column = self.p_entry.col, sticky = "ew", padx = 10)

        self.t_entry = FloatEntryFrame(self, "Tc (BJD-2450000)", 1, 1)
        self.t_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['Tc']) else self.data['Tc']))
        self.t_entry.grid(row = self.t_entry.row, column = self.t_entry.col, sticky = "ew", padx = 10)

        self.r_entry = FloatEntryFrame(self, "Rp/Rstar", 2, 0, min_val = 0)
        self.r_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['Rp/Rs']) else self.data['Rp/Rs']))

        self.a_entry = FloatEntryFrame(self, "a/Rstar", 2, 1, min_val = 0)
        self.a_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['a/Rs']) else self.data['a/Rs']))

        self.i_entry = FloatEntryFrame(self, "cos(i)", 3, 0, min_val = 0, max_val = 1)
        self.i_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['cos(i)']) else self.data['cos(i)']))

        self.k_entry = FloatEntryFrame(self, "K (m/s)", 3, 1, min_val = 0)
        self.k_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['K']) else self.data['K']))

        self.secw_entry = FloatEntryFrame(self, "sqrt(e)cos(w) (-1 to 1)", 4, 0, min_val = -1, max_val = 1)
        self.secw_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['sqrt(e)cos(w)']) else self.data['sqrt(e)cos(w)']))

        self.sesw_entry = FloatEntryFrame(self, "sqrt(e)sin(w) (-1 to 1)", 4, 1, min_val = -1, max_val = 1)
        self.sesw_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['sqrt(e)sin(w)']) else self.data['sqrt(e)sin(w)']))

        self.f_entry = FloatEntryFrame(self, "Fp/Fstar", 5, 0, min_val = 0)
        self.f_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['fp']) else self.data['fp']))


        self.checkbox_frame = ctk.CTkFrame(self)
        self.checkbox_frame.grid_columnconfigure(0, weight = 1)
        self.checkbox_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew", columnspan = 2)

        self.t_check = ToggleCheckbox(self.checkbox_frame, 'Fit Transit', startval = 1 if self.data['Transiting'] else 0, toggles = [self.r_entry, self.a_entry, self.i_entry])
        self.t_check.grid(row = 0, column = 0, padx=10, pady=(10, 0), sticky="w")

        self.r_check = ToggleCheckbox(self.checkbox_frame, 'Fit RV', startval = 1 if self.data['RV Signal'] else 0, toggles = [self.k_entry])
        self.r_check.grid(row = 1, column = 0, padx=10, pady=(10, 0), sticky="w")

        self.e_check = ToggleCheckbox(self.checkbox_frame, 'Fit Eccentricity', startval = 1 if self.data['Fit Ecc'] else 0, toggles = [self.secw_entry, self.sesw_entry])
        self.e_check.grid(row = 2, column = 0, padx=10, pady=(10, 0), sticky="w")

        self.ttv_check = ToggleCheckbox(self.checkbox_frame, 'Fit TTVs', startval = 1 if self.data['Fit TTVs'] else 0)
        self.ttv_check.grid(row = 3, column = 0, padx=10, pady=(10, 0), sticky="w")

        self.se_check = ToggleCheckbox(self.checkbox_frame, 'Fit Secondary Eclipse', startval = 1 if self.data['Fit Eclipse'] else 0, toggles = [self.f_entry])
        self.se_check.grid(row = 4, column = 0, padx=10, pady=(10, 0), sticky="w")


        self.save = ctk.CTkButton(self, text = 'Save', command = self.save_cmd, font = ('Arial', 18, 'bold'))
        self.save.grid(row = 6, column = 0, padx = 10, pady = (10,0), sticky = "ew", columnspan = 2)



    def save_cmd(self):

        self.data['Transiting'] = self.t_check.return_bool()
        self.data['RV Signal'] = self.r_check.return_bool()
        self.data['Fit Ecc'] = self.e_check.return_bool()
        self.data['Fit TTVs'] = self.ttv_check.return_bool()
        self.data['Fit Eclipse'] = self.se_check.return_bool()
        self.data['Period'] = self.p_entry.entry.return_float()
        self.data['Tc'] = self.t_entry.entry.return_float()
        self.data['Rp/Rs'] = self.r_entry.entry.return_float()
        self.data['a/Rs'] = self.a_entry.entry.return_float()
        self.data['cos(i)'] = self.i_entry.entry.return_float()
        self.data['K'] = self.k_entry.entry.return_float()
        self.data['sqrt(e)cos(w)'] = self.secw_entry.entry.return_float()
        self.data['sqrt(e)sin(w)'] = self.sesw_entry.entry.return_float()
        self.data['fp'] = self.f_entry.entry.return_float()

        self.close_cmd()




class PlanetFrame(ctk.CTkFrame):
    def __init__(self, master, topmaster, data: Row, idx: int):
        super().__init__(master)

        self.grid_columnconfigure(list(range(1,4)), weight = 1)

        self.data = data
        self.topmaster = topmaster
        self.idx = idx

        self.labelvar = ctk.StringVar(value = '{0}. P = {1:.2f}'.format(self.idx+1, self.data['Period']))
        self.label = ctk.CTkLabel(self, textvariable = self.labelvar, font = ('Arial', 18, 'bold'))
        self.label.grid(row = 0, column = 0, padx = 10, sticky = 'ew')

        self.edit = ctk.CTkButton(self, text = 'Edit', command = self.edit_cmd, font = ('Arial', 18, 'bold'))
        self.edit.grid(row = 0, column = 1, padx = 10, sticky = 'ew')

        self.copy = ctk.CTkButton(self, text = 'Copy', command = lambda d=self.data: self.topmaster.copy_cmd(d), font = ('Arial', 18, 'bold'))
        self.copy.grid(row = 0, column = 2, padx = 10, sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete', command = self.delete_cmd, font = ('Arial', 18, 'bold'))
        self.delete.grid(row = 0, column = 3, padx = 10, sticky = 'ew')

    def edit_cmd(self):

        if not hasattr(self, 'edit_planet'):

            self.edit_planet = PlanetGUI(self.data, self.idx+1, self.edit_close)
            self.edit_planet.protocol("WM_DELETE_WINDOW", self.edit_close)

            self.wait_window(self.edit_planet)

            self.update_label()

        else:

            self.edit_planet.withdraw()
            self.edit_planet.update()
            self.edit_planet.deiconify()
            self.edit_planet.lift()
            self.edit_planet.focus()


    def edit_close(self):

        self.data = self.edit_planet.data
        self.edit_planet.destroy()
        delattr(self, 'edit_planet')


    def delete_cmd(self):

        self.topmaster.delete_cmd(self.idx)

    def update_label(self):

        self.labelvar.set('{0}. P = {1:.2f}'.format(self.idx+1, self.data['Period']))




class InitPlanetsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('600x400')

        self.setup_items_frame()

    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self)
        self.items_frame.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'nesw', columnspan = 2)
        self.items_frame.grid_columnconfigure(0, weight = 1)

        self.frame_title = ctk.CTkLabel(self.items_frame, text = 'Planets', font = ('Arial', 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, sticky = 'ew')

        self.planets = []

        for i in range(len(self.table)):

            self.add_planet(i)

    def add_planet(self, i):

        planet = PlanetFrame(self.items_frame, self, self.table[i], i)
        planet.grid(row = i+1, column = 0, pady = (10,0), sticky = 'ew')
        self.planets.append(planet)


    def add_cmd(self):
        
        self.table.add_row([False]*5 + [np.nan]*9)
        i = len(self.planets)
        self.add_planet(i)
        self.planets[i].edit_cmd()


    def copy_cmd(self, data: Row):

        self.table.add_row(data)
        i = len(self.planets)
        self.add_planet(i)


    def delete_cmd(self, i):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            self.table.remove_row(i)
            self.planets[i].grid_forget()
            self.planets.pop(i)

            for j in range(len(self.planets)):
                self.planets[j].grid(row = j+1, column = 0, pady = (10, 0), sticky = 'ew')
                self.planets[j].idx = j
                self.planets[j].data = self.table[j]
                self.planets[j].update_label()



##################
#####Init Lcs#####
##################


class LcFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row):
        super().__init__(master)

        self.data = data

        self.grid_rowconfigure(list(range(12)), weight = 1)

        self.checkval = ctk.IntVar(value = 0)
        self.checkbox = ctk.CTkCheckBox(self, variable = self.checkval, text = '')
        self.checkbox.grid(row = 0, column = 0, pady = (10,0), sticky = 'nsew')

        self.filename = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['File']))
        self.filename.grid(row = 1, column = 0, pady = (10,0), sticky = 'nsew')

        self.nickname = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['Nickname']))
        self.nickname.grid(row = 2, column = 0, pady = (10,0), sticky = 'nsew')

        self.time = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['Time Col']))
        self.time.grid(row = 3, column = 0, pady = (10,0), sticky = 'nsew')

        self.flux = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['Flux Col']))
        self.flux.grid(row = 4, column = 0, pady = 10, sticky = 'nsew')

        self.fluxerr = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['Err Col']))
        self.fluxerr.grid(row = 5, column = 0, pady = (10,0), sticky = 'nsew')

        self.quality = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['Quality Col']))
        self.quality.grid(row = 6, column = 0, pady = (10,0), sticky = 'nsew')

        self.offset = FloatEntry(self, textvariable = ctk.StringVar(value = '' if np.isnan(data['Time Offset']) else data['Time Offset']), font = ('Arial', 18))
        self.offset.grid(row = 7, column = 0, pady = (10,0), sticky = 'nsew')

        self.errscale = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Err Scale']) else data['Err Scale']), font = ('Arial', 18))
        self.errscale.grid(row = 8, column = 0, pady = (10,0), sticky = 'nsew')

        self.exptime = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Exp Time']) else data['Exp Time']), font = ('Arial', 18))
        self.exptime.grid(row = 9, column = 0, pady = (10,0), sticky = 'nsew')

        self.filter = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', textvariable = ctk.StringVar(value = data['Filter']))
        self.filter.grid(row = 10, column = 0, pady = (10,0), sticky = 'nsew')

        self.detrend = ctk.CTkCheckBox(self, variable = ctk.IntVar(value = 1 if data['Detrend'] else 0), text = '')
        self.detrend.grid(row = 11, column = 0, pady = (10,0), sticky = 'nsew')


class InitLcsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('1000x800')

        self.setup_items_frame()

    
    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self, orientation = 'horizontal')
        self.items_frame.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'nesw', columnspan = 2)
        self.items_frame.grid_rowconfigure(list(range(13)), weight = 1)

        self.frame_title = ctk.CTkLabel(self.items_frame, text = 'Light Curve Files', font = ('Arial', 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'nsew')

        self.copy = ctk.CTkButton(self.items_frame, text = 'Copy Selected', font = ('Arial', 18, 'bold'), command = self.copy_cmd)
        self.copy.grid(row = 0, column = 1, padx = 10, pady = (10,0), sticky = 'nesw')

        self.delete = ctk.CTkButton(self.items_frame, text = 'Delete Selected', font = ('Arial', 18, 'bold'), command = self.delete_cmd)
        self.delete.grid(row = 0, column = 2, padx = 10, pady = (10,0), sticky = 'nesw')

        labels = ['Select','File Name','Nickname','Time Column','Flux Column','Error Column','Quality Column','Time Offset (BJD)','Error Scale','Exp Time (s)','Filter','Detrend']
        for i in range(12):
            label = ctk.CTkLabel(self.items_frame, text = labels[i], font = ('Arial', 18))
            label.grid(row = i+1, column = 0, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.lcs = []

        for i in range(len(self.table)):

            self.add_lc(i)


    def add_lc(self, i):

        lc = LcFrame(self.items_frame, self.table[i])
        lc.grid(row = 1, column = i+1, padx = 10, sticky = 'nsew', rowspan = 12)
        self.lcs.append(lc)


    def add_cmd(self):
        
        self.table.add_row(['']*6 + [np.nan]*3 + [''] + [False])
        i = len(self.lcs)
        self.add_lc(i)


    def copy_cmd(self):

        checked = []

        for i in range(len(self.lcs)):
            
            if self.lcs[i].checkval.get():
                checked.append(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.lcs),len(self.lcs)+len(checked)):
            self.add_lc(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = []

            for i in range(len(self.lcs)):
                
                if self.lcs[i].checkval.get():
                    checked.append(i)

            self.table.remove_rows(checked)
            for i in checked:
                self.lcs[i].grid_forget()
            self.lcs = list(np.delete(self.lcs, checked))

            for i in range(len(self.lcs)):
                self.lcs[i].grid(row = 1, column = i+1, padx = 10, sticky = 'nsew', rowspan = 12)
                self.lcs[i].data = self.table[i]


    def save_cmd(self):

        for i in range(len(self.lcs)):

            lc = self.lcs[i]

            self.table['File'][i] = lc.filename.get().strip()
            self.table['Nickname'][i] = lc.nickname.get().strip()
            self.table['Time Col'][i] = lc.time.get().strip()
            self.table['Flux Col'][i] = lc.flux.get().strip()
            self.table['Err Col'][i] = lc.fluxerr.get().strip()
            self.table['Quality Col'][i] = lc.quality.get().strip()
            self.table['Time Offset'][i] = lc.offset.return_float()
            self.table['Err Scale'][i] = lc.errscale.return_float()
            self.table['Exp Time'][i] = lc.exptime.return_float()
            self.table['Filter'][i] = lc.filter.get().strip()
            self.table['Detrend'][i] = lc.detrend.get() == 1

        self.initfile.table = self.table

        super().save_cmd()