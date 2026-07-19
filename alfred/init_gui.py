from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alfred.init_class import InitFile

import customtkinter as ctk
ctk.set_appearance_mode('dark')
ctk.DrawEngine.preferred_drawing_method = "circle_shapes"

font = 'Verdana'

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
        self.grid_rowconfigure(1, weight = 1)

        self.add = ctk.CTkButton(self, text = 'Add', command = self.add_cmd, font = (font, 18, 'bold'))
        self.add.grid(row = 2, column = 0, padx = 100, pady = (10, 0), sticky = 'ew', columnspan = 2)

        self.load = ctk.CTkButton(self, text = 'Load', command = self.load_cmd, font = (font, 18, 'bold'))
        self.load.grid(row = 4, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', command = self.save_cmd, font = (font, 18, 'bold'))
        self.save.grid(row = 4, column = 1, padx = 10, pady = (10, 0), sticky = 'ew')

        self.rename = ctk.CTkButton(self, text = 'Rename', command = self.rename_cmd, font = (font, 18, 'bold'))
        self.rename.grid(row = 5, column = 0, padx = 10, pady = 10, sticky = 'ew')

        self.directory = ctk.CTkButton(self, text = 'Directory', command = self.direc_cmd, font = (font, 18, 'bold'))
        self.directory.grid(row = 5, column  = 1, padx = 10, pady = 10, sticky = 'ew')

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

        self.label = ctk.CTkLabel(self, text = 'Delete?', font = (font, 18, 'bold'))
        self.label.grid(row = 0, column = 0, padx = 10, sticky = 'ew', columnspan = 2)

        self.yes = ctk.CTkButton(self, text = 'Yes', command = self.yes_cmd, font = (font, 18, 'bold'))
        self.yes.grid(row = 1, column  = 0, padx = 10, sticky = 'ew')

        self.no = ctk.CTkButton(self, text = 'No', command = self.no_cmd, font = (font, 18, 'bold'))
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
        self.entry = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = self.entry_var)
        self.entry.grid(row = 0, column = 0, padx = 10, pady = (10, 0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', font = (font, 18, 'bold'), command = self.save_cmd)
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
        self.entry = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = self.entry_var)
        self.entry.grid(row = 0, column = 0, padx = 10, pady = (10, 0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', font = (font, 18, 'bold'), command = self.save_cmd)
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

        self.label = ctk.CTkLabel(self, text = label, font = (font, 18, 'bold'))
        self.entry = FloatEntry(self, min_val = self.min_val, max_val = self.max_val, font = (font, 18), justify = 'center')

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

        super().__init__(master, text = self.label, font = (font, 18, 'bold'), variable = self.val, command = self.toggle_entry)
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
    


class ScrollableDropdown(ctk.CTkToplevel):
    def __init__(self, attach, values=None, height=200, width=None, command=None, font = None, **kwargs):
        super().__init__(takefocus=1)
        self.geometry("0x0+0+0")
        self.wm_overrideredirect(True)
        self.attributes("-topmost", True)
        
        self.attach = attach
        self.values = values if values else []
        self.command = command
        self.font = font
        
        # Store requested layout dimensions
        self.requested_width = width
        self.requested_height = height
        
        # Build standard Frame container inside window wrapper
        # We start with placeholder dimensions, then update them on click
        self.frame = ctk.CTkScrollableFrame(self, width=150, height=self.requested_height, fg_color = "#4F4F4F")
        self.frame.pack(fill="both", expand=True)
        
        # Hide window on creation immediately
        self.withdraw()
        
        # Cleanly attach trigger actions onto the target Button / OptionMenu
        if isinstance(attach, ctk.CTkOptionMenu):
            attach.configure(command=self._icon_click)
        elif isinstance(attach, ctk.CTkButton):
            attach.configure(command=self._icon_click)
            
        self._update_options()

    def _update_options(self):
        for child in self.frame.winfo_children():
            child.destroy()
            
        for value in self.values:
            btn = ctk.CTkButton(
                self.frame,
                text=str(value),
                fg_color="transparent",
                text_color=("black", "white"),
                font = self.font,
                hover_color=("#dbdbdb", "#707070"),
                anchor="w",
                command=lambda v=value: self._on_select(v)
            )
            btn.pack(fill="x", pady=1, padx=2)

    def _icon_click(self, *args):
        if self.winfo_viewable():
            self.withdraw()
        else:
            self._place_dropdown()

    def _place_dropdown(self):
        # Crucial Fix: Force Tkinter to refresh geometry buffers before calculation
        self.attach.update_idletasks()
        
        # Compute exact pixel coordinates *at the time of the click*
        x = self.attach.winfo_rootx()
        y = self.attach.winfo_rooty() + self.attach.winfo_height()
        
        # Match button's actual rendered width if no custom override is passed
        width = self.requested_width if self.requested_width else self.attach.winfo_width()
        
        # Resize internal elements safely
        self.frame.configure(width=width - 15)
        
        # Position and display the pop-up overlay window directly below button boundary
        self.geometry(f"{width}x{self.requested_height}+{x}+{y}")
        self.deiconify()
        self.focus_set()
        
        # Close drop menu safely if user clicks anywhere else in the application window
        self.bind("<FocusOut>", lambda e: self.withdraw())

    def _on_select(self, value):
        if self.command:
            self.command(value)
        elif hasattr(self.attach, "set"):
            self.attach.set(value)
        elif isinstance(self.attach, ctk.CTkButton):
            self.attach.configure(text=value)
        self.withdraw()
    


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


        self.save = ctk.CTkButton(self, text = 'Save', command = self.save_cmd, font = (font, 18, 'bold'))
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
        self.label = ctk.CTkLabel(self, textvariable = self.labelvar, font = (font, 18, 'bold'))
        self.label.grid(row = 0, column = 0, padx = 10, sticky = 'ew')

        self.edit = ctk.CTkButton(self, text = 'Edit', command = self.edit_cmd, font = (font, 18, 'bold'))
        self.edit.grid(row = 0, column = 1, padx = 10, sticky = 'ew')

        self.copy = ctk.CTkButton(self, text = 'Copy', command = lambda d=self.data: self.topmaster.copy_cmd(d), font = (font, 18, 'bold'))
        self.copy.grid(row = 0, column = 2, padx = 10, sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete', command = self.delete_cmd, font = (font, 18, 'bold'))
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

        self.frame_title = ctk.CTkLabel(self, text = 'Planets', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, sticky = 'ew', columnspan = 2)

    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self)
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nesw', columnspan = 2)
        self.items_frame.grid_columnconfigure(0, weight = 1)

        self.planets = []

        for i in range(len(self.table)):

            self.add_planet(i)

    def add_planet(self, i):

        planet = PlanetFrame(self.items_frame, self, self.table[i], i)
        planet.grid(row = i, column = 0, pady = (10,0), sticky = 'ew')
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
                self.planets[j].grid(row = j, column = 0, pady = (10, 0), sticky = 'ew')
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

        self.filename = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['File']))
        self.filename.grid(row = 1, column = 0, pady = (10,0), sticky = 'nsew')

        self.nickname = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Nickname']))
        self.nickname.grid(row = 2, column = 0, pady = (10,0), sticky = 'nsew')

        self.time = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Time Col']))
        self.time.grid(row = 3, column = 0, pady = (10,0), sticky = 'nsew')

        self.flux = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Flux Col']))
        self.flux.grid(row = 4, column = 0, pady = (10,0), sticky = 'nsew')

        self.fluxerr = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Err Col']))
        self.fluxerr.grid(row = 5, column = 0, pady = (10,0), sticky = 'nsew')

        self.quality = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Quality Col']))
        self.quality.grid(row = 6, column = 0, pady = (10,0), sticky = 'nsew')

        self.offset = FloatEntry(self, textvariable = ctk.StringVar(value = '' if np.isnan(data['Time Offset']) else data['Time Offset']), font = (font, 18))
        self.offset.grid(row = 7, column = 0, pady = (10,0), sticky = 'nsew')

        self.errscale = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Err Scale']) else data['Err Scale']), font = (font, 18))
        self.errscale.grid(row = 8, column = 0, pady = (10,0), sticky = 'nsew')

        self.exptime = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Exp Time']) else data['Exp Time']), font = (font, 18))
        self.exptime.grid(row = 9, column = 0, pady = (10,0), sticky = 'nsew')

        self.filter = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Filter']))
        self.filter.grid(row = 10, column = 0, pady = (10,0), sticky = 'nsew')

        self.detrend = ctk.CTkCheckBox(self, variable = ctk.IntVar(value = 1 if data['Detrend'] else 0), text = '')
        self.detrend.grid(row = 11, column = 0, pady = 10, sticky = 'nsew')


class InitLcsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('1000x800')

        self.setup_items_frame()

        self.frame_title = ctk.CTkLabel(self, text = 'Light Curve Files', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, sticky = 'ew', columnspan = 2)

        self.copy = ctk.CTkButton(self, text = 'Copy Selected', font = (font, 18, 'bold'), command = self.copy_cmd)
        self.copy.grid(row = 3, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete Selected', font = (font, 18, 'bold'), command = self.delete_cmd)
        self.delete.grid(row = 3, column = 1, padx = 10, pady = (10,0), sticky = 'ew')

    
    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self, orientation = 'horizontal')
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nesw', columnspan = 2)
        self.items_frame.grid_rowconfigure(list(range(12)), weight = 1)

        labels = ['Select','File Name','Nickname','Time Column','Flux Column','Error Column','Quality Column','Time Offset (BJD)','Error Scale','Exp Time (s)','Filter','Detrend']
        for i in range(12):
            label = ctk.CTkLabel(self.items_frame, text = labels[i], font = (font, 18, 'bold'))
            label.grid(row = i, column = 0, padx = (10,0), pady = (10,0) if i < 11 else 10, sticky = 'nsew')

        self.lcs = []

        for i in range(len(self.table)):

            self.add_lc(i)


    def add_lc(self, i):

        lc = LcFrame(self.items_frame, self.table[i])
        lc.grid(row = 0, column = i+1, padx = 10, sticky = 'nsew', rowspan = 12)
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
                self.lcs[i].grid(row = 0, column = i+1, padx = 10, sticky = 'nsew', rowspan = 12)
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



#################
#####Init RV#####
#################


class RVFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row):
        super().__init__(master)

        self.data = data

        self.grid_rowconfigure(list(range(9)), weight = 1)

        self.checkval = ctk.IntVar(value = 0)
        self.checkbox = ctk.CTkCheckBox(self, variable = self.checkval, text = '')
        self.checkbox.grid(row = 0, column = 0, pady = (10,0), sticky = 'nsew')

        self.filename = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['File']))
        self.filename.grid(row = 1, column = 0, pady = (10,0), sticky = 'nsew')

        self.nickname = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Nickname']))
        self.nickname.grid(row = 2, column = 0, pady = (10,0), sticky = 'nsew')

        self.time = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Time Col']))
        self.time.grid(row = 3, column = 0, pady = (10,0), sticky = 'nsew')

        self.rv = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['RV Col']))
        self.rv.grid(row = 4, column = 0, pady = (10,0), sticky = 'nsew')

        self.rverr = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Err Col']))
        self.rverr.grid(row = 5, column = 0, pady = (10,0), sticky = 'nsew')

        self.offset = FloatEntry(self, textvariable = ctk.StringVar(value = '' if np.isnan(data['Time Offset']) else data['Time Offset']), font = (font, 18))
        self.offset.grid(row = 6, column = 0, pady = (10,0), sticky = 'nsew')

        self.errscale = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Err Scale']) else data['Err Scale']), font = (font, 18))
        self.errscale.grid(row = 7, column = 0, pady = (10,0), sticky = 'nsew')

        self.units = ctk.CTkOptionMenu(self, values = ['m/s','km/s'], font = (font, 18), variable = ctk.StringVar(value = 'm/s' if data['m/s or km/s'] == '' else data['m/s or km/s']))
        self.units.grid(row = 8, column = 0, pady = 10, sticky = 'nesw')
        self.units._dropdown_menu.configure(font = (font, 18))


class InitRVGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('1000x700')

        self.setup_items_frame()

        self.frame_title = ctk.CTkLabel(self, text = 'RV Files', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, sticky = 'ew', columnspan = 2)

        self.copy = ctk.CTkButton(self, text = 'Copy Selected', font = (font, 18, 'bold'), command = self.copy_cmd)
        self.copy.grid(row = 3, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete Selected', font = (font, 18, 'bold'), command = self.delete_cmd)
        self.delete.grid(row = 3, column = 1, padx = 10, pady = (10,0), sticky = 'ew')

    
    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self, orientation = 'horizontal')
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nesw', columnspan = 2)
        self.items_frame.grid_rowconfigure(list(range(9)), weight = 1)

        labels = ['Select','File Name','Nickname','Time Column','RV Column','Error Column','Time Offset (BJD)','Error Scale','Units']
        for i in range(9):
            label = ctk.CTkLabel(self.items_frame, text = labels[i], font = (font, 18, 'bold'))
            label.grid(row = i, column = 0, padx = (10,0), pady = (10,0) if i < 8 else 10, sticky = 'nsew')

        self.rvs = []

        for i in range(len(self.table)):

            self.add_rv(i)


    def add_rv(self, i):

        rv = RVFrame(self.items_frame, self.table[i])
        rv.grid(row = 0, column = i+1, padx = 10, sticky = 'nsew', rowspan = 9)
        self.rvs.append(rv)


    def add_cmd(self):
        
        self.table.add_row(['']*5 + [np.nan]*2 + [''])
        i = len(self.rvs)
        self.add_rv(i)


    def copy_cmd(self):

        checked = []

        for i in range(len(self.rvs)):
            
            if self.rvs[i].checkval.get():
                checked.append(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.rvs),len(self.rvs)+len(checked)):
            self.add_rv(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = []

            for i in range(len(self.rvs)):
                
                if self.rvs[i].checkval.get():
                    checked.append(i)

            self.table.remove_rows(checked)
            for i in checked:
                self.rvs[i].grid_forget()
            self.rvs = list(np.delete(self.rvs, checked))

            for i in range(len(self.rvs)):
                self.rvs[i].grid(row = 0, column = i+1, padx = 10, sticky = 'nsew', rowspan = 9)
                self.rvs[i].data = self.table[i]


    def save_cmd(self):

        for i in range(len(self.rvs)):

            rv = self.rvs[i]

            self.table['File'][i] = rv.filename.get().strip()
            self.table['Nickname'][i] = rv.nickname.get().strip()
            self.table['Time Col'][i] = rv.time.get().strip()
            self.table['RV Col'][i] = rv.rv.get().strip()
            self.table['Err Col'][i] = rv.rverr.get().strip()
            self.table['Time Offset'][i] = rv.offset.return_float()
            self.table['Err Scale'][i] = rv.errscale.return_float()
            self.table['m/s or km/s'][i] = rv.units.get()

        self.initfile.table = self.table

        super().save_cmd()



#####################
#####Init Priors#####
#####################


class PriorFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row):
        super().__init__(master)

        self.data = data

        self.grid_columnconfigure(list(range(6)), weight = 1)
        self.grid_rowconfigure((0,1), weight = 1)

        self.checkval = ctk.IntVar(value = 0)
        self.checkbox = ctk.CTkCheckBox(self, variable = self.checkval, text = '')
        self.checkbox.grid(row = 1, column = 0, padx = (10,0), pady = 10, sticky = 'nsew')

        varoptions = ['log(P)', 'P', 'Tc', 'ror', 'log(a/rs)', 'a/rs', 'rhos', 'cos(i)', 'i', 'log(K)', 'K', 'secw', 'sesw', 'e', 'w', 'TT', 'fp',
                    'F0', 'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp', 'u1', 'u2', 'gamma', 'gamma_dot', 'gamma_ddot', 'rv_offset',
                    'eep', 'log10(age)', 'age', 'feh', 'distance', 'AV', 'mstar', 'rstar', 'rhostar']
        # self.variable = ctk.CTkOptionMenu(self, values = varoptions, font = (font, 18), variable = ctk.StringVar(value = 'P' if data['Variable'] == '' else data['Variable'].split()[0]), command = self.varoption_cmd)
        self.variable_anchor = ctk.CTkButton(self, textvariable = ctk.StringVar(value = 'P' if data['Variable'] == '' else data['Variable'].split()[0]), font = (font, 18))
        self.variable_anchor.grid(row = 1, column = 1, padx = (10,0), pady = 10, sticky = 'nesw')
        self.variable = ScrollableDropdown(self.variable_anchor, values = varoptions, height = 400, command = self.varoption_cmd, font = (font, 18))

        self.varnumberlabel_var = ctk.StringVar(value = '')
        self.varnumberlabel = ctk.CTkLabel(self, textvariable = self.varnumberlabel_var, font = (font, 18, 'bold'))
        self.varnumberlabel.grid(row = 0, column = 2, padx = (10,0), pady = (10,0), sticky = 'nsew')
        self.varnumber = ctk.CTkEntry(self,  font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = '' if data['Variable'] == '' else data['Variable'].split()[1:]))
        self.varnumber.grid(row = 1, column = 2, padx = (10,0), pady = 10, sticky = 'nsew')

    def varoption_cmd(self, choice):

        self.variable_anchor.configure(text=choice)



class InitPriorsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('800x800')

        self.setup_items_frame()

        self.frame_title = ctk.CTkLabel(self, text = 'Priors', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'ew', columnspan = 2)

        self.copy = ctk.CTkButton(self, text = 'Copy Selected', font = (font, 18, 'bold'), command = self.copy_cmd)
        self.copy.grid(row = 3, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete Selected', font = (font, 18, 'bold'), command = self.delete_cmd)
        self.delete.grid(row = 3, column = 1, padx = 10, pady = (10,0), sticky = 'ew')


    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self)
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nsew', columnspan = 2)
        self.items_frame.grid_columnconfigure(list(range(6)), weight = 1)

        labels = ['Select','Variable','','Prior Distribution','','']
        for i in range(6):
            label = ctk.CTkLabel(self.items_frame, text = labels[i], font = (font, 18, 'bold'))
            label.grid(row = 0, column = i, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.priors = []

        for i in range(len(self.table)):

            self.add_prior(i)


    def add_prior(self, i):

        prior = PriorFrame(self.items_frame, self.table[i])
        prior.grid(row = i+1, column = 0, pady = (10,0), sticky = 'ew', columnspan = 6)
        self.priors.append(prior)


    def add_cmd(self):
        
        self.table.add_row(['']*5 + [np.nan]*2 + [''])
        i = len(self.rvs)
        self.add_rv(i)


    def copy_cmd(self):

        checked = []

        for i in range(len(self.rvs)):
            
            if self.rvs[i].checkval.get():
                checked.append(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.rvs),len(self.rvs)+len(checked)):
            self.add_rv(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = []

            for i in range(len(self.rvs)):
                
                if self.rvs[i].checkval.get():
                    checked.append(i)

            self.table.remove_rows(checked)
            for i in checked:
                self.rvs[i].grid_forget()
            self.rvs = list(np.delete(self.rvs, checked))

            for i in range(len(self.rvs)):
                self.rvs[i].grid(row = 0, column = i+1, padx = 10, sticky = 'nsew', rowspan = 9)
                self.rvs[i].data = self.table[i]


    def save_cmd(self):

        for i in range(len(self.rvs)):

            rv = self.rvs[i]

            self.table['File'][i] = rv.filename.get().strip()
            self.table['Nickname'][i] = rv.nickname.get().strip()
            self.table['Time Col'][i] = rv.time.get().strip()
            self.table['RV Col'][i] = rv.rv.get().strip()
            self.table['Err Col'][i] = rv.rverr.get().strip()
            self.table['Time Offset'][i] = rv.offset.return_float()
            self.table['Err Scale'][i] = rv.errscale.return_float()
            self.table['m/s or km/s'][i] = rv.units.get()

        self.initfile.table = self.table

        super().save_cmd()