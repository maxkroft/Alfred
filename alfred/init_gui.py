from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alfred.init_class import InitFile

from alfred.ld_grids import ld_grid_list, calc_ld

import customtkinter as ctk
ctk.set_appearance_mode('dark')
ctk.DrawEngine.preferred_drawing_method = "circle_shapes"

font = 'Verdana'

from astropy.table import Table, Row, Column, vstack
from astropy import units as u
import numpy as np
import re
from astroquery.vizier import Vizier
import platform


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

        if platform.system() == 'Windows':
            self.deiconify()
            self.attributes("-topmost", True)
            self.lift()
            self.focus_force()
            self.after(200, lambda: self.attributes("-topmost", False))

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

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()
        self.grab_set()

        self.after(200, lambda: self.attributes("-topmost", False))

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

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()
        self.grab_set()

        self.after(200, lambda: self.attributes("-topmost", False))

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

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()
        self.grab_set()

        self.after(200, lambda: self.attributes("-topmost", False))

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

        self.bind("<FocusOut>", self._on_focus_out)
        

    def validate_float_input(self, proposed_text):

        if proposed_text == "":
            return True
        
        text = proposed_text.strip()

        incomplete_sci_pattern = r'^[-+]?(\d*\.?\d*)[eE][-+]?$'

        if text in ("-", "+", ".", "-.", "+.") or re.match(incomplete_sci_pattern, text):
            if text.startswith("-") and self.min_val is not None and float(self.min_val) >= 0:
                return False
            return True

        try:
            val = float(proposed_text)
            return True
        
        except ValueError:
            return False


    def _on_focus_out(self, event):
        try:
            raw_text = self.get().strip()
            if not raw_text:
                return
                
            val = float(raw_text)
            
            if self.min_val is not None and val < self.min_val:
                self.delete(0, "end")
                self.insert(0, str(self.min_val))
                self.configure(text_color = "#CB2626")

            elif self.max_val is not None and val > self.max_val:
                self.delete(0, "end")
                self.insert(0, str(self.max_val))
                self.configure(text_color = "#CB2626")

            else:
                self.configure(text_color = '#FFFFFF')

        except ValueError:
            self.delete(0, "end")

    def return_float(self):
        try:
            x = float(self.get().strip())
            if self.min_val is not None and x < self.min_val:
                return np.nan
            if self.max_val is not None and x > self.max_val:
                return np.nan
            return x
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
    def __init__(self, master, attach, values=None, height=200, width=None, command=None, font = None, **kwargs):
        super().__init__(master, takefocus=1)
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

        self.attach.update_idletasks()
        
        x = self.attach.winfo_rootx()
        y = self.attach.winfo_rooty() + self.attach.winfo_height()
        
        width = self.requested_width if self.requested_width else self.attach.winfo_width()
        
        self.frame.configure(width=width - 15)
        
        self.geometry(f"{width}x{self.requested_height}+{x}+{y}")
        self.deiconify()
        self.lift()
        self.focus_force()
        
        self.bind("<FocusOut>", self._on_focus_out)

    def _on_select(self, value):
        if self.command:
            self.command(value)
        elif hasattr(self.attach, "set"):
            self.attach.set(value)
        elif isinstance(self.attach, ctk.CTkButton):
            self.attach.configure(text=value)
        self.withdraw()

    def _on_focus_out(self, event):

        mouse_x = self.winfo_pointerx()
        mouse_y = self.winfo_pointery()
        
        geo_x = self.winfo_rootx()
        geo_y = self.winfo_rooty()
        geo_w = self.winfo_width()
        geo_h = self.winfo_height()
        
        if (geo_x <= mouse_x <= geo_x + geo_w) and (geo_y <= mouse_y <= geo_y + geo_h):
            return
            
        self.withdraw()
    


class CheckDropDown(ctk.CTkFrame):
    def __init__(self, master, options, startval, **kwargs):
        super().__init__(master, fg_color = 'transparent', **kwargs)

        self.options = options

        self.grid_columnconfigure(1, weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        self.checkval = ctk.IntVar(value = 0)
        self.checkbox = ctk.CTkCheckBox(self, variable = self.checkval, text = '', width = 24)
        self.checkbox.grid(row = 0, column = 0, sticky = 'nsew')

        self.dropdown_anchor = ctk.CTkButton(self, text = startval, font = (font, 18))
        self.dropdown_anchor.grid(row = 0, column = 1, sticky = 'nesw')
        self.dropdown = ScrollableDropdown(master, self.dropdown_anchor, values = self.options, height = 300, font = (font, 18))

        


def update_string_col(col: Column, value: str, idx: int) -> Column:

    if len(value) > int(col.dtype.str[2:]):
        col = col.astype(f'S{len(value)}')

    col[idx] = value

    return col


######################
#####Init Planets#####
######################

class PlanetGUI(ctk.CTkToplevel):
    def __init__(self, data: Row, idx: int, close_cmd):
        super().__init__()

        self.data = data
        self.close_cmd = close_cmd

        self.title('Planet {0}'.format(idx))
        self.geometry('600x600')
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

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()

        self.after(200, lambda: self.attributes("-topmost", False))



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

            self.edit_planet = PlanetGUI(self.data, self.idx+1, self.save_close)
            self.edit_planet.protocol("WM_DELETE_WINDOW", self.edit_close)

            self.wait_window(self.edit_planet)

            self.update_label()

        else:

            self.edit_planet.withdraw()
            self.edit_planet.update()
            self.edit_planet.deiconify()
            self.edit_planet.lift()
            self.edit_planet.focus()

    
    def save_close(self):

        self.data = self.edit_planet.data
        self.edit_planet.destroy()
        delattr(self, 'edit_planet')

    def edit_close(self):

        self.edit_planet.destroy()
        delattr(self, 'edit_planet')


    def delete_cmd(self):

        self.topmaster.delete_cmd(self.idx)

    def update_label(self):

        self.labelvar.set('{0}. P = {1:.2f}'.format(self.idx+1, self.data['Period']))




class InitPlanetsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('800x600')

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
                self.planets[j].grid_configure(row = j)
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

        self.offset = FloatEntry(self, textvariable = ctk.StringVar(value = '' if np.isnan(data['Time Offset']) else data['Time Offset']), font = (font, 18), justify = 'center')
        self.offset.grid(row = 7, column = 0, pady = (10,0), sticky = 'nsew')

        self.errscale = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Err Scale']) else data['Err Scale']), font = (font, 18), justify = 'center')
        self.errscale.grid(row = 8, column = 0, pady = (10,0), sticky = 'nsew')

        self.exptime = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Exp Time']) else data['Exp Time']), font = (font, 18), justify = 'center')
        self.exptime.grid(row = 9, column = 0, pady = (10,0), sticky = 'nsew')

        self.filter = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = data['Filter']))
        self.filter.grid(row = 10, column = 0, pady = (10,0), sticky = 'nsew')

        self.detrend = ctk.CTkCheckBox(self, variable = ctk.IntVar(value = 1 if data['Detrend'] else 0), text = '')
        self.detrend.grid(row = 11, column = 0, pady = 10, sticky = 'nsew')


class InitLcsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('1000x750')

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
        
        self.table.add_row(['']*6 + [np.nan,1,np.nan] + [''] + [False])
        i = len(self.lcs)
        self.add_lc(i)


    def copy_cmd(self):

        checked = [i for i, lc in enumerate(self.lcs) if lc.checkval.get()]

        for i in checked:
            self.update_table(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.lcs),len(self.lcs)+len(checked)):
            self.add_lc(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = [i for i, lc in enumerate(self.lcs) if lc.checkval.get()]

            if not checked:
                return

            self.table.remove_rows(checked)

            for i in checked:
                self.lcs[i].grid_forget()
                self.lcs[i].destroy()

            self.lcs = [lc for i, lc in enumerate(self.lcs) if i not in checked]

            for i in range(len(self.lcs)):
                self.lcs[i].grid_configure(column = i+1)
                self.lcs[i].data = self.table[i]


    def save_cmd(self):

        for i in range(len(self.lcs)):

            self.update_table(i)

        self.initfile.table = self.table

        super().save_cmd()


    def update_table(self, i):

        lc = self.lcs[i]

        self.table['File'] = update_string_col(self.table['File'], lc.filename.get().strip(), i)
        self.table['Nickname'] = update_string_col(self.table['Nickname'], lc.nickname.get().strip(), i)
        self.table['Time Col'] = update_string_col(self.table['Time Col'], lc.time.get().strip(), i)
        self.table['Flux Col'] = update_string_col(self.table['Flux Col'], lc.flux.get().strip(), i)
        self.table['Err Col'] = update_string_col(self.table['Err Col'], lc.fluxerr.get().strip(), i)
        self.table['Quality Col'] = update_string_col(self.table['Quality Col'], lc.quality.get().strip(), i)
        self.table['Time Offset'][i] = lc.offset.return_float()
        self.table['Err Scale'][i] = lc.errscale.return_float()
        self.table['Exp Time'][i] = lc.exptime.return_float()
        self.table['Filter'] = update_string_col(self.table['Filter'], lc.filter.get().strip(), i)
        self.table['Detrend'][i] = lc.detrend.get() == 1



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

        self.offset = FloatEntry(self, textvariable = ctk.StringVar(value = '' if np.isnan(data['Time Offset']) else data['Time Offset']), font = (font, 18), justify = 'center')
        self.offset.grid(row = 6, column = 0, pady = (10,0), sticky = 'nsew')

        self.errscale = FloatEntry(self, min_val = 0, textvariable = ctk.StringVar(value = '' if np.isnan(data['Err Scale']) else data['Err Scale']), font = (font, 18), justify = 'center')
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
        
        self.table.add_row(['']*5 + [np.nan,1] + [''])
        i = len(self.rvs)
        self.add_rv(i)


    def copy_cmd(self):

        checked = [i for i, rv in enumerate(self.rvs) if rv.checkval.get()]

        for i in checked:
            self.update_table(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.rvs),len(self.rvs)+len(checked)):
            self.add_rv(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = [i for i, rv in enumerate(self.rvs) if rv.checkval.get()]

            if not checked:
                return

            self.table.remove_rows(checked)

            for i in checked:
                self.rvs[i].grid_forget()
                self.rvs[i].destroy()

            self.rvs = [rv for i, rv in enumerate(self.rvs) if i not in checked]

            for i in range(len(self.rvs)):
                self.rvs[i].grid_configure(column = i+1)
                self.rvs[i].data = self.table[i]

        

    def save_cmd(self):

        for i in range(len(self.rvs)):

            self.update_table(i)

        self.initfile.table = self.table

        super().save_cmd()


    def update_table(self, i):

        rv = self.rvs[i]

        self.table['File'] = update_string_col(self.table['File'], rv.filename.get().strip(), i)
        self.table['Nickname'] = update_string_col(self.table['Nickname'], rv.nickname.get().strip(), i)
        self.table['Time Col'] = update_string_col(self.table['Time Col'], rv.time.get().strip(), i)
        self.table['RV Col'] = update_string_col(self.table['RV Col'], rv.rv.get().strip(), i)
        self.table['Err Col'] = update_string_col(self.table['Err Col'], rv.rverr.get().strip(), i)
        self.table['Time Offset'][i] = rv.offset.return_float()
        self.table['Err Scale'][i] = rv.errscale.return_float()
        self.table['m/s or km/s'] = update_string_col(self.table['m/s or km/s'], rv.units.get().strip(), i)



#####################
#####Init Priors#####
#####################


class PriorFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row):
        super().__init__(master)

        self.data = data

        self.prior_convert = {'U': 'Uniform', 'G': 'Gaussian', 'F': 'Fixed', 'J': "Jeffrey's", 'MJ': "Mod. Jeffrey's"}

        self.grid_columnconfigure(list(range(1,8)), weight = 1)
        self.grid_rowconfigure((0,1), weight = 1)

        checkboxlabel = ctk.CTkLabel(self, text = 'Select', font = (font, 16, 'bold'))
        checkboxlabel.grid(row = 0, column = 0, padx = (10,0), sticky = 'nsew')
        self.checkval = ctk.IntVar(value = 0)
        self.checkbox = ctk.CTkCheckBox(self, variable = self.checkval, text = '', width = 24)
        self.checkbox.grid(row = 1, column = 0, padx = (10,0), pady = (10,0), sticky = 'nsew')

        varlabel = ctk.CTkLabel(self, text = 'Variable', font = (font, 16, 'bold'))
        varlabel.grid(row = 0, column = 1, padx = (10,0), sticky = 'nsew')
        varoptions = ['log(P)', 'P', 'Tc', 'ror', 'log(a/rs)', 'a/rs', 'rhos', 'cos(i)', 'i', 'log(K)', 'K', 'secw', 'sesw', 'e', 'w', 'TT', 'fp',
                    'F0', 'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp', 'u1', 'u2', 'gamma', 'gamma_dot', 'gamma_ddot', 'rv_offset',
                    'eep', 'log10(age)', 'age', 'feh', 'distance', 'AV', 'mstar', 'rstar', 'rhostar']
        self.variable_anchor = ctk.CTkButton(self, text = 'P' if data['Variable'] == '' else data['Variable'].split()[0], font = (font, 18))
        self.variable_anchor.grid(row = 1, column = 1, padx = (10,0), pady = (10,0), sticky = 'nesw')
        self.variable = ScrollableDropdown(self, self.variable_anchor, values = varoptions, height = 400, command = self.varoption_cmd, font = (font, 18))

        self.unitlabel = ctk.CTkLabel(self, text = 'Units', font = (font, 16, 'bold'))
        self.unitlabel.grid(row = 0, column = 2, padx = (10,0), sticky = 'nsew')
        self.unitsvar = ctk.StringVar(value = '')
        self.units = ctk.CTkLabel(self, textvariable = self.unitsvar, font = (font, 16))
        self.units.grid(row = 1, column = 2, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.varnumberlabel_var = ctk.StringVar(value = '')
        self.varnumberlabel = ctk.CTkLabel(self, textvariable = self.varnumberlabel_var, font = (font, 16, 'bold'))
        self.varnumberlabel.grid(row = 0, column = 3, padx = (10,0), sticky = 'nsew')
        self.varnumber_var = ctk.StringVar(value = '' if data['Variable'] == '' else (data['Variable'].split()[1:] if data['Variable'].split()[0] != 'TT' else data['Variable'].split()[1]))
        self.varnumber = ctk.CTkEntry(self,  font = (font, 18), justify = 'center', textvariable = self.varnumber_var)
        self.varnumber.grid(row = 1, column = 3, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.tnumberlabel = ctk.CTkLabel(self, text = 'Transit Number', font = (font, 16, 'bold'))
        self.tnumber = ctk.CTkEntry(self,  font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = '' if data['Variable'] == '' or data['Variable'].split()[0] != 'TT' else data['Variable'].split()[2]))

        priortypelabel = ctk.CTkLabel(self, text = 'Prior Type', font = (font, 16, 'bold'))
        priortypelabel.grid(row = 0, column = 5, padx = (10,0), sticky = 'nsew')
        prioroptions = ['Uniform','Gaussian','Fixed',"Jeffrey's", "Mod. Jeffrey's"]
        self.priortype = ctk.CTkOptionMenu(self, values = prioroptions, font = (font, 18), variable = ctk.StringVar(value = 'Gaussian' if data['Prior Type'] == '' else self.prior_convert[data['Prior Type']]), command = self.priortype_cmd)
        self.priortype.grid(row = 1, column = 5, padx = (10,0), pady = (10,0), sticky = 'nsew')
        self.priortype._dropdown_menu.configure(font = (font, 18))

        self.par1label_var = ctk.StringVar(value = '')
        self.par1label = ctk.CTkLabel(self, textvariable = self.par1label_var, font = (font, 16, 'bold'))
        self.par1label.grid(row = 0, column = 6, padx = (10,0), sticky = 'nsew')
        self.par1_var = ctk.StringVar(value = '' if np.isnan(data['Param 1']) else data['Param 1'])
        self.par1 = FloatEntry(self, textvariable = self.par1_var, font = (font, 18), justify = 'center')
        self.par1.grid(row = 1, column = 6, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.par2label_var = ctk.StringVar(value = '')
        self.par2label = ctk.CTkLabel(self, textvariable = self.par2label_var, font = (font, 16, 'bold'))
        self.par2label.grid(row = 0, column = 7, padx = (10,0), sticky = 'nsew')
        self.par2_var = ctk.StringVar(value = '' if np.isnan(data['Param 2']) else data['Param 2'])
        self.par2 = FloatEntry(self, textvariable = self.par2_var, font = (font, 18), justify = 'center')
        self.par2.grid(row = 1, column = 7, padx = 10, pady = (10,0), sticky = 'nsew')

        self.varoption_cmd(self.variable_anchor.cget('text'), update = False)
        self.priortype_cmd(self.priortype.get(), update = False)


    def varoption_cmd(self, choice, update = True):

        if update:
            self.varnumber_var.set('')
        
        if not self.varnumber.grid_info():
            self.varnumberlabel.grid(row = 0, column = 3, padx = (10,0), sticky = 'nsew')
            self.varnumber.grid(row = 1, column = 3, padx = (10,0), pady = (10,0), sticky = 'nsew')

        if self.tnumber.grid_info():
            self.tnumberlabel.grid_forget()
            self.tnumber.grid_forget()

        self.variable_anchor.configure(text=choice)

        varunits = {'log(P)': 'days', 'P': 'days', 'Tc': 'BJD-2450000', 'ror': '', 'log(a/rs)': '', 'a/rs': '', 'rhos': 'g/cm^3', 'cos(i)': '',
                    'i': 'rads', 'log(K)': 'm/s', 'K': 'm/s', 'secw': '', 'sesw': '', 'e': '', 'w': 'rads', 'TT': 'BJD-2450000', 'fp': '',
                    'F0': '', 'log(rho_gp)': 'days', 'rho_gp': 'days', 'log(sigma_gp)': '', 'sigma_gp': '', 'u1': '', 'u2': '', 'gamma': 'm/s',
                    'gamma_dot': 'm/s/day', 'gamma_ddot': 'm/s/day^2', 'rv_offset': 'm/s', 'eep': '', 'log10(age)': 'yr', 'age': 'yr', 'feh': 'dex',
                    'distance': 'pc', 'AV': 'mag', 'mstar': 'Msun', 'rstar': 'Rsun', 'rhostar': 'g/cm^3'}
        
        self.unitsvar.set(varunits[choice])

        planet_vars = ['log(P)', 'P', 'Tc', 'ror', 'log(a/rs)', 'a/rs', 'rhos', 'cos(i)', 'i', 'log(K)', 'K', 'secw', 'sesw', 'e', 'w', 'fp']
        lc_vars = ['F0', 'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp']
        ld_vars = ['u1', 'u2']

        if choice in planet_vars:
            self.varnumberlabel_var.set('Planet Number\nx for all')

        elif choice == 'TT':
            self.varnumberlabel_var.set('Planet Number')
            self.tnumberlabel.grid(row = 0, column = 4, padx = (10,0), sticky = 'nsew')
            self.tnumber.grid(row = 1, column = 4, padx = (10,0), pady = (10,0), sticky = 'nsew')

        elif choice in lc_vars:
            self.varnumberlabel_var.set('LC Nickname\nx for all')

        elif choice in ld_vars:
            self.varnumberlabel_var.set('Filter\nx for all')

        elif choice == 'rv_offset':
            self.varnumberlabel_var.set('RV Nickname\nx for all')

        else:
            self.varnumberlabel.grid_forget()
            self.varnumber.grid_forget()

        if update:
            self.update_prior_pars()


    def priortype_cmd(self, choice, update = True):

        prior_param_convert = {'Uniform': ['Lower Bound','Upper Bound'],
                               'Gaussian': ['Center', 'Width'],
                               'Fixed': ['Value',''],
                               "Jeffrey's": ['Lower Bound', 'Upper Bound'],
                               "Mod. Jeffrey's": ['Upper Bound', 'Knee Value']}

        self.par1label_var.set(prior_param_convert[choice][0])
        self.par2label_var.set(prior_param_convert[choice][1])

        if choice in ['Fixed']:
            self.par2label.grid_forget()
            self.par2.grid_forget()
        
        elif not self.par2.grid_info():
            self.par2label.grid(row = 0, column = 7, padx = (10,0), sticky = 'nsew')
            self.par2.grid(row = 1, column = 7, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.update_prior_pars(update = update)

    
    def update_prior_pars(self, update = True):

        if update:
            self.par1_var.set('')
            self.par2_var.set('')

        self.par1.min_val = None
        self.par1.max_val = None
        self.par2.min_val = None
        self.par2.max_val = None

        variable = self.variable_anchor.cget('text')
        priortype = self.priortype.get()

        pos_def_vars = ['log(P)', 'P', 'log(a/rs)', 'a/rs', 'rhos', 'log(K)', 'K',
                    'log(rho_gp)', 'rho_gp', 'log(sigma_gp)', 'sigma_gp',
                    'eep', 'log10(age)', 'age', 'distance', 'mstar', 'rstar', 'rhostar']

        if priortype in ["Jeffrey's", "Mod. Jeffrey's"] or variable in pos_def_vars:
            self.par1.min_val = 0
        
        if priortype in ['Gaussian',"Jeffrey's", "Mod. Jeffrey's"] or variable in pos_def_vars:
            self.par2.min_val = 0

        bounded_vars = {'cos(i)':[0,1], 'i':[-np.pi,np.pi], 'secw':[-1,1], 'sesw':[-1,1], 'e':[0,1], 'w':[-np.pi,np.pi], 
                        'u1':[0,1], 'u2':[0,1], 'feh':[-0.5,0.5], 'AV':[0,1]}
        
        if variable in bounded_vars:

            self.par1.max_val = bounded_vars[variable][1]

            if priortype in ["Jeffrey's", "Mod. Jeffrey's"]:

                self.par1.min_val = max(bounded_vars[variable][0],0)
                self.par2.min_val = max(bounded_vars[variable][0],0)
                self.par2.max_val = bounded_vars[variable][1]

            else:

                self.par1.min_val = bounded_vars[variable][0]

            if priortype in ['Uniform']:

                self.par2.min_val = bounded_vars[variable][0]
                self.par2.max_val = bounded_vars[variable][1]
            



class InitPriorsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('1200x700')

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
        self.items_frame.grid_columnconfigure(0, weight = 1)

        self.priors = []

        for i in range(len(self.table)):

            self.add_prior(i)


    def add_prior(self, i):

        prior = PriorFrame(self.items_frame, self.table[i])
        prior.grid(row = i, column = 0, pady = (10,0), sticky = 'ew')
        self.priors.append(prior)


    def add_cmd(self):
        
        self.table.add_row(['']*2 + [np.nan]*2)
        i = len(self.priors)
        self.add_prior(i)


    def copy_cmd(self):

        checked = [i for i, prior in enumerate(self.priors) if prior.checkval.get()]

        for i in checked:
            self.update_table(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.priors),len(self.priors)+len(checked)):
            self.add_prior(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = [i for i, prior in enumerate(self.priors) if prior.checkval.get()]

            if not checked:
                return

            self.table.remove_rows(checked)

            for i in checked:
                self.priors[i].grid_forget()
                self.priors[i].destroy()

            self.priors = [prior for i, prior in enumerate(self.priors) if i not in checked]

            for i in range(len(self.priors)):
                self.priors[i].grid_configure(row = i)
                self.priors[i].data = self.table[i]


    def save_cmd(self):

        for i in range(len(self.priors)):

            self.update_table(i)

        self.initfile.table = self.table

        super().save_cmd()



    def update_table(self, i):

        prior_convert = {'Uniform': 'U', 'Gaussian': 'G', 'Fixed': 'F', "Jeffrey's": 'J', "Mod. Jeffrey's": 'MJ'}

        prior = self.priors[i]

        self.table['Variable'] = update_string_col(self.table['Variable'], prior.variable_anchor.cget('text').strip() + ' ' + prior.varnumber.get().strip() + (' ' + prior.tnumber.get().strip() if prior.tnumber.get().strip() != '' else ''), i)
        self.table['Prior Type'] = update_string_col(self.table['Prior Type'], prior_convert[prior.priortype.get().strip()], i)
        self.table['Param 1'][i] = prior.par1.return_float()
        self.table['Param 2'][i] = prior.par2.return_float()



###################
#####Init Star#####
###################


class StarFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row, value_min=None, value_max=None, **kwargs):
        super().__init__(master, **kwargs)

        self.data = data
        self.value_min = value_min
        self.value_max = value_max

        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(list(range(5)), weight = 1)

        self.labelframe = ctk.CTkFrame(self, fg_color = 'transparent')
        self.labelframe.grid(row = 0, column = 0, sticky = 'nsew')
        self.labelframe.grid_columnconfigure(1, weight = 1)
        self.labelframe.grid_rowconfigure((0,1), weight = 1)

        self.checkval = ctk.IntVar(value = 0)
        self.check = ctk.CTkCheckBox(self.labelframe, text = '', variable = self.checkval, width = 24)
        self.check.grid(row = 0, column = 0, rowspan = 2, sticky = 'nsew')

        self.par = ctk.CTkLabel(self.labelframe, text = data['Parameter'], font = (font, 18, 'bold'))
        self.par.grid(row = 0, column = 1, sticky = 'nsew')

        self.unit = ctk.CTkLabel(self.labelframe, text = data['Units'], font = (font, 16))
        self.unit.grid(row = 1, column = 1, sticky = 'nsew')

        valuelabel = ctk.CTkLabel(self, text = 'Value', font = (font, 18, 'bold'))
        valuelabel.grid(row = 1, column = 0, pady = (10,0), sticky = 'nsew')

        self.value_var = ctk.StringVar(value = '' if np.isnan(data['Value']) else data['Value'])
        self.value = FloatEntry(self, min_val = self.value_min, max_val = self.value_max, textvariable = self.value_var, font = (font, 18), justify = 'center')
        self.value.grid(row = 2, column = 0, sticky = 'nsew')

        errorlabel = ctk.CTkLabel(self, text = 'Error', font = (font, 18, 'bold'))
        errorlabel.grid(row = 3, column = 0, pady = (10,0), sticky = 'nsew')
        
        self.error_var = ctk.StringVar(value = '' if np.isnan(data['Error']) else data['Error'])
        self.error = FloatEntry(self, min_val = 0, textvariable = self.error_var, font = (font, 18), justify = 'center')
        self.error.grid(row = 4, column = 0, sticky = 'nsew')


class PhotFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row, **kwargs):
        super().__init__(master, **kwargs)

        self.data = data

        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(list(range(5)), weight = 1)

        photoptions = ['Gaia G','Gaia BP','Gaia RP','2MASS J','2MASS H','2MASS K','Wise 1','Wise 2','Wise 3','SDSS u','SDSS g','SDSS r','SDSS i','SDSS z','U','B','V','R','I','TESS']

        self.band = CheckDropDown(self, photoptions, data['Parameter'])
        self.band.grid(row = 0, column = 0, sticky = 'nsew')

        valuelabel = ctk.CTkLabel(self, text = 'Value', font = (font, 18, 'bold'))
        valuelabel.grid(row = 1, column = 0, pady = (10,0), sticky = 'nsew')

        self.value_var = ctk.StringVar(value = '' if np.isnan(data['Value']) else data['Value'])
        self.value = FloatEntry(self, textvariable = self.value_var, font = (font, 18), justify = 'center')
        self.value.grid(row = 2, column = 0, sticky = 'nsew')

        errorlabel = ctk.CTkLabel(self, text = 'Error', font = (font, 18, 'bold'))
        errorlabel.grid(row = 3, column = 0, pady = (10,0), sticky = 'nsew')
        
        self.error_var = ctk.StringVar(value = '' if np.isnan(data['Error']) else data['Error'])
        self.error = FloatEntry(self, min_val = 0, textvariable = self.error_var, font = (font, 18), justify = 'center')
        self.error.grid(row = 4, column = 0, sticky = 'nsew')


##############query only for selected

class StarQueryPrompt(ctk.CTkToplevel):
    def __init__(self, initgui: InitStarGUI):
        super().__init__()

        self.initgui = initgui

        self.title('Query Stellar Pars')
        self.geometry('300x200')
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(list(range(3)), weight = 1)

        self.label = ctk.CTkLabel(self, text = 'Enter Catalog ID', font = (font, 18, 'bold'))
        self.label.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.entry_var = ctk.StringVar(value = '')
        self.entry = ctk.CTkEntry(self, font = (font, 18), justify = 'center', textvariable = self.entry_var)
        self.entry.grid(row = 1, column = 0, padx = 10, pady = (10, 0), sticky = 'ew')

        self.run = ctk.CTkButton(self, text = 'Run Query', font = (font, 18, 'bold'), command = self.run_cmd)
        self.run.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'ew')

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()
        self.grab_set()

        self.after(200, lambda: self.attributes("-topmost", False))

    def run_cmd(self):

        cid = self.entry_var.get().strip()

        bands = []
        checked = []
        for phot in self.initgui.phots:

            bands.append(phot.band.dropdown_anchor.cget('text').strip())
            checked.append(phot.band.checkval.get())

        vizier = Vizier(columns = ['**','+_r'])

        try:

            tic = vizier.query_object(object_name = cid, catalog = 'IV/39/tic82', radius = 1*u.arcmin)[0][0]

            if self.initgui.r.checkval.get():
                self.initgui.r.value_var.set(tic['Rad'])
                self.initgui.r.value.configure(text_color = '#FFFFFF')
                self.initgui.r.error_var.set(tic['s_Rad'])
                self.initgui.r.error.configure(text_color = '#FFFFFF')

            if self.initgui.m.checkval.get():
                self.initgui.m.value_var.set(tic['Mass'])
                self.initgui.m.value.configure(text_color = '#FFFFFF')
                self.initgui.m.error_var.set(tic['s_Mass'])
                self.initgui.m.error.configure(text_color = '#FFFFFF')

            if self.initgui.t.checkval.get():
                self.initgui.t.value_var.set(tic['Teff'])
                self.initgui.t.value.configure(text_color = '#FFFFFF')
                self.initgui.t.error_var.set(tic['s_Teff'])
                self.initgui.t.error.configure(text_color = '#FFFFFF')

            if self.initgui.logg.checkvalue.get():
                self.initgui.logg.value_var.set(tic['logg'])
                self.initgui.logg.value.configure(text_color = '#FFFFFF')
                self.initgui.logg.error_var.set(tic['s_logg'])
                self.initgui.logg.error.configure(text_color = '#FFFFFF')

            ticconvert = {'TESS': 'Tmag', 'B': 'Bmag', 'V': 'Vmag'}

            for i in range(len(bands)):

                band = bands[i]

                if band in ticconvert and checked[i]:

                    phot = self.initgui.phots[i]

                    phot.value_var.set(tic[ticconvert[band]])
                    phot.error_var.set(tic['e_'+ticconvert[band]])
                    phot.error.configure(text_color = '#FFFFFF')


        except ConnectionError or TimeoutError:

            self.label.configure(text = 'Connection issue.')
            return

        except:

            pass

        try:

            gaia = vizier.query_object(object_name = cid, catalog = 'I/355/gaiadr3', radius = 1*u.arcmin)[0][0]

            if self.initgui.feh.checkval.get():
                self.initgui.feh.value_var.set(gaia['[Fe/H]'])
                self.initgui.feh.error_var.set((gaia['B_[Fe/H]']-gaia['b_[Fe/H]'])/2)
                self.initgui.feh.error.configure(text_color = '#FFFFFF')

            if self.initgui.plx.checkval.get():
                self.initgui.plx.value_var.set(gaia['Plx'])
                self.initgui.plx.value.configure(text_color = '#FFFFFF')
                self.initgui.plx.error_var.set(gaia['e_Plx'])
                self.initgui.plx.error.configure(text_color = '#FFFFFF')

            gaiaconvert = {'Gaia G': 'Gmag', 'Gaia BP': 'BPmag', 'Gaia RP': 'RPmag'}

            for i in range(len(bands)):

                band = bands[i]

                if band in gaiaconvert and checked[i]:

                    phot = self.initgui.phots[i]

                    phot.value_var.set(gaia[gaiaconvert[band]])
                    phot.error_var.set(gaia['e_'+gaiaconvert[band]])
                    phot.error.configure(text_color = '#FFFFFF')

        except:

            pass
        
        twomassconvert = {'2MASS J': 'Jmag', '2MASS H': 'Hmag', '2MASS K': 'Kmag'}

        if np.any(np.isin(bands, list(twomassconvert.keys()))):

            try:

                twomass = vizier.query_object(object_name = cid, catalog = 'II/246/out', radius = 1*u.arcmin)[0][0]

                for i in range(len(bands)):

                    band = bands[i]

                    if band in twomassconvert and checked[i]:

                        phot = self.initgui.phots[i]

                        phot.value_var.set(twomass[twomassconvert[band]])
                        phot.error_var.set(twomass['e_'+twomassconvert[band]])
                        phot.error.configure(text_color = '#FFFFFF')

            except:

                pass

        wiseconvert = {'Wise 1': 'W1mag', 'Wise 2': 'W2mag', 'Wise 3': 'W3mag'}

        if np.any(np.isin(bands, list(wiseconvert.keys()))):

            try:

                wise = vizier.query_object(object_name = cid, catalog = 'II/328/allwise', radius = 1*u.arcmin)[0][0]

                for i in range(len(bands)):

                    band = bands[i]

                    if band in wiseconvert and checked[i]:

                        phot = self.initgui.phots[i]

                        phot.value_var.set(wise[wiseconvert[band]])
                        phot.error_var.set(wise['e_'+wiseconvert[band]])
                        phot.error.configure(text_color = '#FFFFFF')

            except:

                pass

        sdssconvert = {'SDSS u': 'upmag', 'SDSS g': 'gpmag', 'SDSS r': 'rpmag', 'SDSS i': 'ipmag', 'SDSS z': 'zpmag'}

        if np.any(np.isin(bands, list(sdssconvert.keys()))):

            try:

                sdss = vizier.query_object(object_name = cid, catalog = 'V/154/sdss16', radius = 1*u.arcmin)[0][0]

                for i in range(len(bands)):

                    band = bands[i]

                    if band in sdssconvert and checked[i]:

                        phot = self.initgui.phots[i]

                        phot.value_var.set(sdss[sdssconvert[band]])
                        phot.error_var.set(sdss['e_'+sdssconvert[band]])
                        phot.error.configure(text_color = '#FFFFFF')

            except:

                pass


        self.destroy()



class InitStarGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('1000x750')

        self.phot_convert = {'Gaia G': 'G', 'Gaia BP': 'BP', 'Gaia RP': 'RP', '2MASS J': 'J', '2MASS H': 'H', '2MASS K': 'K',
                        'Wise 1': 'W1', 'Wise 2': 'W2', 'Wise 3': 'W3', 'SDSS u': 'u', 'SDSS g': 'g', 'SDSS r': 'r', 'SDSS i': 'i', 'SDSS z': 'z',
                        'U': 'U', 'B': 'B', 'V': 'V', 'R': 'R', 'I': 'I','TESS': 'TESS'}

        self.setup_items_frame()

        self.frame_title = ctk.CTkLabel(self, text = 'Stellar Parameters', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'ew', columnspan = 2)

        self.query = ctk.CTkButton(self, text = 'Query for Selected Values', font = (font, 18, 'bold'), command = self.query_cmd)
        self.query.grid(row = 2, column = 0, padx = 100, pady = (10, 0), sticky = 'ew', columnspan = 2)

        self.add.configure(text = 'Add Photometry')
        self.add.grid_forget()
        self.add.grid(row = 3, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete Selected Photometry', font = (font, 18, 'bold'), command = self.delete_cmd)
        self.delete.grid(row = 3, column = 1, padx = 10, pady = (10,0), sticky = 'ew')

    
    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self)
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nsew', columnspan = 2)
        self.items_frame.grid_columnconfigure(list(range(3)), weight = 1)

        self.r = StarFrame(self.items_frame, self.table[0], value_min = 0)
        self.r.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'nsew')

        self.m = StarFrame(self.items_frame, self.table[1], value_min = 0)
        self.m.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = 'nsew')

        self.t = StarFrame(self.items_frame, self.table[2], value_min = 0)
        self.t.grid(row = 0, column = 2, padx = 10, pady = 10, sticky = 'nsew')

        self.logg = StarFrame(self.items_frame, self.table[3], value_min = 0)
        self.logg.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'nsew')

        self.feh = StarFrame(self.items_frame, self.table[4])
        self.feh.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = 'nsew')

        self.plx = StarFrame(self.items_frame, self.table[5], value_min = 0)
        self.plx.grid(row = 1, column = 2, padx = 10, pady = 10, sticky = 'nsew')

        photlabel = ctk.CTkLabel(self.items_frame, text = 'Photometry (mags)', font = (font, 18, 'bold'))
        photlabel.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'nsew', columnspan = 3)

        self.phots = []
        
        for i in range(6, len(self.table)):

            self.add_phot(i)

    
    def add_phot(self, i):

        phot = PhotFrame(self.items_frame, self.table[i])
        phot.grid(row = int(i//3)+1, column = i%3, padx = 10, pady = 10, sticky = 'nsew')
        self.phots.append(phot)


    def add_cmd(self):
        
        self.table.add_row(['Band', 'mag'] + [np.nan]*2)
        i = len(self.phots)+6
        self.add_phot(i)


    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = [i for i, phot in enumerate(self.phots) if phot.band.checkval.get()]

            if not checked:
                return

            self.table.remove_rows(checked)

            for i in checked:
                self.phots[i].grid_forget()
                self.phots[i].destroy()

            self.phots = [phot for i, phot in enumerate(self.phots) if i not in checked]

            for i in range(len(self.phots)):
                self.phots[i].grid_configure(row = int(i//3)+3, column = i%3)
                self.phots[i].data = self.table[i]

    

    def save_cmd(self):

        self.table['Value'][0] = self.r.value.return_float()
        self.table['Error'][0] = self.r.error.return_float()

        self.table['Value'][1] = self.m.value.return_float()
        self.table['Error'][1] = self.m.error.return_float()

        self.table['Value'][2] = self.t.value.return_float()
        self.table['Error'][2] = self.t.error.return_float()

        self.table['Value'][3] = self.logg.value.return_float()
        self.table['Error'][3] = self.logg.error.return_float()

        self.table['Value'][4] = self.feh.value.return_float()
        self.table['Error'][4] = self.feh.error.return_float()

        self.table['Value'][5] = self.plx.value.return_float()
        self.table['Error'][5] = self.plx.error.return_float()

        for i in range(len(self.phots)):

            phot = self.phots[i]

            j = i + 6

            self.table['Parameter'] = update_string_col(self.table['Parameter'], self.phot_convert[phot.band.dropdown_anchor.cget('text').strip()], j)
            self.table['Value'][j] = phot.value.return_float()
            self.table['Error'][j] = phot.error.return_float()

        self.initfile.table = self.table

        super().save_cmd()

    
    def query_cmd(self):

        query_prompt = StarQueryPrompt(self)

        self.wait_window(query_prompt)




#################
#####Init LD#####
#################

class LDFrame(ctk.CTkFrame):
    def __init__(self, master, data: Row, **kwargs):
        super().__init__(master, **kwargs)

        self.data = data

        self.grid_columnconfigure(list(range(5)), weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        self.checkval = ctk.IntVar(value = 0)
        self.checkbox = ctk.CTkCheckBox(self, variable = self.checkval, text = '')
        self.checkbox.grid(row = 0, column = 0, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.filter_frame = ctk.CTkFrame(self)
        self.filter_frame.grid(row = 0, column = 1, padx = (10,0), pady = (10,0), sticky = 'nsew')
        self.filter_frame.grid_columnconfigure(0, weight = 1)

        self.filter = ctk.CTkOptionMenu(self.filter_frame, values = ld_grid_list+['Other'], font = (font, 18), variable = ctk.StringVar(value = 'Other' if data['Filter'] not in ld_grid_list else data['Filter']), state = 'readonly', command = self.other_filter)
        self.filter.grid(row = 0, column = 0, sticky = 'nsew')
        self.filter._dropdown_menu.configure(font = (font, 18))

        self.otherfilter = ctk.CTkEntry(self.filter_frame, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = ''))

        self.u1_var = ctk.StringVar(value = '' if np.isnan(data['u1']) else data['u1'])
        self.u1 = FloatEntry(self, min_val = 0, max_val = 1, textvariable = self.u1_var, font = (font, 18), justify = 'center')
        self.u1.grid(row = 0, column = 2, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.u2_var = ctk.StringVar(value = '' if np.isnan(data['u2']) else data['u2'])
        self.u2 = FloatEntry(self, min_val = 0, max_val = 1, textvariable = self.u2_var, font = (font, 18), justify = 'center')
        self.u2.grid(row = 0, column = 3, padx = (10,0), pady = (10,0), sticky = 'nsew')

        self.generate = ctk.CTkButton(self, text = 'Generate', font = (font, 18, 'bold'), command = self.gen_cmd)
        self.generate.grid(row = 0, column = 4, padx = 10, pady = (10,0), sticky = 'nsew')

        if self.filter.get() == 'Other':
            self.otherfilter.set(data['Filter'])
            self.other_filter('Other')


    def gen_cmd(self):

        if self.filter.get() == 'Other':
            return

        gen_prompt = LDGenPrompt(self.filter.get())
        
        self.wait_window(gen_prompt)

        if hasattr(gen_prompt, 'u1'):

            self.u1_var.set(gen_prompt.u1)
            self.u2_var.set(gen_prompt.u2)

            self.u1.configure(text_color = '#FFFFFF')
            self.u2.configure(text_color = '#FFFFFF')

    def other_filter(self, choice):

        if choice == 'Other' and not self.otherfilter.grid_info():

            self.otherfilter.grid(row = 1, column = 0, pady = (10,0), sticky = 'nsew')

            self.generate.configure(fg_color = "#CE2424", hover_color = '#CE2424')

        elif self.otherfilter.grid_info():

            self.otherfilter.grid_forget()

            self.generate.configure(fg_color = '#1F6AA5', hover_color = '#144870')




class LDGenPrompt(ctk.CTkToplevel):
    def __init__(self, filter: str):
        super().__init__()

        self.filter = filter
        
        self.title('Gen LD Pars')
        self.geometry('300x300')
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(list(range(7)), weight = 1)

        self.tlabel = ctk.CTkLabel(self, text = 'Teff (K) [2300,7800]', font = (font, 18, 'bold'))
        self.tlabel.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.t = FloatEntry(self, min_val = 2300, max_val = 7800, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = ''))
        self.t.grid(row = 1, column = 0, padx = 10, sticky = 'ew')

        self.logglabel = ctk.CTkLabel(self, text = 'log(g) (cgs) [3.0,6.0]', font = (font, 18, 'bold'))
        self.logglabel.grid(row = 2, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.logg = FloatEntry(self, min_val = 3, max_val = 6, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = ''))
        self.logg.grid(row = 3, column = 0, padx = 10, sticky = 'ew')

        self.fehlabel = ctk.CTkLabel(self, text = 'Fe/H (dex) [-0.5,0.5]', font = (font, 18, 'bold'))
        self.fehlabel.grid(row = 4, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.feh = FloatEntry(self, min_val = -0.5, max_val = 0.5, font = (font, 18), justify = 'center', textvariable = ctk.StringVar(value = ''))
        self.feh.grid(row = 5, column = 0, padx = 10, sticky = 'ew')

        self.run = ctk.CTkButton(self, text = 'Generate', font = (font, 18, 'bold'), command = self.run_cmd)
        self.run.grid(row = 6, column = 0, padx = 10, pady = 10, sticky = 'ew')

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()
        self.grab_set()

        self.after(200, lambda: self.attributes("-topmost", False))


    def run_cmd(self):

        x = calc_ld(self.filter, self.t.return_float(), self.logg.return_float(), self.feh.return_float())
        self.u1 = x[0][0]
        self.u2 = x[1][0]

        self.destroy()


class InitLDGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('700x700')

        self.setup_items_frame()

        self.frame_title = ctk.CTkLabel(self, text = 'Limb Darkening Parameters', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = 'ew', columnspan = 2)

        self.copy = ctk.CTkButton(self, text = 'Copy Selected', font = (font, 18, 'bold'), command = self.copy_cmd)
        self.copy.grid(row = 3, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete Selected', font = (font, 18, 'bold'), command = self.delete_cmd)
        self.delete.grid(row = 3, column = 1, padx = 10, pady = (10,0), sticky = 'ew')


    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self)
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nsew', columnspan = 2)
        self.items_frame.grid_columnconfigure(list(range(5)), weight = 1)

        labels = ['Select','Filter','  u1  ','  u2  ','           ']
        for i in range(5):
            label = ctk.CTkLabel(self.items_frame, text = labels[i], font = (font, 18, 'bold'))
            label.grid(row = 0, column = i, padx = (10,0) if i < 4 else (10 if i > 0 else 0), pady = (10,0), sticky = 'nsew')

        self.lds = []

        for i in range(len(self.table)):

            self.add_ld(i)


    def add_ld(self, i):

        ld = LDFrame(self.items_frame, self.table[i])
        ld.grid(row = i+1, column = 0, pady = (10,0), sticky = 'ew', columnspan = 5)
        self.lds.append(ld)


    def add_cmd(self):
        
        self.table.add_row([''] + [np.nan]*2)
        i = len(self.lds)
        self.add_ld(i)


    def copy_cmd(self):

        checked = [i for i, ld in enumerate(self.lds) if ld.checkval.get()]

        for i in checked:
            self.update_table(i)

        tablecopy = self.table[checked]

        self.initfile.table = vstack([self.table, tablecopy])
        self.table = self.initfile.table

        for i in range(len(self.lds),len(self.lds)+len(checked)):
            self.add_ld(i)

    
    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = [i for i, ld in enumerate(self.lds) if ld.checkval.get()]

            if not checked:
                return

            self.table.remove_rows(checked)

            for i in checked:
                self.lds[i].grid_forget()
                self.lds[i].destroy()

            self.lds = [ld for i, ld in enumerate(self.lds) if i not in checked]

            for i in range(len(self.lds)):
                self.lds[i].grid_configure(row = i+1)
                self.lds[i].data = self.table[i]


    def save_cmd(self):

        for i in range(len(self.lds)):

            self.update_table(i)

        self.initfile.table = self.table

        super().save_cmd()


    def update_table(self, i):

        ld = self.lds[i]

        filter = ld.filter.get().strip()
        if filter == 'Other':
            filter = ld.otherfilter.get().strip()

        self.table['Filter'] = update_string_col(self.table['Filter'], filter, i)
        self.table['u1'][i] = ld.u1.return_float()
        self.table['u2'][i] = ld.u2.return_float()




###################
#####Init TTVs#####
###################

class TTVGUI(ctk.CTkToplevel):
    def __init__(self, data: Column, close_cmd):
        super().__init__()

        self.data = data
        self.close_cmd = close_cmd

        self.title('Planet {0} TTVs'.format(data.name))
        self.geometry('500x600')
        self.grid_columnconfigure((0,1), weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        self.setup_frame()

        self.add = ctk.CTkButton(self, text = 'Add', command = self.add_cmd, font = (font, 18, 'bold'))
        self.add.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete Selected', command = self.delete_cmd, font = (font, 18, 'bold'))
        self.delete.grid(row = 1, column = 1, padx = 10, pady = (10,0), sticky = 'ew')

        self.save = ctk.CTkButton(self, text = 'Save', command = self.save_cmd, font = (font, 18, 'bold'))
        self.save.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = "ew", columnspan = 2)

        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()
        self.focus_force()

        self.after(200, lambda: self.attributes("-topmost", False))


    def setup_frame(self):

        self.frame = ctk.CTkScrollableFrame(self)
        self.frame.grid(row = 0, column = 0, sticky = 'nsew', columnspan = 2)
        self.frame.grid_columnconfigure(1, weight = 1)

        labels = ['Select','Transit Time\n(BJD-2450000)']
        for i in range(2):

            label = ctk.CTkLabel(self.frame, text = labels[i], font = (font, 18, 'bold'))
            label.grid(row = 0, column = i, padx = (10,0) if i < 1 else 10, sticky = 'ew')

        self.checkvals = []
        self.checks = []
        self.tts = []
        for i in range(len(self.data)):

            if np.isnan(self.data[i]):
                break

            self.add_tt(i)


    def add_tt(self, i):

        checkval = ctk.IntVar(value = 0)
        self.checkvals.append(checkval)

        check = ctk.CTkCheckBox(self.frame, variable = checkval, text = '')
        check.grid(row = i+1, column = 0, padx = (10,0), pady = (10,0), sticky = 'ew')
        self.checks.append(check)

        tt = FloatEntry(self.frame, textvariable = ctk.StringVar(value = '' if np.isnan(self.data[i]) else self.data[i]), font = (font, 18), justify = 'center')
        tt.grid(row = i+1, column = 1, padx = 10, pady = (10,0), sticky = 'nsew')
        self.tts.append(tt)



    def add_cmd(self):

        if len(self.tts) >= len(self.data):
            
            self.data.info.parent_table.add_row([np.nan]*len(self.data.info.parent_table.colnames))
            self.data = self.data.info.parent_table[self.data.name]

        i = len(self.tts)
        self.add_tt(i)


    def delete_cmd(self):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            checked = [i for i, c in enumerate(self.checkvals) if c.get()]

            if not checked:
                return

            vals = self.data.data
            vals = np.delete(vals, checked)
            vals = np.append(vals, [np.nan]*(len(checked)))
            self.data.info.parent_table[self.data.name] = vals
            self.data = self.data.info.parent_table[self.data.name]

            for i in checked:
                self.checks[i].grid_forget()
                self.checks[i].destroy()
                self.tts[i].grid_forget()
                self.tts[i].destroy()

            self.checkvals = [c for i, c in enumerate(self.checkvals) if i not in checked]
            self.checks = [c for i, c in enumerate(self.checks) if i not in checked]
            self.tts = [tt for i, tt in enumerate(self.tts) if i not in checked]

            for i in range(len(self.tts)):
                self.checks[i].grid_configure(row = i+1)
                self.tts[i].grid_configure(row = i+1)



    def save_cmd(self):

        for i in range(len(self.tts)):
            self.data[i] = self.tts[i].return_float()

        self.close_cmd()



class TTVFrame(ctk.CTkFrame):
    def __init__(self, master, topmaster, data: Column, idx: int):
        super().__init__(master)

        self.grid_columnconfigure(list(range(1,3)), weight = 1)

        self.data = data
        self.topmaster = topmaster
        self.idx = idx

        self.num = ctk.CTkEntry(self, textvariable = ctk.StringVar(value = data.name), font = (font, 18, 'bold'), justify = 'center')
        self.num.grid(row = 0, column = 0, padx = 10, sticky = 'ew')
        validate_int_cmd = self.num.register(self.validate_int_input)
        self.num.configure(validate = 'key', validatecommand = (validate_int_cmd, '%P'))

        self.edit = ctk.CTkButton(self, text = 'Edit', command = self.edit_cmd, font = (font, 18, 'bold'))
        self.edit.grid(row = 0, column = 1, padx = 10, sticky = 'ew')

        self.delete = ctk.CTkButton(self, text = 'Delete', command = self.delete_cmd, font = (font, 18, 'bold'))
        self.delete.grid(row = 0, column = 2, padx = 10, sticky = 'ew')


    def validate_int_input(self, proposed_text):
    
        if proposed_text == "":
            return True
        
        try:
            val = int(proposed_text)
            self.data.name = str(val)
            self.num.configure(text_color = '#FFFFFF')
            return True

        except KeyError:

            self.num.configure(text_color = '#CB2626')
            return True
        
        except ValueError:
            return False


    def edit_cmd(self):

        if not hasattr(self, 'edit_ttvs'):

            self.edit_ttvs = TTVGUI(self.data, self.save_close)
            self.edit_ttvs.protocol("WM_DELETE_WINDOW", self.edit_close)

            self.wait_window(self.edit_ttvs)

        else:

            self.edit_ttvs.withdraw()
            self.edit_ttvs.update()
            self.edit_ttvs.deiconify()
            self.edit_ttvs.lift()
            self.edit_ttvs.focus()


    def save_close(self):

        self.data = self.edit_ttvs.data
        self.edit_ttvs.destroy()
        delattr(self, 'edit_ttvs')

    
    def edit_close(self):

        self.edit_ttvs.destroy()
        delattr(self, 'edit_ttvs')


    def delete_cmd(self):

        self.topmaster.delete_cmd(self.idx)



class InitTTVsGUI(InitGUI):
    def __init__(self, initfile: InitFile):
        super().__init__(initfile)

        self.geometry('600x600')

        self.setup_items_frame()

        self.frame_title = ctk.CTkLabel(self, text = 'Planets with TTVs', font = (font, 18, 'bold'))
        self.frame_title.grid(row = 0, column = 0, padx = 10, sticky = 'ew', columnspan = 2)

    def setup_items_frame(self):

        self.items_frame = ctk.CTkScrollableFrame(self)
        self.items_frame.grid(row = 1, column = 0, padx = 10, pady = (10,0), sticky = 'nesw', columnspan = 2)
        self.items_frame.grid_columnconfigure(0, weight = 1)

        self.planets = []

        for i in range(len(self.table.colnames)):

            self.add_planet(i)

    def add_planet(self, i):

        planet = TTVFrame(self.items_frame, self, self.table[self.table.colnames[i]], i)
        planet.grid(row = i, column = 0, pady = (10,0), sticky = 'ew')
        self.planets.append(planet)


    def add_cmd(self):

        try:
            col = Column(data = [np.nan]*len(self.table), name = 'X', unit = 'BJD-2450000', dtype = float)
            self.table.add_column(col)
            i = len(self.planets)
            self.add_planet(i)
            self.planets[i].edit_cmd()

        except ValueError:

            i = np.where(np.array(self.table.colnames) == 'X')[0][0]
            self.planets[i].num.configure(text_color = '#CB2626')



    def delete_cmd(self, i):

        delete_prompt = DeletePrompt()

        self.wait_window(delete_prompt)

        if delete_prompt.answer:

            self.table.remove_column(self.table.colnames[i])
            self.planets[i].grid_forget()
            self.planets.pop(i)

            for j in range(len(self.planets)):
                self.planets[j].grid_configure(row = j)
                self.planets[j].idx = j
                self.planets[j].data = self.table[self.table.colnames[j]]


    def save_cmd(self):

        for i in range(len(self.table)-1, -1, -1):

            if np.all(np.isnan(list(self.table[i]))):

                self.table.remove_row(i)

        super().save_cmd()