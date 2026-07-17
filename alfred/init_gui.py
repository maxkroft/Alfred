import customtkinter as ctk
ctk.set_appearance_mode('dark')
ctk.DrawEngine.preferred_drawing_method = "circle_shapes"

from astropy.table import Table, Row
import numpy as np


class FloatEntry(ctk.CTkFrame):
    def __init__(self, master, label, row, col, min_val = None, max_val = None, **kwargs):
        super().__init__(master, fg_color = 'transparent', **kwargs)

        self.row = row
        self.col = col
        self.min_val = min_val
        self.max_val = max_val

        self.grid_columnconfigure(0, weight = 1)

        validate_float_cmd = self.register(self.validate_float_input)

        self.label = ctk.CTkLabel(self, text = label, font = ('Arial',18,'bold'))
        self.entry = ctk.CTkEntry(self, font = ('Arial', 18), justify = 'center', validate = 'key', validatecommand = (validate_float_cmd, '%P'))

        self.label.grid(row = 0, column = 0, pady = (10,0))
        self.entry.grid(row = 1, column = 0, sticky = "ew", padx = 10)


    def validate_float_input(self, proposed_text):
        """Single reusable function that validates text typing and min/max limits."""

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
    



class PlanetGUI(ctk.CTk):
    def __init__(self, data: Row = None):
        super().__init__()

        self.data = data

        self.title('Add a Planet')
        self.geometry('500x600')
        self.grid_columnconfigure((0,1), weight = 1)

        self.p_entry = FloatEntry(self, "P (days)", 1, 0, min_val = 0)
        if self.data is not None:
            self.p_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['Period']) else self.data['Period']))
        self.p_entry.grid(row = self.p_entry.row, column = self.p_entry.col, sticky = "ew", padx = 10)

        self.t_entry = FloatEntry(self, "Tc (BJD-2450000)", 1, 1)
        if self.data is not None:
            self.t_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['Tc']) else self.data['Tc']))
        self.t_entry.grid(row = self.t_entry.row, column = self.t_entry.col, sticky = "ew", padx = 10)

        self.r_entry = FloatEntry(self, "Rp/Rstar", 2, 0, min_val = 0)
        if self.data is not None:
            self.r_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['Rp/Rs']) else self.data['Rp/Rs']))

        self.a_entry = FloatEntry(self, "a/Rstar", 2, 1, min_val = 0)
        if self.data is not None:
            self.a_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['a/Rs']) else self.data['a/Rs']))

        self.i_entry = FloatEntry(self, "cos(i)", 3, 0, min_val = 0, max_val = 1)
        if self.data is not None:
            self.i_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['cos(i)']) else self.data['cos(i)']))

        self.k_entry = FloatEntry(self, "K (m/s)", 3, 1, min_val = 0)
        if self.data is not None:
            self.k_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['K']) else self.data['K']))

        self.secw_entry = FloatEntry(self, "sqrt(e)cos(w) (-1 to 1)", 4, 0, min_val = -1, max_val = 1)
        if self.data is not None:
            self.secw_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['sqrt(e)cos(w)']) else self.data['sqrt(e)cos(w)']))

        self.sesw_entry = FloatEntry(self, "sqrt(e)sin(w) (-1 to 1)", 4, 1, min_val = -1, max_val = 1)
        if self.data is not None:
            self.sesw_entry.entry.configure(textvariable = ctk.StringVar(value = '' if np.isnan(self.data['sqrt(e)sin(w)']) else self.data['sqrt(e)sin(w)']))

        self.f_entry = FloatEntry(self, "Fp/Fstar", 5, 0, min_val = 0)
        if self.data is not None:
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


tab = Table.read('/mnt/c/Users/kroft/Documents/Data/7690/init_planets.txt', format = 'ascii.fixed_width_two_line', header_rows = ['name','unit'], delimiter = '|', converters = {'*': [int, float, bool, str]})
app = PlanetGUI(data = tab[0])
app.mainloop()