""" 
This file has the codes to plot the response surface formed by the Monte-Carlo simulations
"""

# ===========================
# Library Imports
# ===========================

import os, sys, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===========================
# Custom Imports
# ===========================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

class RSM_2():

    def __init__(self, model, var1, var2):
        self.model = model
        self.var_1 = var1 # variable for the input file name
        self.var = var2 # variable for the plotting
        self.setup()

    def setup(self):

        with open('deviation_order.yaml', 'r') as file:
            order_dict = yaml.safe_load(file)

        yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')

        with open(yaml_file_path, 'r') as file:
            dict = yaml.safe_load(file)
            self.symbols = dict['Symbols'][self.model]
            self.units = dict['units'][self.model]

        # dictionary of all the parameters and the corresponding errors
        temp_dict = order_dict[self.model][self.var_1]

        # Sort the dictionary by value and get the top three keys
        top_three = sorted(temp_dict, key=temp_dict.get, reverse=True)[:3]
        self.model_vars = list(top_three)

        if self.model in ['newell', 'bando']:
            try:
                data = pd.read_csv(f'{self.model}_rsm_{self.var_1}.csv') # This needs more attention in teh formation block
            except KeyError:
                raise SystemExit('The response surface for this variable does not exist')
            
        elif self.model in ['idm', 'fvdm', 'shamoto', 'tsh', 'gfm']:
            data = pd.read_csv(f'{self.model}_rsm_{self.var_1}.csv')

        # Extract the parameters that were extracted
        self.x1 = data[self.model_vars[0]].to_numpy()
        self.x2 = data[self.model_vars[1]].to_numpy()
        self.x3 = data[self.model_vars[2]].to_numpy()

        if self.var == 'vkt':
            self.error = data['err_x'].to_numpy()/1000
        elif self.var == 'vel':
            self.error = data['err_v'].to_numpy()

    def plotting(self, starting_index, el, az):

        fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, dpi=600)

        # Set a consistent view angle for all subplots
        view_angle_elev = el  # Elevation angle
        view_angle_azim = az  # Azimuth angle

        colormap = 'rainbow'

        if starting_index == 0:
            x, y = self.x1, self.x2
        elif starting_index == 1:
            x, y = self.x2, self.x3
        else:
            x, y = self.x3, self.x1

        # First subplot (acc_exp vs safe_head)
        sc1 = axs.scatter(x, y, self.error, c=self.error, cmap=colormap, marker='o', s=1, alpha=1)
        axs.set_xlabel(self.symbols[self.model_vars[starting_index]] +'(' +
                    self.units[self.model_vars[starting_index]] + ')', fontsize = 15)
        axs.set_ylabel(self.symbols[self.model_vars[(starting_index + 1)%3]] + '(' + 
                    self.units[self.model_vars[starting_index]]+')', fontsize = 15)
        
        if self.var == 'vkt':
            axs.set_zlabel('$MAD_{vkt}$ (km)', fontsize = 15)
        if self.var == 'vel':
            axs.set_zlabel('$MAD_v$  (m/s)', fontsize=15)

        axs.tick_params(axis='both', which='major', labelsize=8)
        axs.view_init(elev=view_angle_elev, azim=view_angle_azim)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
        plt.gca().zaxis.set_major_locator(MaxNLocator(nbins = 4))

        # Add a single colorbar for all subplots

        cbar = fig.colorbar(sc1, ax=axs, orientation='horizontal', fraction=0.045, pad= -0.1, shrink=0.7)
        
        if self.var == 'vel':
            cbar.set_label(' $MAD_v$  (m/s)', fontsize = 12)
        if self.var == 'vkt':
            cbar.set_label('$MAD_{vkt}$ (km)', fontsize = 12)

        if  (-90 < view_angle_azim < 0) or (90 < view_angle_azim < 180):
            plt.subplots_adjust(left=-0.1, right=1.05, top=1.2, bottom=0.1)
        else:
            plt.subplots_adjust(left = 0.05, right = 1.1, top = 1.2, bottom = 0.1)

        plt.savefig(f'rsm_{self.var_1}_{self.var}_{self.model}_{self.model_vars[(starting_index)%3]}_{self.model_vars[(starting_index + 1)%3]}.png')
        
        # plt.show()
        plt.close()

    def free(self):

        for starting_index in [0, 1, 2]:
            fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, dpi=200)
            colormap = 'rainbow'

            if starting_index == 0:
                x, y = self.x1, self.x2
            elif starting_index == 1:
                x, y = self.x2, self.x3
            else:
                x, y = self.x3, self.x1

            # First subplot (acc_exp vs safe_head)
            sc1 = axs.scatter(x, y, self.error, c=self.error, cmap=colormap, marker='o', s=1, alpha=1)
            if self.units[self.model_vars[(starting_index)]] != '':
                axs.set_xlabel(self.symbols[self.model_vars[starting_index]] +'(' + 
                               self.units[self.model_vars[starting_index]] + ')', fontsize = 15)
            else:
                axs.set_xlabel(self.symbols[self.model_vars[starting_index]], fontsize = 15)

            if self.units[self.model_vars[(starting_index + 1)%3]] != '':
                axs.set_ylabel(self.symbols[self.model_vars[(starting_index + 1)%3]] + '(' + 
                               self.units[self.model_vars[(starting_index + 1)%3]]+')', fontsize = 15)
            else:
                axs.set_ylabel(self.symbols[self.model_vars[(starting_index + 1)%3]], fontsize = 15)
            
            if self.var == 'vkt':
                axs.set_zlabel('dVKT (km)', fontsize = 15)
            if self.var == 'vel':
                axs.set_zlabel('dv (m/s)', fontsize=15)

            axs.tick_params(axis='both', which='major', labelsize=8)
            axs.view_init(elev=0, azim=0)
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
            plt.gca().zaxis.set_major_locator(MaxNLocator(nbins = 4))

            # Add a single colorbar for all subplots

            cbar = fig.colorbar(sc1, ax=axs, orientation='horizontal', fraction=0.045, pad= -0.1, shrink=0.7)
            
            if self.var == 'vel':
                cbar.set_label(' dv  (m/s)', fontsize = 12)
            if self.var == 'vkt':
                cbar.set_label('dVKT (km)', fontsize = 12)

            # Adjust layout
            plt.tight_layout()
            plt.show()
            plt.close()

# To manually save the angles to plot the surface from. Disable the block after the angles are set for the final plots

rsm = RSM_2('idm', 'vel', 'vel')
rsm.free()
raise SystemExit()
        
# Reading the file for angles of the response surface

with open('rsm_angles.yaml', 'r') as file:
    angles = yaml.safe_load(file)

for model in list(angles.keys()):
    for var1 in ['vkt', 'vel']:
        if model in ['newell', 'bando']:
            var1 = 'vkt'
        
        for var2 in ['vkt', 'vel']:
            argument = angles[model][var1][var2]
            if type(argument) != dict:
                print('This input is not a dictionary')
            for i in argument.keys():
                rsm = RSM_2(model, var1, var2)
                rsm.plotting(i, argument[i][0], argument[i][1])