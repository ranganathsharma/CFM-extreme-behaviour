"""
This file has the codes for running the simulation and saving them for further extreme behaviour analysis. 
"""

# ================================
# Library imports
# ================================

import os, sys, yaml, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ================================
# Custom imports
# ================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')
with open(yaml_file_path, 'r') as file:
    model_param_dict = yaml.safe_load(file)

from main import simulations


class validation():

    def __init__(self, model: str) -> None:
        """
        This class is used to run the simulation and save the results.

        Parameters
        ----------
        model : str
            The name of the model.
        """

        self.model_name = model
        self.param_dict = model_param_dict['Models'][model]

    def run(self, velocity_class):
        """
        This function runs the simulation.

        Parameters
        ----------
        velocity_class : int
            The velocity class of the leader vehicle.
        """

        self.velocity_class = velocity_class

        if velocity_class == 4:
            name = 'acceleration'
            self.name = name
        elif velocity_class == 5:
            name = 'shock'
            self.name = name
        else:
            self.name = None
            pass

        param_dict = model_param_dict['Models'][self.model_name]
        
        # Acceleration wave
        simulator = simulations(velocity_class, self.model_name, param_dict, dt = 0.1)
        self.t, self.x, self.v = simulator.evolve()

    def save_data(self)-> None:

        """
        This function saves the data.
        """
        chunk_size = 20

        with open(f'Data\{self.model_name}_original_velocity_class_position' + str(self.velocity_class) + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.x[0])  
            for i in range(1, len(self.x), chunk_size):
                chunk = self.x[i:i+chunk_size]  # Get the current chunk
                writer.writerows(chunk)

        with open(f'Data\{self.model_name}_original_velocity_class_velocity' + str(self.velocity_class) + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.v[0]) 

            for i in range(1, len(self.v), chunk_size):
                chunk = self.v[i:i+chunk_size]  # Get the current chunk
                writer.writerows(chunk)
        
    def velocity(self)-> None:

        """
        This function plots the velocity.
        """

        t = self.t/60
        x = self.x/1000
        v = self.v

        plt.figure(dpi = 300)
        for i in range(50):
            plt.plot(t, v[i], linewidth = 0.6, alpha = 0.6)

        plt.xlabel('time (min)', fontsize = 15)
        plt.ylabel('velocity (m/s)', fontsize = 15)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # plt.tight_layout()
        # plt.ylim(0, 15)
        # plt.savefig(f'{self.velocity_class}_wave.png', bbox_inches = 'tight')
        plt.show()
        plt.close()
   
    def trajectory(self)-> None:
        
        """
        This function plots the trajectory.
        """

        t = self.t/60
        x = self.x/1000
        v = self.v

        plt.figure(dpi=600)

        # Iterate over each trajectory
        for i in range(0, len(x), 5):
            points = np.array([t, x[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a LineCollection
            lc = LineCollection(segments, cmap= custom_cmap, linewidth = 1, norm=plt.Normalize(2, 18))
            lc.set_array(v[i])
            plt.gca().add_collection(lc)

        # Colorbar
        if self.model_name == 'idm':
            cbar = plt.colorbar(lc)
            cbar.set_label('velocity (m/s)', rotation=270, labelpad=20, fontsize=20)
            cbar.ax.tick_params(labelsize = 18)

        # Axis labels
        plt.xlabel('time (min)', fontsize=20)
        plt.ylabel('position (km)', fontsize=20)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
        plt.xlim(np.min(t), np.max(t))
        plt.ylim(np.min(x), np.max(x))

        plt.xlim(9, 30)
        plt.ylim(0.5, 5.2)

        plt.tight_layout()

        if self.velocity_class in [4, 5]:
            rect = plt.Rectangle((9.5, 4.5), 1, 0.45, linewidth = 0.6, edgecolor='black', facecolor='none')
            plt.gca().add_patch(rect)
            plt.savefig(f'{model}_trajectory_{self.name}_wave.png', bbox_inches='tight')
            # raise SystemExit()
        if self.velocity_class == 2:
            plt.savefig(f'{model}_trajectory_{self.velocity_class}.png', bbox_inches = 'tight')
        # plt.show()
        plt.close()
    
    def fundamental_diag(self)-> None:
        
        """
        This function plots the fundamental diagram.
        """

        # https://doi.org/10.1109/FISTS60717.2024.10485593
        
        t = self.t
        x = self.x
        v = self.v

        dx, dt = 100, 60
        start_time, end_time = t[0], t[-1]
        start_pos, end_pos = np.min(x), np.max(x)
        num_veh = x.shape[0]

        space_grid = np.arange(start_pos, end_pos, dx)
        time_grid = np.arange(start_time, end_time, dt)

        num_space_cells = len(space_grid) - 1
        num_time_cells = len(time_grid) - 1

        flow_grid = np.zeros((num_space_cells, num_time_cells))
        density_grid = np.zeros((num_space_cells, num_time_cells))
        speed_grid = np.zeros((num_space_cells, num_time_cells))

        for i in range(num_space_cells):
            for j in range(num_time_cells):
       

                space_min, space_max = space_grid[i], space_grid[i + 1]
                time_min, time_max = time_grid[j], time_grid[j + 1]

                space_mask = (x >= space_min) & (x < space_max)
                time_mask = (t >= time_min) & (t < time_max)

                combined_mask = space_mask & time_mask[:,np.newaxis].T

                if not np.any(combined_mask): 
                    continue # rule out all the empty grid cells

                vehicles_in_cell = np.any(combined_mask, axis = 1)

                time_in_cell = np.sum(combined_mask[vehicles_in_cell], axis = 1)* (t[1] - t[0])
                total_time = np.sum(time_in_cell)

                distance_in_cell = x*combined_mask
                vehicle_indices = vehicles_in_cell*range(1, num_veh + 1)

                vehicle_indices = vehicle_indices[vehicle_indices > 0]-1
                
                distance_in_cell = np.where(distance_in_cell == 0, np.nan, distance_in_cell)
                

                distance_start = np.array([np.nanmin(distance_in_cell[k]) for k in vehicle_indices])
                distance_end = np.array([np.nanmax(distance_in_cell[k]) for k in vehicle_indices])
                
                np.set_printoptions(threshold=np.inf)
                total_distance = np.sum(distance_end - distance_start)
                

                if total_time > 0:
                
                    flow_grid[i, j] = total_distance/dx/dt*3600
                    density_grid[i,j] = total_time/dx/dt*1000
                    speed_grid[i, j] = (total_distance / total_time) * 3.6  # Convert m/s to km/h
                    if flow_grid[i,j] > 4000:
                        print('Something went wrong')
                        raise SystemExit
        
        
        # plt.figure(dpi = 400)
        # plt.scatter(density_grid, flow_grid, color = 'r', s = 0.1, marker = '.')
        # plt.xlabel('density (Num/km)', fontsize = 12)
        # plt.ylabel('Flow (Num/hr)', fontsize = 12)
        # plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        # plt.tight_layout()
        # plt.savefig(f'{model}_trajectory_fundamental_diagram_' + str(self.velocity_class) + '.png', bbox_inches='tight')
        # # plt.show()
        # plt.close()

        plt.figure(dpi = 400)
        plt.imshow(density_grid[:,10:], cmap = 'coolwarm', aspect = 'auto')
        plt.colorbar()
        plt.xlabel('time (mins)', fontsize = 18)
        plt.ylabel('space (0.1 km)', fontsize = 18)
        plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
        plt.savefig(f'{self.model_name}_density_space_time.png', bbox_inches = 'tight')
        # plt.show()
        plt.close()

# ================================
# Colors
# ================================

colors = [(165/255, 42/255, 42/255),   # Brown
          (255/255, 69/255, 0/255),    # Red (OrangeRed)
          (255/255, 165/255, 0/255),   # Orange
          (255/255, 255/255, 0/255),   # Yellow
          (34/255, 139/255, 34/255),   # Green
          (64/255, 224/255, 208/255),  # Turquoise
          (135/255, 206/255, 235/255), # Skyblue
          (0/255, 0/255, 255/255)]     # Blue

custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

for model in ['idm', 'bando', 'newell']:

    Valid = validation(model)
    Valid.run(5)
    Valid.trajectory()
    
    Valid.run(4)
    Valid.trajectory()

    Valid.run(2)
    Valid.trajectory()

    # for velocity_class in [2]:
    #     Valid.run(velocity_class)
    #     Valid.trajectory()

        # Valid.save_data()
        # Valid.velocity()
        # Valid.fundamental_diag()
    