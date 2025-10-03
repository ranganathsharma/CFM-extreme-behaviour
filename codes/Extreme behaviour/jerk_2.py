'''
This file has the codes for plotting the acceleration changes in the trajectory plot mainly used for the OVM model termed as bando in this codebase. 
'''

# ===================================
# Library imports
# ===================================

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


# Loading the file

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

model = 'bando'

velocity_class = 2
colors = ['b', 'r', 'g']

# Load the data of the trajectories. 

num_vehs = 1100 # Analysis done considering these many vehicles to avoid non moving vehicles. 
x = pd.read_csv(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_position' + str(velocity_class) + '.csv', nrows = num_vehs)
v = pd.read_csv(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_velocity' + str(velocity_class) + '.csv', nrows = num_vehs)
t = np.arange(0, x.shape[1]*0.1, 0.1) # Assuming the time difference to be 0.1 seconds

x = x.to_numpy()
v = v.to_numpy()

print('The data has been imported')


colors = [(165/255, 42/255, 42/255),   # Brown
          (255/255, 69/255, 0/255),    # Red (OrangeRed)
          (255/255, 165/255, 0/255),   # Orange
          (255/255, 255/255, 0/255),   # Yellow
          (34/255, 139/255, 34/255),   # Green
          (64/255, 224/255, 208/255),  # Turquoise
          (135/255, 206/255, 235/255), # Skyblue
          (0/255, 0/255, 255/255)]     # Blue

custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

def trajectory(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> None:
    """
    Function to plot the trajectories with color indicating the acceleration

    Parameters
    ----------
    t : np.ndarray
        time array
    x : np.ndarray
        position array - 2d (number of vehicles, time stamps)
    v : np.ndarray
        velocity array - 2d (number of vehicles, time stamps)
    """

    t = t[1:]/60
    x = x[:,1:]/1000

    a = np.diff(v, axis = 1)/0.1

    plt.figure(dpi=600)

    # Iterate over each trajectory
    for i in range(0, len(x), 2):
        linewidth = 0.6
        alpha = 0.8
        # if i in [0, 100, 500]:
        #     linewidth = 2
        #     alpha = 0.2
        # else:
        #     pass

        points = np.array([t, x[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        
        lc = LineCollection(segments, cmap = custom_cmap, alpha = alpha, linewidth = linewidth, norm = plt.Normalize(-6, 6))
        # lc = LineCollection(segments, cmap='coolwarm', alpha = alpha, linewidth = linewidth, norm=plt.Normalize(-6, 6))

        if i in [0, 100, 500]:

            colors = [(0, 0, 0), (0, 0, 0)]
            new_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
            lc = LineCollection(segments, cmap = new_cmap, alpha = alpha, linewidth = linewidth, norm = plt.Normalize(-6, 6))

        lc.set_array(a[i])
        plt.gca().add_collection(lc)

    # Colorbar
    cbar = plt.colorbar(lc)
    cbar.set_label('acceleration ($m/s^2$)', rotation=270, labelpad=20, fontsize=15)

    # Axis labels
    plt.xlabel('time (min)', fontsize=15)
    plt.ylabel('position (km)', fontsize=15)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
    plt.xlim(75, 95)
    plt.ylim(51, 62)

    plt.tight_layout()

    plt.savefig(f'{model}_acceleration_trajectory.png', bbox_inches='tight')
    # plt.show()
    plt.close()

if __name__ == '__main__':
    trajectory(t, x, v)

