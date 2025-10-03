'''
This file has the codes for preliminary acceleration jerk analysis. 
'''

# ================================
# Library imports
# ================================

import os, sys, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ================================
# Custom imports
# ================================

model_param_dict = yaml.safe_load(open('parameters.yaml', 'r'))
model_names = model_param_dict['Model_names']

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Set to False since the data is too huge
write_csv = False 

if write_csv:
    output_file = 'acc_jerk.csv'
    data = {'model': [],
            'var': [],
            'value': []}

    pd.DataFrame(data).to_csv(output_file, mode = 'w', index = False)

def calculate(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> None:
    """
    This function calculates the acceleration and jerk values that are extreme and prints it in the terminal.

    Parameters
    ----------
    t : np.ndarray
        The time array.
    x : np.ndarray
        The position array.
    v : np.ndarray
        The velocity array.
    """
    dt = t[1] - t[0]

    v = v[:,:-10] # Removing last few values to ensure no sudden velocity drops

    print('The number of vehicles is ', len(v))

    print('The time step value is ', np.round(dt, 2))

    a = np.diff(v)/dt
    a = a[1:]

    j = (v[:,:-2] + v[:,2:] - 2*v[:,1:-1])/dt**2
    j = j[1:]
    t_j = t[1:-1]

    # Writing the csv datafile for acceleration and jerk values
     
    if write_csv:

        data = {'model': [], 'var': [], 'value': []}
        data['value'] = list(a.flatten())
        data['model'] = [model]* len(a.flatten())
        data['var'] = ['acc']* len(a.flatten())
        pd.DataFrame(data).to_csv(output_file, mode = 'a', header = False, index = False)

        data = {'model': [], 'var': [], 'value': []}
        data['value'] = list(j.flatten())
        data['model'] = [model]* len(j.flatten())
        data['var'] = ['jerk']* len(j.flatten())
        pd.DataFrame(data).to_csv(output_file, mode = 'a', header = False, index = False)

    #===========================================================================================================================================
    # Analysis for statistics of jerk and acceleration
    #===========================================================================================================================================
        
    extreme_acc = np.sum(np.abs(a) > 3)
    print(f'For {model}, the percentage of values with acceleration more than 3 is {extreme_acc / (a.shape[0] * a.shape[1]):.3f}')

    extreme_jerk = np.sum(np.abs(j) > 3)
    print(f'For {model}, the percentage of values with jerk more than 3 is {extreme_jerk / (j.shape[0] * j.shape[1]):.3f}')
    
    plt.figure(1, dpi = 600)
    counts, bins = np.histogram(a.flatten(), bins = np.arange(-3, 3+ 0.1, 0.25))

    total_counts = sum(counts)
    counts_percentage = (counts / total_counts) * 100
    plt.bar(bins[:-1], counts_percentage, width=np.diff(bins), align="edge", alpha = 0.5, label = model_names[model], edgecolor = 'black')


    plt.figure(2, dpi = 600)
    counts, bins = np.histogram(j.flatten(), bins = np.arange(-3, 3 + 0.1, 0.25))
    total_counts = sum(counts)
    counts_percentage = (counts / total_counts) * 100
    plt.bar(bins[:-1], counts_percentage, width=np.diff(bins), align="edge", alpha = 0.5, label = model_names[model], edgecolor = 'black')


    j_1 = j[:,1:]
    j_2 = j[:,:-1]

    sign_changes = (j_1*j_2) < -1e-9

    #===========================================================================================================================================
    # Analysis for windows for all vehicles
    #===========================================================================================================================================

    column_sums = np.sum(sign_changes, axis = 0)
    window_size = int(1/dt)
    many_inv = np.array([np.sum(column_sums[i*window_size: (i + 1)*window_size]) for i in range(len(column_sums)//window_size)])

    fraction = np.sum(many_inv > 1)
    total = len(many_inv)
    plt.figure(3, dpi = 600)
    plt.plot(many_inv, label = model_names[model], linewidth = 0.6)
    print(f'The proportion of 1 second windows with more than one jerk inversion is {fraction}, {total}, {fraction/total:.2f}')

    #===========================================================================================================================================
    # Analysis for windows for each vehicle
    #===========================================================================================================================================

    many_inv_ind = np.array([np.sum(sign_changes[:,i*window_size: (i + 1)*window_size], axis = 1) for i in range(len(column_sums)//window_size + 1)])

    fraction = np.sum(many_inv_ind > 1)
    total = len(many_inv_ind)*len(many_inv_ind[0])

    print(f'The proportion of 1 second windows for individual vehicle with more than one jerk inversion is {fraction}, {total}, {fraction/total:.2f}')

for model in ['idm', 'bando', 'newell']:
    velocity_class = 2
    colors = ['b', 'r', 'g']
    x = np.loadtxt(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_position' + str(velocity_class) + '.csv', delimiter=',')
    v = np.loadtxt(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_velocity' + str(velocity_class) + '.csv', delimiter=',')
    t = np.arange(0, x.shape[1], 0.1)

    calculate(t, x, v)

fig1 = plt.figure(1)
plt.xlabel('acc $m/s^2$', fontsize = 15)
plt.ylabel('Frequency (%)', fontsize = 15)
plt.ylim(0, 100)
plt.legend()
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.savefig('acc_histogram.png', bbox_inches = 'tight')
plt.close()

fig2 = plt.figure(2)
plt.xlabel('jerk $(m/s^3)$', fontsize = 15)
plt.ylabel('Frequency (%)', fontsize = 15)
plt.ylim(0, 100)
plt.legend()
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.savefig('jerk_histogram.png', bbox_inches = 'tight')
plt.close()

fig3 = plt.figure(3)
plt.xlabel('time (s)', fontsize = 15)
plt.ylabel('Num jerk inv', fontsize = 15)
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.legend()
plt.savefig('jerk_inv_time_series.png', bbox_inches = 'tight')
plt.close()