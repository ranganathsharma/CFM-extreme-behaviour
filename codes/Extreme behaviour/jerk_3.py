"""
This file has the codes for calculating the way the followers behave with time. Since the metric of number of windows with more than one jerk inversion is very insignificant, 
we only calculate the number of instances the acceleration and jerk values are out of physical range.
"""

# ====================================
# Library imports
# ====================================

import os, sys, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================================
# Custom imports
# ====================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

model_param_dict = yaml.safe_load(open('parameters.yaml', 'r'))
model_names = model_param_dict['Model_names']


write_csv = False

if write_csv:
    output_file = 'jerk_acc_exceed.csv'
    data = {'model': [],
            'id': [],
            'acc': [],
            'jerk': []}
    pd.DataFrame(data).to_csv(output_file, mode = 'w', index = False)

def calculate(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> None:
    """
    Function to quantify the aggregated extreme behaviour of each vehicle thoughout the journey

    Parameters
    ----------
    t : np.ndarray
        time array
    x : np.ndarray
        position array - 2d - (number of vehicles, time stamp)
    v : np.ndarray
        velocity array - 2d - (number of vehicles, time stamp)
    """

    v=  v[:,:-10]
    dt = t[1] - t[0]

    a = np.diff(v)/dt
    a = a[1:]

    j = (v[:,:-2] + v[:,2:] - 2*v[:,1:-1])/dt**2
    j = j[1:]

    extreme_acc = np.sum(np.abs(a) > 3, axis = 1)
    extreme_jerk = np.sum(np.abs(j) > 3, axis = 1)

    print(extreme_acc.shape, a.shape)
    extreme_acc = extreme_acc/ a.shape[1]
    extreme_jerk = extreme_jerk/ j.shape[1]

    if write_csv:

        data = {'model': [], 'id': [], 'acc': [], 'jerk': []}
        data['id'] = range(a.shape[0])
        data['model'] = [model]* a.shape[0]
        data['acc'] = extreme_acc
        data['jerk'] = extreme_jerk
        pd.DataFrame(data).to_csv(output_file, mode = 'a', header = False, index = False)

# Call this block only to create the data of extreme acceleration and jerk

if write_csv:       
    for model in ['idm', 'newell', 'bando']:
        velocity_class = 2
        colors = ['b', 'r', 'g']

        x = np.loadtxt(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_position' + str(velocity_class) + '.csv', delimiter=',')
        v = np.loadtxt(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_velocity' + str(velocity_class) + '.csv', delimiter=',')
        t = np.arange(0, x.shape[1]*0.1, 0.1)
        calculate(t, x, v)


data = pd.read_csv('jerk_acc_exceed.csv')

plt.figure(dpi=600)
bar_width = 0.4  # Width of the bars
model = 'bando'

data_temp = data[(data['model'] == model)]

ids = data_temp['id'].to_numpy()
plt.bar(ids - bar_width / 2, data_temp['acc'].to_numpy(), width=bar_width, align="center", alpha=0.8, label='acc')
plt.bar(ids + bar_width / 2, data_temp['jerk'].to_numpy(), width=bar_width, align="center", alpha=0.8, label='jerk')


plt.ylabel('Probability of exceedance', fontsize=15)
plt.xlabel('Vehicle', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
plt.savefig(f'Excedence_{model}.png', bbox_inches = 'tight')
plt.close()