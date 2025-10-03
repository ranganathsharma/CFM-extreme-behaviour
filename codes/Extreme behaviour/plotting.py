import os, sys
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

model = 'newell'
velocity_class = 2

v = np.loadtxt(f'{parent_dir}\\Nonlin\Data\{model}_original_velocity_class_velocity' + str(velocity_class) + '.csv', delimiter=',')

plt.figure(dpi = 600)
t = np.arange(0, v.shape[1]*0.1, 0.1)/60
plt.plot(t, v[0])
plt.xlim(0, 30)
plt.xlabel('time (min)', fontsize = 15)
plt.ylabel('velocity (m/s)', fontsize = 15)
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.savefig('WLTC class 2', bbox_inches = 'tight')
plt.close()

