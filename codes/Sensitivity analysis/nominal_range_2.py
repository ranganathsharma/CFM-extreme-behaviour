"""
This file has the codes to calculate the errors and plot the diagrams for nominal range sensitivity analysis. In addition a deviation order list is created.
"""
# ====================================
# Library imports
# ====================================

import os, sys, yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ====================================
# Custom imports and setup
# ====================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# ====================================
# Load parameter symbols and data
# ====================================

file_name = 'sensitivity.csv'  # This file has the deviations from the assumed parameter values

df = pd.read_csv(file_name)

yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')
with open(yaml_file_path, 'r') as file:
    model_param_dict = yaml.safe_load(file)
    symbol = model_param_dict['Symbols']
 
main_dict = {}

for model in ['idm', 'newell', 'bando']:
    fig_vel, ax_vel = plt.subplots(figsize=(4, 4), dpi=600)
    fig_vkt, ax_vkt = plt.subplots(figsize=(4, 4), dpi=600)

    model_df = df[df['Model'] == model]
    parameters = np.unique(model_df['parameter'])

    temp_dict = {'vkt': {}, 'vel': {}}

    for count, params in enumerate(parameters):
        param_df = model_df[model_df['parameter'] == params]
        value_list = param_df['parameter'].to_numpy()

        # Velocity deviation
        deviation_list = param_df['deviation (v)'].to_numpy()
        symbol_list = [symbol[model][value_list[i]] for i in range(len(value_list))]
        mean_deviation = np.mean(deviation_list)
        std_deviation = np.std(deviation_list)

        temp_dict['vel'][params] = float(mean_deviation)

        # Plot velocity deviation
        ax_vel.scatter(symbol_list, deviation_list, color='r', marker='o', s=3, alpha=1)
        ax_vel.scatter(symbol_list[0], mean_deviation, color='b', s=6)
        ax_vel.errorbar(symbol_list[0], mean_deviation, yerr=np.sqrt(std_deviation),
                        color='b', elinewidth=2, capsize=4, capthick=1, alpha=0.5)

        # VKT deviation
        deviation_list = param_df['deviation (x)'].to_numpy() / 1000
        mean_deviation = np.mean(deviation_list)
        std_deviation = np.std(deviation_list)
        temp_dict['vkt'][params] = float(mean_deviation)

        # Plot VKT deviation
        ax_vkt.scatter(symbol_list, deviation_list, color='r', marker='o', s=3, alpha=1)
        ax_vkt.scatter(symbol_list[0], mean_deviation, color='b', s=6)
        ax_vkt.errorbar(symbol_list[0], mean_deviation, yerr=np.sqrt(std_deviation),
                        color='b', elinewidth=2, capsize=4, capthick=1, alpha=0.5)
    
    main_dict[model] = temp_dict

    # Apply consistent axis limits and settings
    ax_vel.set_ylim(-5, 20)
    ax_vkt.set_ylim(-5, 20)

    # Add labels and style
    ax_vel.set_ylabel('$MAD_v$ (m/s)', fontsize=18)
    ax_vel.tick_params(axis='both', which='major', labelsize=15)
    fig_vel.tight_layout()  # Adjust layout
    fig_vel.savefig(f'{model}_sensitivity_vel.png', bbox_inches='tight')

    ax_vkt.set_ylabel('$MAD_{vkt}$ (km)', fontsize=18)
    ax_vkt.tick_params(axis='both', which='major', labelsize=15)
    fig_vkt.tight_layout()  # Adjust layout
    fig_vkt.savefig(f'{model}_sensitivity_vkt.png', bbox_inches='tight')

    plt.close(fig_vel)
    plt.close(fig_vkt)

with open('deviation_order.yaml', 'w') as yaml_file:
    yaml.dump(main_dict, yaml_file, default_flow_style=False)