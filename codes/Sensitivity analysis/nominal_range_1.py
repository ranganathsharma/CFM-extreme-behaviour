"""
This file conducts the nominal range sensitivity analysis for all the models.
The range is given as an input and the number of samples can be varied to create the deviations in the output values.
"""

# ====================================
# Library imports
# ====================================

import os, sys, yaml, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import logging

# ====================================
# Custom imports
# ====================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')
with open(yaml_file_path, 'r') as file:
    model_param_dict = yaml.safe_load(file)

from main import simulations  # Custom library


# The outputs of the analysis is saved in a file. Both the velocity and vkt variations are recorded for each combination of parameter values.
file_name = 'sensitivity.csv'

edit = True # control the saved information in sensitivity.csv

if edit == False:
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'parameter', 'value', 'deviation (x)', 'deviation (v)'])
else:
    pass

class NSA():

    def __init__(self, model: str) -> None:
        """
        Class to conduct nominal range sensitivity analysis 

        Parameters
        ----------
        model : str
            name of the model
        """
        self.model_name = model
        self.param_dict = model_param_dict['Models'][model]
        self.param_ranges = model_param_dict['Parameter_range'][model]
        self.velocity_class = 2

        if self.model_name in ['gipps', 'newell_delay']:
            self.delay = True
        else:
            self.delay = False

        simulator = simulations(self.velocity_class, self.model_name, self.param_dict)
        self.t, self.x, self.v = simulator.evolve()

    def run(self):

        for param in self.param_dict.keys():
            self.each_param(param)
            
        pass

    def each_param(self, param_name: str, velocity_class: int = 2) -> None:
        """
        For each parameter, a range of values is assumed and the difference is calculated agianst the initial values. 
        The outputs are the parameter values, deviation in positions, deviation in velocity

        Parameters
        ----------
        param_name : str
            name of the parameter that is being varied
        velocity_class : int, optional
            sensitivity is tested while the traffic stream follows leader velocity profile, by default WLTC class 2

        Raises
        ------
        SystemExit
            Input error is raised when the parameter range of a particular parameter is not noted in the parameters.yaml

        OverflowError
            If the system crashes for a particular set of parameter values, the deviations are set as np.nan
        """

        temp_dict = self.param_dict.copy()

        if param_name in self.param_ranges.keys():
            # The range is defined
            print(f'Processing the parameter {param_name}')
            ind_param_range = np.linspace(self.param_ranges[param_name][0], self.param_ranges[param_name][1], 15)
            for param in ind_param_range:
                temp_dict[param_name] = param
                try:

                    simulator = simulations(velocity_class = velocity_class, 
                                            model_name = self.model_name, 
                                            params = temp_dict, 
                                            dt = 0.1,
                                            delay = self.delay)
                    
                    t, x, v = simulator.evolve()
                    error_diff_x = self.difference(x, self.x)
                    error_diff_v = self.difference(v, self.v)

                    with open(file_name, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.model_name, param_name, param, error_diff_x, error_diff_v])

                except OverflowError:
                    with open(file_name, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.model_name, param_name, param, np.nan, np.nan])

                except:
                    logging.error(f'There was a different problem other than the ones mentioned in main.py at {param_name} = {param}')
                    input()
                    with open(file_name, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.model_name, param_name, param, np.nan, np.nan])

                
        # Some parameters are not varied like the length of the vehicle and jam_dis_1 which is assumed to be 0 always
        elif param_name in ['len_veh', 'jam_dis_1']:
            pass

        else:
            raise logging.warning(f'The input parameter {param_name} is not valid')

    # Gives the root mean square error
    def difference(self, x: np.ndarray, y: np.ndarray) -> float:
        """A function to calculate RMSE

        Parameters
        ----------
        x : np.ndarray
            velocity or position array
        y : np.ndarray
            velocity or position array

        Returns
        -------
        float
            RMSE between the two input arrays of the same size

        Raises
        ------
        SystemExit
            Checking for the same size of the input
        """
        try:
            return np.sqrt(np.mean((y - x)**2))
        except:
            raise SystemExit('Size Error: The sizes of the baseline and deviation arrays are not equal')
        
# Conduct analysis for all the models

for model in ['newell', 'bando', 'idm']:

    sens_analysis = NSA(model)
    sens_analysis.run()