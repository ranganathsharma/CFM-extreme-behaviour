"""
This file implements the methodology for analyzing the sensitivity of models using the Response Surface Method (RSM).
The order of parameter importance may differ based on the output variable (e.g., velocity or VKT), leading to varying results.
"""

# ===========================
# Library Imports
# ===========================

import os, sys, yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ===========================
# Custom Imports
# ===========================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from main import simulations

# Number of samples for the monte-carlo simulation
num_samples = 500 

class RSM_1():

    def __init__(self, model, var):
        self.model = model
        self.var = var
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
        temp_dict = order_dict[self.model][self.var]

        # Sort the dictionary by value and get the top three keys
        top_three = sorted(temp_dict, key=temp_dict.get, reverse=True)[:3]
        self.allowed_params = list(top_three)

        yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')

        with open(yaml_file_path, 'r') as file:
            model_param_dict = yaml.safe_load(file)

        self.param_dict = model_param_dict['Models'][self.model]
        self.param_ranges = model_param_dict['Parameter_range'][self.model]
        self.velocity_class = 2
        simulator = simulations(self.velocity_class, self.model, self.param_dict)
        self.t, self.x, self.v = simulator.evolve()

        
        self.data = {'id': []}
        for param in self.allowed_params:
            self.__setattr__(f'{param}_range', 
                             np.random.uniform(self.param_ranges[param][0], 
                                               self.param_ranges[param][1], num_samples))
            self.data[param] = []
        self.data['err_x'] = []
        self.data['err_v'] = []

    def difference(self, x, y):

        # Check if the sizes are the same
        if y.shape == x.shape:
            pass
        else:
            raise SystemExit('Size Error: The sizes of the baseline and deviation arrays are not equal')
        
        return np.sqrt(np.mean((y - x)**2))

    def run(self):
        pd.DataFrame(self.data).to_csv(f'{self.model}_rsm_{self.var}.csv', mode = 'w', index = False)
        for run_id in range(num_samples):
            for param in self.allowed_params:
                self.param_dict[param] = getattr(self, f'{param}_range')[run_id]

            try:
                simulator = simulations(self.velocity_class, self.model, self.param_dict)
                t, x, v = simulator.evolve()
                error_diff_x = self.difference(x, self.x)
                error_diff_v = self.difference(v, self.v)
                data = {}
                data['id'] = [run_id]
                for allowed_param in self.allowed_params:
                    data[allowed_param] = self.param_dict[allowed_param]
                data['err_x'] = error_diff_x
                data['err_v'] = error_diff_v
                pd.DataFrame(data).to_csv(f'{self.model}_rsm_{self.var}.csv', mode = 'a', header= False, index = False)

            except OverflowError:
                print(self.param_dict, 'Input Error: The input parameter values leads to model crash')
                # raise SystemExit()    

    def regression(self, err):

        data = pd.read_csv(f'{self.model}_rsm_{self.var}.csv')

        for allowed_param in self.allowed_params:
            mid_point = (self.param_ranges[allowed_param][0] + self.param_ranges[allowed_param][1])/2
            param_range_value = - self.param_ranges[allowed_param][0] + self.param_ranges[allowed_param][1]
            data[allowed_param] = 2*(data[allowed_param] - mid_point)/param_range_value
        

        X = np.vstack([data[f'{allowed_param}'] for allowed_param in self.allowed_params]).T

        if err == 1:
            print('For vkt as the model output')
            print('')
            y = data['err_x'].to_numpy() # To fit the vkt response surface

        elif err == 2:
            print('For vel as the model output')
            y = data['err_v'].to_numpy() # To fit the vel reponse surface

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        coefficients = model.coef_
        intercept = model.intercept_

        print(f'For model {self.model}:')
        print('The variables are in the following order')
        
        for param in self.allowed_params:
            print(param)
        print('')
        print("Coefficients:", np.round(coefficients, 2))
        print("Intercept:", intercept)
        print('')

        # Calculate R-squared to check the fit quality
        r_squared = model.score(X_poly, y)
        print("R-squared:", r_squared)
        print('')

# for model in ['idm', 'bando', 'newell']:
for model in ['idm', 'bando', 'newell']:
    print(f'Analysing the model {model}')
    print('=================================================================================================')
    for vars in ['vkt', 'vel']:
        
        if model not in ['bando', 'newell']: # These two models have only three parameters that are varied. The parameters for RSM are the same
            print(f'Analysing the surface formed by the MOD {vars}')
            method = RSM_1(model, vars)
            # method.run() # If data doesn't exist
            method.regression(1) # To procure the coefficients of the fit for vkt response surface
            method.regression(2) # To procure the coefficients of the fit for vel response surface
            print('=================================================================================================')

        else:
            if vars == 'vkt': # Avoiding running the analysis twice
                print(f'Analysing the surface formed by the MOD {vars}')
                method = RSM_1(model, vars)
                # method.run()
                method.regression(1)
                method.regression(2)
                print('=================================================================================================')