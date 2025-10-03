"""
This file has the codes for simulating the system of each model.
"""

# ================================
# Library imports
# ================================

import os, sys, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# ================================
# Custom imports
# ================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from model_equations import Model_equations
from solvers import solvers
from inhouse_drive_cycle_2 import vel_lead_acc, vel_lead_dec

model_param_dict = yaml.safe_load(open('parameters.yaml', 'r'))
model_order = model_param_dict['model_order']

class simulations():

    def __init__(self,
                 velocity_class: int,
                 model_name :str,
                 params: dict,
                 solver: str = 'euler',
                 dt: float = 0.1, 
                 spacing: bool = True) -> None:
        """
        This class is used to simulate the system.

        Parameters
        ----------
        velocity_class : int
            The velocity class of the leader vehicle, 2 and 3 for WLTC class 2 and 3, 4 for acceleration and 5 for shock.
        model_name : str
            The name of the model
        params : dict
            The parameters of the model.
        solver : str, optional
            The solver to use for the simulation, 'euler' or 'RK'.
        dt : float, optional
            The time step for the simulation.
        spacing : bool, optional
            Switch whether to use manual spacing or not.
        """
        self.model = model_name
        self.velocity_class = velocity_class
        self.n = 2000
        self.dt = dt
        self.num_rep = 4 # Number of times the leader cycle is repeated to simulate the system for 2 hours.
        self.solver = solver
        self.params_dict = params
        self.equations = Model_equations(self.params_dict)
        self.unpack_params()
        self.initialise_velocity()
        self.init_spacing()
        self.spacing = spacing
        self.setup_arrays()
        
    def unpack_params(self) -> None:
        """
        This function unpacks the parameters of the model.
        """
        for key in self.params_dict.keys():
            setattr(self, key, self.params_dict[key])

    def initialise_velocity(self) -> None:
        """
        This function initialises the velocity of the leader vehicle.

        Raises
        ------
        SystemExit
            If the input class for velocity is not valid.
        """
        
        if self.velocity_class in [2, 3]:
            
            # Load in the velocity profile from WLTC class 2 data
            dataframe_dc = pd.read_csv(f'{parent_dir}/data/WLTP_class_{self.velocity_class}_DC.csv')
            time = dataframe_dc['Time in s'].to_numpy()
            velocity = dataframe_dc['Speed in kmph'].to_numpy()

            # Interpolate to increase the frequency to 0.1 second. The original data is of 1 second frequency. Other interpolators can be used.
            interpolation_function = interpolate.interp1d (time, velocity, kind = 'linear')
            time_array = np.arange(0, len(time)-1 + self.dt, self.dt)
            velocity_array = interpolation_function(time_array)

            # concatenate multiple times to increase the duration of the simulation
            self.velocity_leader =  np.tile(velocity_array/3.6 , self.num_rep) # The 3.6 factor is considered to convert velocity units from km/h to m/s


        elif self.velocity_class == 4:
            self.velocity_leader = vel_lead_acc(self.dt)[1]

        elif self.velocity_class == 5:
            self.velocity_leader = vel_lead_dec(self.dt)[1]

        else:
            raise SystemExit('The input class for velocity is not valid')

    def init_spacing(self) -> None:
        """
        This function calculates the initial spacing of the vehicles based on the velocity.
        """

        init_vel = self.velocity_leader[0]
        init_spacing_function = getattr(self.equations, 'init_spacing_' + str(self.model), None)
        if callable(init_spacing_function):
            self.init_spacing_value = init_spacing_function(init_vel)
        else:
            raise SystemExit('Function error: init_spacing function is not callable')
        
    def setup_arrays(self) -> None:
        """
        This function sets up the arrays for the simulation.
        """

        len_sim = self.velocity_leader.shape[0]
        self.tim_array = np.arange(0, self.velocity_leader.shape[0]*self.dt, self.dt)
        self.pos_array = np.zeros((self.n, len_sim))
        self.vel_array = np.zeros((self.n, len_sim))
        

        self.vel_array[0] = self.velocity_leader
        self.vel_array[:,0] = self.velocity_leader[0]

        if self.spacing:
            for i in range(self.n):
                self.pos_array[i,0] = -i*self.init_spacing_value

        else:
            for i in range(1, self.n):
                self.pos_array[i,0] = self.pos_array[i-1,0] - np.random.uniform(6, 100)

    def evolve(self) -> tuple:
        """
        This function evolves the system.

        Returns
        -------
        tuple
            The time array, the position array and the velocity array.

        Raises
        ------
        OverflowError
            If the velocity has blown up.
        OverflowError
            If the vehicles have crashed.
        """

        if model_order[self.model] == 'second':
            v_equation = getattr(self.equations, 'u_dot_' + self.model + '_open')
            x_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')
        else:
            v_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')
            x_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')

        solver_class = solvers(model_order[self.model],
                               'open',
                               10000)
        

        numerical_solver = getattr(solver_class, self.solver + '_open_' + model_order[self.model])

        for count, time in enumerate(self.tim_array[:-2]):

            x_new, v_new = numerical_solver(self.pos_array[:,count],
                                            self.vel_array[:,count],
                                            self.vel_array[:,count + 1],
                                            self.dt,
                                            v_equation,
                                            x_equation)
            
            if np.max(v_new) > 40:
                raise OverflowError('The velocity has blown up', np.max(v_new), count)

            if np.min(x_new[:-1] - x_new[1:]) < 0:
                raise OverflowError('The vehicles have crashed')
                
            
            self.pos_array[:, count + 1], self.vel_array[1:, count + 1] = x_new, np.maximum(v_new[1:], v_new[1:]*0)

        return self.tim_array, self.pos_array, self.vel_array