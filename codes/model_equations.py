# This file gives the output x_dot and u_dot for any given model. The output functions are different for closed and open system
import numpy as np
import warnings
from scipy.optimize import fsolve
warnings.simplefilter('error')
import logging


class Model_equations():

    def __init__(self, 
                 params: dict):
        self.params_dict = params
        self.unpack_params()
        
        try:
            self.max_vel = self.des_vel
        except AttributeError:
            pass
        
    def unpack_params(self):

        for key in self.params_dict.keys():
            setattr(self, key, self.params_dict[key])
        pass

    # Bando
    
    def x_dot_bando_closed(self, x, v):
        return v
    
    def x_dot_bando_open(self, x, v):
        return v
    
    def u_dot_bando_closed(self, x, v):

        vel_foll = np.concatenate((v[1:], [v[0]]))
        vel_lead = v
        pos_foll = np.concatenate((x[1:], [x[0]]))
        pos_lead = x

        acc = (self.max_vel*(np.tanh(((pos_lead - pos_foll)%self.len_road - self.len_veh)/self.ds - self.beta) + np.tanh(self.beta))/(1 + np.tanh(self.beta)) - vel_foll)/self.tau
        return acc

    def u_dot_bando_open(self, x, v):

        vel_foll = v[1:]
        pos_foll = x[1:]
        pos_lead = x[:-1]

        acc = (self.max_vel*(np.tanh((pos_lead - pos_foll - self.len_veh)/self.ds - self.beta) + np.tanh(self.beta))/(1 + np.tanh(self.beta)) - vel_foll)/self.tau
        acc = np.append(np.array(0), acc)
        return acc
    
    def init_spacing_bando(self, vel):
        return (np.arctanh(vel*(1 + np.tanh(self.beta))/self.max_vel - np.tanh(self.beta)) + self.beta)*self.ds + self.len_veh
    
    def init_vel_bando(self, dx):
        return self.max_vel*(np.tanh((dx - self.len_veh)/self.ds - self.beta) + np.tanh(self.beta))/(1 + np.tanh(self.beta))
    
    # Newell
   
    def init_spacing_newell(self, vel):
        return self.jam_dis - self.max_vel/self.sens*np.log(1-vel/self.max_vel) 

    def init_vel_newell(self, dx):
        return self.max_vel*(1 - np.exp(-self.sens/self.max_vel*(dx - self.jam_dis)))

    def x_dot_newell_open(self, x, v):
        xdot = self.max_vel*(1 - np.exp(-self.sens/self.max_vel*(x[:-1] - x[1:] - self.jam_dis)))
        return np.concatenate((v[:1], xdot))

    # Newell_delay

    def init_spacing_newell_delay(self, vel):
        return self.jam_dis - self.max_vel/self.sens*np.log(1-vel/self.max_vel) 

    def init_vel_newell_delay(self, dx):
        return self.max_vel*(1 - np.exp(-self.sens/self.max_vel*(dx - self.jam_dis)))

    def x_dot_newell_delay_open(self, x, v):
        xdot = self.max_vel*(1 - np.exp(-self.sens/self.max_vel*(x[:-1] - x[1:] - self.jam_dis)))
        xdot = np.maximum(np.zeros_like(xdot), xdot)
        return np.concatenate((v[:1], xdot))

    # IDM
  
    def eq_idm(self, v):
        return 1 - (v/self.des_vel)**self.acc_exp - ((self.jam_dis_0 + self.safe_head*v)/(self.dx - self.len_veh))**2

    def init_vel_idm(self, dx):
        self.dx = dx
        return fsolve(self.eq_idm, [dx/self.safe_head, dx/self.safe_head, dx/self.safe_head, dx/self.safe_head])[0]

    def init_spacing_idm(self, vel):
        return (self.jam_dis_0 + self.safe_head*vel)/np.sqrt(1 - (vel/self.des_vel)**self.acc_exp) + self.len_veh

    def u_dot_idm_open(self, x, v):
       
        ideal_spacing = self.jam_dis_0 + self.safe_head*v[1:] + v[1:]*(v[1:] - v[:-1])/2/np.sqrt(self.max_acc*self.des_dec)
        acceleration = self.max_acc*(1 - (v[1:]/self.des_vel)**self.acc_exp - (ideal_spacing/(x[:-1] - x[1:] - self.len_veh))**2)
        return np.append(np.array(0), acceleration)

    def x_dot_idm_open(self, x, v):
        return v

    def string_stability_analytical_idm(self, eq_gap, eq_vel):
        # The expressions for string stability is particular to the model. This method contains the expressions for the stability equation and gives the string stability status
        eq_gap -= self.len_veh

        f_s = 2* self.max_acc/eq_gap**3 *(self.jam_dis_0 + self.safe_head*eq_vel)**2
        f_v = - self.max_acc *(4/self.des_vel**4*eq_vel**3 + 2*self.safe_head*(self.jam_dis_0 + self.safe_head*eq_vel)/eq_gap**2)
        f_dv = np.sqrt(self.max_acc/self.des_dec)*eq_vel/eq_gap**2 *(self.jam_dis_0 + self.safe_head*eq_vel)

        # print(self.max_acc, self.safe_head, eq_gap, 0.5 - f_dv/f_v - f_s/f_v**2)

        if 0.5 - f_dv/f_v - f_s/f_v**2 > 0:
            return 1 # Stable
        else:
            return 0 # Unstable 

    # GFM  
  
    def x_dot_gfm_open(self, x, v):
        return v
    
    def u_dot_gfm_open(self, x, v):
        
        ideal_dis = self.jam_dis + self.safe_head*v[1:]
        gap = x[:-1] - x[1:] - self.len_veh
        ideal_vel = self.des_vel*(1 - np.exp(-(gap - ideal_dis)/self.ran_acc))
        acceleration = (ideal_vel - v[1:])/self.acc_tim - (v[1:] - v[:-1])/self.brak_tim*np.heaviside(v[1:] - v[:-1], 0)*np.exp(-(gap - ideal_dis)/self.ran_dec)

        return np.append(np.array(0), acceleration)
    
    def init_spacing_gfm(self, v):
        return self.ran_acc*np.log(self.des_vel - v) - self.jam_dis - self.safe_head*v + self.len_veh
    
    def eq_gfm(self, v):
        return self.ran_acc*np.log(self.des_vel - v) - self.jam_dis - self.safe_head*v + self.len_veh - self.dx
    
    def init_vel_gfm(self, dx):
        self.dx = dx
        return fsolve(self.eq_gfm, [dx/self.safe_head, dx/self.safe_head, dx/self.safe_head, dx/self.safe_head])[0]
    
    # FVDM

    def x_dot_fvdm_open(self, x, v):
        return v

    def u_dot_fvdm_open(self, x, v):

        optimal_velocity = self.V1 + self.V2*(np.tanh(self.C1*(x[:-1] - x[1:] - self.len_veh) - self.C2))
        acceleration = self.sens_vel*(optimal_velocity - v[1:]) + self.sens_relvel*(v[1:] - v[:-1])
        return np.append(np.array(0), acceleration)
    
    def init_vel_fvdm(self, dx):
        return self.V1 + self.V2*np.tanh(self.C1*(dx - self.len_veh) - self.C2)
    
    def init_spacing_fvdm(self, v):
        return self.len_veh + (self.C2 + np.arctanh((v - self.V1)/self.V2))/self.C1
    
    # Shamoto

    def x_dot_shamoto_open(self, x, v):
        return v
    
    def u_dot_shamoto_open(self, x, v):
        acceleration = self.max_acc - self.str_int*v[1:]/(x[:-1] - x[1:] - self.jam_dis)**2*np.exp(-self.weig_rel*(v[:-1]-v[1:])) - self.drag*v[1:]
        return np.append(np.array(0), acceleration)
    
    def init_vel_shamoto(self, dx):
        return self.max_acc/(self.drag + self.self.str_int/(dx - self.jam_dis)**2)

    def init_spacing_shamoto(self, v):
        return self.jam_dis + np.sqrt(self.str_int*v/(self.max_acc - self.drag*v))

    # tsh

    def x_dot_tsh_open(self, x, v):
        return v

    def Zee(self, x):
        return (x + np.abs(x))*0.5
    
    def u_dot_tsh_open(self, x, v):
        ideal_spacing = self.jam_dis + self.safe_head*v[1:]
        acceleration = self.max_acc*(1 - ideal_spacing/(x[:-1] - x[1:])) - (self.Zee(v[1:] - v[:-1]))**2/(2*(x[:-1] - x[1:] - self.jam_dis)) - self.kappa*self.Zee(v[1:] - self.v_per)
        return np.append(np.array(0), acceleration)
    
    def eq_tsh(self, v):
        return self.max_acc*(1 - (self.jam_dis + self.safe_head*v)/self.dx) - self.kappa*self.Zee(v - self.v_per)
    
    def init_vel_tsh(self, dx):
        self.dx = dx
        return fsolve(self.eq_tsh, [dx/self.safe_head, dx/self.safe_head])[0]

    def init_spacing_tsh(self, v):
        ideal_spacing = self.jam_dis + self.safe_head*v
        return ideal_spacing/(1 - self.kappa/self.max_acc*self.Zee(v - self.v_per))

    # gipps

    def x_dot_gipps_open(self, x, v):
        v_free = v[1:] + 2.5*self.max_acc*self.delay*(1 - v[1:]/self.max_vel)*np.sqrt(0.025 + v[1:]/self.max_vel)
        try:
            v_cong = -self.max_dec*self.delay + np.sqrt(self.max_dec**2*self.delay**2 + self.max_dec*(2*(x[:-1] - x[1:] - self.jam_dis) - self.delay*v[1:] + v[:-1]**2/self.lead_dec))
        except:
            v_cong = np.zeros_like(v_free)
            root_min = np.min(self.max_dec**2*self.delay**2 + self.max_dec*(2*(x[:-1] - x[1:] - self.jam_dis) - self.delay*v[1:] + v[:-1]**2/self.lead_dec))
            logging.info(f'The congested velocity could not be calculated. The error was raised for the vehicle {np.where(self.max_dec**2*self.delay**2 + self.max_dec*(2*(x[:-1] - x[1:] - self.jam_dis) - self.delay*v[1:] + v[:-1]**2/self.lead_dec) == root_min)}')
        
        v_next = np.maximum(0, np.minimum(v_free, v_cong))
        return np.concatenate((v[:1], v_next))
    
    def init_spacing_gipps(self, v):
        if v > self.max_vel:
            v = self.max_vel
            (v**2 + 3*v*self.max_dec*self.delay - self.max_dec*v**2/self.lead_dec)/(2*self.max_dec) + self.jam_dis 
        else:
            return (v**2 + 3*v*self.max_dec*self.delay - self.max_dec*v**2/self.lead_dec)/(2*self.max_dec) + self.jam_dis 
    
    def init_vel_gipps(self, dx):
        if self.max_dec == self.lead_dec:
            v = 2*(dx - self.jam_dis)/(3*self.delay)
        else:
            v = (-3*self.max_dec*self.delay + np.sqrt( (3*self.max_dec*self.delay)**2 + 4*(1 - self.max_dec/self.lead_dec)*2*(dx - self.jam_dis)*self.max_dec))/(2*(1 - self.max_dec/self.lead_dec))

        return np.min(self.max_vel, v)
    