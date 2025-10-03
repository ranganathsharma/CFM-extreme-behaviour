import numpy as np
from typing import Callable

class solvers():

    def __init__(self, 
                 closed_road: bool = 'open',
                 road_len: float = 1000) -> None:
        """
        Class of solver methods

        Euler, Runge Kutta and others. 

        {solver}_{road}_{order} is the format of the methods

        solver: euler / RK (runge kutta 4th order)
        road: closed/open 
        order: first/second

        Parameters
        ----------
        closed_road : bool
            ring road (True) open road (False)
        road_len : float, optional
            Length of the road for ring road condition, by default 0

        Summary of each solver method:

        Parameters
        ----------
        x : np.ndarray
            position array of all vehicles at time 0
        v : np.ndarray
            velocity array of all vehicles at time 0
        v_next : np.ndarray
            velocity of the next time step (applicable only when leader velocity is known)
        dt : float
            simulation time step
        u_dot : function
            velocity rate equation
        x_dot : _type_
            position rate equation

        Returns
        -------
        tuple of np.ndarray
            updated position x_new and velocity v_new
        """

        self.road_len = road_len
        self.closed = closed_road

    def euler_closed_second(self, x, v, v_next, dt, u_dot, x_dot):

        dx = u_dot(x, v)*dt
        dv = x_dot(x, v)*dt + 0.5*u_dot(x, v)*dt**2

        x_new = (x + dx)%self.road_len
        v_new = np.maximum(v + dv, v*0)

        return x_new, v_new
    
    def euler_open_second(self, x, v, v_next, dt, u_dot, x_dot):

        dv = u_dot(x, v)*dt
        dx = x_dot(x, v)*dt + 0.5*u_dot(x, v)*dt**2

        new_dx = np.where(dx < 0, 0, dx)
        x_new = x + new_dx
        
        v_new = np.maximum(v + dv, v*0)
        return x_new, v_new
    
    def euler_closed_first(self, x, v, v_next, dt, u_dot, x_dot):

        x_new = x + x_dot(x, v)*dt
        v_new = (x_new - x)/dt
        x_new = x_new%self.road_len

        return x_new, v_new
    
    def euler_open_first(self, x, v, v_next, dt, u_dot, x_dot):

        x_new = x + x_dot(x, v)*dt
        v_new = (x_new - x)/dt

        return x_new, v_new
    
    def euler_delay_open_first(self, x, v, v_next, dt, u_dot, x_dot):
        v_new = x_dot(x, v)
        return v_new
    
    def RK_closed_second(self, x, v, v_next, dt, u_dot, x_dot):

        x_temp = x.copy()
        v_temp = v.copy()

        k1 = dt*x_dot(x, v)
        l1 = dt*u_dot(x, v)

        k2 = dt/2*x_dot((x + k1)%self.road_len, v + l1)
        l2 = dt/2*u_dot((x + k1)%self.road_len, v + l1)

        k3 = dt/2*x_dot((x + k2)%self.road_len, v + l2)
        l3 = dt/2*u_dot((x + k2)%self.road_len, v + l2)

        k4 = dt*x_dot((x + k3)%self.road_len, v + l3)
        l4 = dt*u_dot((x + k3)%self.road_len, v + l3)

        dx = 1/6*(k1 + 2*(k2 + k3) + k4)
        dv = 1/6*(l1 + 2*(l2 + l3) + l4)

        x_new = (x + dx)%self.road_len
        v_new = v + dv

        return x_new, v_new
    
    def RK_open_second(self, x, v, v_next, dt, u_dot, x_dot):

        x_temp = x.copy()
        v_temp = v.copy()

        k1 = dt*x_dot(x, v)
        l1 = dt*u_dot(x, v)

        x_temp[0] += v[0]*dt/2
        x_temp[1:] += k1[1:]
        v_temp[0] = (v_next[0] + v[0])/2 # step to include the leader velocity as an input
        v_temp[1:] += l1[1:]

        k2 = dt/2*x_dot(x_temp, v_temp)
        l2 = dt/2*u_dot(x_temp, v_temp)

        x_temp[1:] = x[1:] + k2[1:]
        v_temp[1:] = v[1:] + l2[1:]

        k3 = dt/2*x_dot(x_temp, v_temp)
        l3 = dt/2*u_dot(x_temp, v_temp)

        x_temp[1:] = x[1:] + k3[1:]
        v_temp[1:] = v[1:] + l3[1:]
        v_temp[0] = v_next[0]

        k4 = dt*x_dot(x_temp, v_temp)
        l4 = dt*u_dot(x_temp, v_temp)

        dx = 1/6*(k1 + 2*(k2 + k3) + k4)
        dv = 1/6*(l1 + 2*(l2 + l3) + l4)

        x_new = x + dx
        v_new = v + dv

        return x_new, v_new
    
    def RK_closed_first(self, x, v, v_next, dt, u_dot, x_dot):

        k1 = x_dot(x, v)*dt/2
        k2 = x_dot((x + k1)%self.road_len, v)*dt/2
        k3 = x_dot((x + k2)%self.road_len, v)*dt
        k4 = x_dot((x + k3)%self.road_len, v)*dt

        dx = 1/6*(k1 + 2*(k2 + k3) + k4)
        dv = dx/dt
        x_new = (x + dx)%self.road_len
        v_new = v + dv

        return x_new, v_new
    
    def RK_open_first(self, x, v, v_next, dt, u_dot, x_dot):

        k1 = dt*x_dot(x, v)
        x_temp = x.copy()
        x_temp[0] = x_temp[0] + v[0]*self.dt/2
        x_temp[1:] = x[1:] + k1/2 

        k2 = dt*x_dot(x_temp)
        x_temp[1:] = x[1:] + k2/2

        k3 = dt*x_dot(x_temp)
        x_temp[1:] = x[1:] + k3
        x_temp[0] = x_temp[0] + v[0]*dt/2

        k4 = dt*x_dot(x_temp)

        dx = 1/6*(k1 + 2*k2 + 2*k3 + k4)
        dx = np.append(np.array(v[0]*self.dt), dx)

        dv = dx/dt
        x_new = x + dx
        v_new = v + dv

        return x_new, v_new
    
    def euler_second_TK(self, 
                 x: np.ndarray, 
                 v: np.ndarray, 
                 a_prev: np.ndarray, 
                 dt: float, 
                 u_dot: Callable, 
                 x_dot: Callable) -> tuple[np.ndarray]:
        """
        https://www.sciencedirect.com/science/article/pii/S0191261517308536?via%3Dihub
        Numerical method with first order approximation for a second order ode

        Parameters
        ----------
        x : np.ndarray
            position 1d array
        v : np.ndarray
            velocity 1d array
        a_prev : np.ndarray
            acceleration 1d array
        dt : float
            simulation time step
        u_dot : function
            method that dictates the acceleration
        x_dot : function
            method that dictates the velocity

        Returns
        -------
        tuple[np.ndarray]
            updated position x_new, velocities v_new and calculated acceleration a_curr
        """

        a_cur = u_dot(x, v)
        v_new = v + 0.5*(a_prev + a_cur)*dt
        x_new = x + v*dt + 0.5*a_prev*dt**2

        return x_new, v_new, a_cur
    
