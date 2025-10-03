"""
This file has the codes for creating the velocity profiles for the leader vehicle for the acceleration and deceleration scenarios.
"""

# ================================
# Library imports
# ================================

import numpy as np

def vel_increase(vel_initial: float,
                 vel_final: float,
                 dt: float = 0.1) -> np.ndarray:
    """
    This function creates an increasing velocity subprofile.
    Parameters
    ----------
    vel_initial : float
        The initial velocity of the leader vehicle.
    vel_final : float
        The final velocity of the leader vehicle.
    dt : float, optional
        The time step for the simulation.

    Returns
    -------
    np.ndarray
        The velocity array.
    """
    if vel_initial < vel_final:
        acc_max = 2
    else:
        acc_max = -3

    time_dur = np.pi*(vel_final - vel_initial)/(2*acc_max)
    time = np.arange(0, time_dur, dt)
    velocity = vel_initial + (vel_final-vel_initial)*0.5*(1 - np.cos(np.pi * time/time_dur))

    return velocity

def vel_constant(vel_mean: float,
                 time_duration: float,
                 dt: float = 0.1) -> np.ndarray:
    """
    This function creates a constant velocity subprofile.

    Parameters
    ----------
    vel_mean : float
        The mean velocity of the leader vehicle.
    time_duration : float
        The time duration of the subprofile.
    dt : float, optional
        The time step for the simulation.

    Returns
    -------
    np.ndarray
        The velocity array.
    """

    time_array = np.arange(0, time_duration, dt)
    velocity = vel_mean*np.ones((time_array.shape[0]))

    return velocity

def vel_lead_acc(dt: float = 0.1) -> tuple:

    """
    This method creates a velocity profile where the leader temporarily accelerates. 

    Returns
    -------
    tuple
        The time array and the velocity array.
    """
    # Acceleration set to 12 m/s

    segments = {
        1: {'method': vel_constant, 'inputs': (8, 600)},
        2: {'method': vel_increase, 'inputs': (8, 12)},
        3: {'method': vel_constant, 'inputs': (12, 5)},
        4: {'method': vel_increase, 'inputs': (12, 8)},
        5: {'method': vel_constant, 'inputs': (8, 600)}
    }

    main_velocity_array = [_segment['method'](*_segment['inputs']) for _segment in segments.values()]
    velocity_array = np.concatenate(main_velocity_array)
    time_max = len(velocity_array)*dt
    time_array = np.arange(0, time_max, dt)

    return [time_array, velocity_array]

def vel_lead_dec(dt: float = 0.1) -> tuple:
    """
    This method creates a velocity profile where the leader temporarily decelerates.

    Returns
    -------
    tuple
        The time array and the velocity array.
    """
    # Deceleration set to 1 m/s
    segments = {
        1: {'method': vel_constant, 'inputs': (8, 600)},
        2: {'method': vel_increase, 'inputs': (8, 1)},
        3: {'method': vel_constant, 'inputs': (1, 5)},
        4: {'method': vel_increase, 'inputs': (1, 8)},
        5: {'method': vel_constant, 'inputs': (8, 600)}
    }

    main_velocity_array = [_segment['method'](*_segment['inputs']) for _segment in segments.values()]
    velocity_array = np.concatenate(main_velocity_array)
    time_max = len(velocity_array)*dt
    time_array = np.arange(0, time_max, dt)

    return [time_array, velocity_array]