import numpy as np
from numpy import pi
from utilities import find_magnetization

def hysteresis_loop(h_min, h_max, num_h_points, energy_func, derivative, second_derivative, min_x=-pi, max_x=pi, x_steps=50, args=()):
    """Calculate the hysteresis loop for the Stoner Wohlfarth model

    Params:
      h_min (float): Minimum field
      h_max (float): Maximum field
      num_h_points (int): Number of field points
      energy_func (function):
      derivative (function):
      second_derivative (function):
      min_x (float): Minimum value in which to search for roots
      max_x (float): Maximum value in which to search for roots
      x_steps (int): Number of trials to do between [min_x, max_x] for finding roots
      args (tuple): Extra arguments that will be passed to the energy function and its derivatives

    Returns:
      np.ndarray: The hysteresis loop as (h, m) pairs
    """
    h_values = np.linspace(h_max, h_min, num_h_points)

    mh_curve = []
    mag = -1

    for h in h_values:
        _, mag = find_magnetization(energy_func, derivative, second_derivative, mag, min_x, max_x, x_steps, args=((h,) + args))
        mh_curve.append((h, mag))

    return np.array(mh_curve)
