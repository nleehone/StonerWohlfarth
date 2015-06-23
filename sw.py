import scipy as sp
import numpy as np
import scipy.optimize
from numpy import sin, cos, abs, pi
import matplotlib.pyplot as plt


def find_roots(func, min_x, max_x, x_steps, args=()):
    """Find the roots of a function.

    The roots are determined by using sp.optimize.newton on various starting points between min_x and max_x.

    Params:
      func (function): The function that will be finding the roots of. The function must take the form func(x, arg1, arg2, ...) where x is the variable that we want to find roots for.
      min_x (float): The minimum value that x can take
      max_x (float): The maximum value that x can take
      x_steps (int): The number of slices to create between min_x and max_x. We will try to find a root for each slice, so more slices takes longer to evaluate.
      args (tuple): The arguments that will be passed to the function

    Returns:
      roots (list): A list of all the roots found by this function. Will be at most x_steps long. If no roots are found, the list will be empty.
    """
    x_trials = np.linspace(min_x, max_x, x_steps)
    possible_solutions = []

    # Try to find the roots of the first_derivative
    for x in x_trials:
        try:
            possible_solution = sp.optimize.newton(func, x, args=args)
        except RuntimeError:
            # The newton root finder could not find a root
            continue
        else:
            possible_solutions.append(possible_solution)

    return possible_solutions


def check_root_is_minimum(root, deriv, second_deriv, resolution=0.00001, args=()):
    """Check if the root of the derivative is a minimum of the function

    Params:
      root (float): Possible minimum that we want to check.
      deriv (function): The derivative of the function that we want to find a minimum of. The function should take the form func(x, arg1, arg2, ...) where x is the variable that we will be putting 'root' into.
      second_deriv (function): The second derivative of the function that we want to find a minimum of. The function should take the form func(x, arg1, arg2, ...) where x is the variable that we will be putting 'root' into.
    resolution (float): How close the second derivative has to be from zero to be considered a valid solution
    args (tuple): The arguments that will be passed to the derivative and second derivative.

    Returns:
      bool: Is this a valid minimum or not?
    """
    if abs(deriv(root, *args)) < resolution and second_deriv(root, *args) > 0:
        return True

    return False


def find_magnetization(func, deriv, second_deriv, previous_magnetization, min_x, max_x, x_steps, resolution=0.0001, args=()):
    """Determine the magnetization from the energy function and its first and second derivatives.

    Params:
      func (function): The energy function. The function should take the form func(x, arg1, arg2, ...) where x is the variable that we will be minimizing against.
      deriv (function): The derivative of the energy function. The function should take the form func(x, arg1, arg2, ...) where x is the variable that we will be minimizing agains.
      second_deriv (function): The second derivative of the energy function. The function should take the form func(x, arg1, arg2, ...) where x is the variable that we will be minimizing agains.
      args (tuple): The arguments to pass to the energy function and its derivatives
    """
    possible_minima = find_roots(deriv, min_x, max_x, x_steps, args)
    minima = []
    for root in possible_minima:
        if check_root_is_minimum(root, deriv, second_deriv, resolution=resolution, args=args):
            minima.append(root)

    diff = abs(previous_magnetization - cos(minima))
    magnetization = min(zip(diff, cos(minima)))

    return magnetization


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

