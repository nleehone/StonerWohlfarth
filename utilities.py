import numpy as np
from numpy import sin, cos, abs, pi
import scipy as sp
import matplotlib.pyplot as plt


def plot_direction_distribution(directions):
    # First make sure that the directions are normalized
    dirs = np.array([v/np.linalg.norm(v) for v in directions])

    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_aspect('equal')

    # Make a sphere that has slightly smaller radius that 1 unit
    u = np.linspace(0, 2*pi, 100)
    v = np.linspace(0, pi, 100)
    x = 0.95 * np.outer(np.cos(u), np.sin(v))
    y = 0.95 * np.outer(np.sin(u), np.sin(v))
    z = 0.95 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='w')

    # Plot the directions on the sphere
    ax.scatter(dirs[:,0], dirs[:,1], dirs[:,2], color='k', s=1)

    ax.azim = 0
    ax.elev = 0

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    # Draw a second copy but this time looking from above
    ax = fig.add_subplot(122, projection='3d')
    ax.set_aspect('equal')

    ax.plot_surface(x, y, z, color='w')

    # Plot the directions on the sphere
    ax.scatter(dirs[:,0], dirs[:,1], dirs[:,2], color='k', s=1)

    ax.azim = 0
    ax.elev = 90

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.3)
    plt.show()

    return fig


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


def find_coercivity(arg0, arg1=None):
    """Find the coercivity of a magnetization loop. The data should
    represent only one branch of the magnetization loop as this routine
    needs to sort the data prior to finding the coercivity.

    The function has two signatures
    find_coercivity(H, M)
    find_coercivity(H_M)

    Params:
      H (list-like): List of fields. The length of M and H must match.
      M (list-like): List of magnetization values
      H_M (list-like): Pairs of (H, M) values

    Raises:
      ValueError: If we cannot find the coercivity (magnetization did not change sign)
    """
    if arg1 is not None:
        H_M = np.array([arg0, arg1]).T
    else:
        H_M = np.array(arg0)

    # Check whether or not the magnetization changes sign.
    # If it does not, then there is no point in searching for the coercivity since it does not exist.
    if np.sign(min(H_M[:,1])) == np.sign(max(H_M[:,1])):
        raise ValueError('Could not find the coercivity as the magnetization does not change sign.')

    # Sort the (H, M) pairs by increasing field
    H_M = H_M[H_M[:,0].argsort()]

    previous_h, sign = H_M[0,0], np.sign(H_M[0,1])
    previous_m = H_M[0, 1]
    # We already checked that the coercivity exists, so we are guaranteed to hit the return statement
    for h, m in zip(H_M[:,0], H_M[:,1]):
        if sign != np.sign(m):
            # When the sign changes we know that we have just passed the coercivity
            # Return the zero point by linear interpolation
            return (h - previous_h)/(m - previous_m)*(-previous_m) + previous_h
        else:
            previous_h = h
            previous_m = m


def find_remanance(arg0, arg1=None):
    """Find the remanance of a magnetization loop. The data should represent
    only one branch of the magnetization loop as this routine needs to sort
    the data prior to finding the remanance.

    The function has two signatures
    fund_remanance(H, M)
    find_remanance(H_M)

    Params:
      H (list-like): List of fields. The length of M and H must match
      M (list-like): List of magnetization values.
      H_M (list-like): Pairs of (H, M) values

    Raises:
      ValueError: If we cannot find the remanance (field does not cross zero)
    """
    if arg1 is not None:
        H_M = np.array([arg0, arg1]).T
    else:
        H_M = np.array(arg0)

    # Check whether or not the field goes through zero
    if not (min(H_M[:,0]) <= 0 and max(H_M[:,0]) >= 0):
        raise ValueError('Could not find the remanance since the field does not pass through zero.')

    H_M = H_M[H_M[:,0].argsort()]

    previous_h = H_M[0,0]
    sign = np.sign(previous_h)
    previous_m = H_M[0,1]
    for h, m in zip(H_M[:,0], H_M[:,1]):
        if np.sign(h) != sign:
            # When the sign changes we know that we just passed through zero field
            # Use linear interpolation between the previous and current points to find the remanance
            return (m - previous_m)/(h - previous_h)*(-previous_h) + previous_m
        else:
            previous_h = h
            previous_m = m


