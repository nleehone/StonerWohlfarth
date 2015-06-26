import scipy as sp
import numpy as np
import scipy.optimize
from numpy import sin, cos, abs, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autodiff import gradient, function

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


def hysteresis_loop(H, H_dir, Ku_dir, hs, energy_func, derivative, second_derivative, min_x=-pi, max_x=pi, x_steps=50, args=()):
    """Calculate the hysteresis loop for the Stoner Wohlfarth model.

    Params:
      H (list-like): Field values that we will be simulating
      H_dir (list-like (len 3)): The direction of the magnetic field
      Ku_dir (list-like (len 3)): The direction of the uniaxial anisotropy
      K_shape_dir (list-like (len 3)): The direction of the shape anisotropy
      energy_func (function):
      derivative (function):
      second_derivative (function):
      min_x (float): Minimum value in which to search for roots
      max_x (float): Maximum value in which to search for roots
      x_steps (int): Number of trials to do between [min_x, max_x] for finding roots
      args (tuple): Extra arguments that will be passed to the energy function and its derivatives

    Returns:
      np.ndarray: The hysteresis loop as (h, mx, my, mz) tuples
    """

    # Create the transformation matrix that will allow us to convert from problem space into xyz coords
    H_dir = np.array(H_dir)
    Ku_dir = np.array(Ku_dir)
    
    # If the two vectors are parallel then we cannot define the cross product. 
    if np.all(H_dir == Ku_dir):
        # Just use the 'up' vector (0, 0, 1) or the 'left' vector (1, 0, 0)
        if np.all(H_dir == [0, 0, 1]):
            v_dir = np.cross(H_dir, [1, 0, 0])
        else:
            v_dir = np.cross(H_dir, [0, 0, 1])
    else:
        v_dir = np.cross(H_dir, Ku_dir)
        
    v_dir = v_dir/np.linalg.norm(v_dir)
    Kp_dir = np.cross(v_dir, H_dir)
    Kp_dir = Kp_dir/np.linalg.norm(Kp_dir)
    
    U = np.array([v_dir, H_dir, Kp_dir]).T

    # Alpha is the angle between Ku and H
    alpha = np.arccos(np.dot(Ku_dir, H_dir))
    
    # Alpha prime is the angle between Kshape and H
    alpha_prime = alpha#np.dot(Kshape_dir, H_dir)

    mh_curve = []
    mag = -1

    for h in sorted(H):
        _, mag = find_magnetization(energy_func, derivative, second_derivative, mag, min_x, max_x, x_steps, args=(h, hs, alpha, alpha_prime))
        mag_angle = np.arccos(mag)
        m = (0, cos(mag_angle), sin(mag_angle))
        vals = (h,U.dot(m)[0], U.dot(m)[1], U.dot(m)[2])
        mh_curve.append(vals)

    return np.array(mh_curve)

def hysteresis_loop_anisotropy_distribution(H, H_dir, K_dirs, hs, energy_func, derivative, second_derivative, min_x=-pi, max_x=pi, x_steps=50, args=()):
    loops = []
    for k_dir in K_dirs:
        try:
            loop = hysteresis_loop(H, H_dir, k_dir, hs, energy_func, derivative, second_derivative, min_x=min_x, max_x=max_x, x_steps=x_steps, args=args)
        except KeyboardInterrupt:
            # Make sure we can exit out of the computation
            break
        except Exception as e:
            # Catch any other exceptions, print them, but allow the computation to continue
            print(e.message, e.args)
            continue
        else:
            loops.append(loop)
    return loops


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
