import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autodiff import gradient, function

from utilities import find_magnetization

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


