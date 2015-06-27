import numpy as np

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


