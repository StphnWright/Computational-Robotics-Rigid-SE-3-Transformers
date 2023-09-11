from numba import njit, prange
import numpy as np

def transform_is_valid(t, tolerance=1e-3):
    """ Check if array is a valid transform.
    You can refer to the lecture notes to 
    see how to check if a matrix is a valid
    transform. 

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """
    # Is t 4x4?
    if t.shape != (4, 4): return False

    # Is t numeric?
    if np.any(np.isnan(t)): return False

    # Is R transpose the left inverse of R?
    R = t[:3, :3]
    if not np.isclose(np.matmul(R.T, R), np.eye(3), atol=tolerance).all(): return False

    # Is R transpose the right inverse of R?
    if not np.isclose(np.matmul(R, R.T), np.eye(3), atol=tolerance).all(): return False

    # Is the determinant of R one?
    if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance): return False

    # Finally, the last row must be [0, 0, 0, 1]
    return np.isclose(np.array([[0, 0, 0, 1.0]]), t[3, :], atol=tolerance).all()

def transform_concat(t1, t2):
    """ Concatenate two transforms. Hint: 
        use numpy matrix multiplication. 

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """

    # Check the validity of t1 and t2
    if not transform_is_valid(t1): raise ValueError('t1 is not a valid transform')
    if not transform_is_valid(t2): raise ValueError('t2 is not a valid transform')

    # Return t1 times t2
    return np.matmul(t1, t2)

def transform_point3s(t, ps):
    """ Transfrom a list of 3D points
    from one coordinate frame to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    
    # Check the validity of t
    if not transform_is_valid(t): raise ValueError('t is not a valid transform')

    # Check that ps is a 2-D matrix with 3 columns
    if len(ps.shape) != 2 or ps.shape[1] != 3: raise ValueError('ps does not have the correct shape')

    # Add a column of ones to ps (that is, make the points homogeneous)
    psh = np.hstack([ps, np.ones((ps.shape[0], 1))])

    # Apply the transformation matrix to the points
    # Note that the points matrix lists points by rows, so they must be 
    # transposed first for column-wise multiplication, then transposed back
    ps_after = (np.matmul(t, psh.T)).T

    # Return the points, excluding the last column
    return ps_after[:, :3]

def transform_inverse(t):
    """Find the inverse of the transfom. Hint:
        use Numpy's linear algebra native methods. 

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """

    # Check the validity of t
    if not transform_is_valid(t): raise ValueError('t is not a valid transform')

    # Return the inverse of t
    return np.linalg.inv(t)