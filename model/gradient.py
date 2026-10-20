import numpy as np
import scipy
import scipy.ndimage


_DX_ = np.array([
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

_MULT_ = np.linalg.inv(_DX_.T.dot(_DX_)).dot(_DX_.T)


def point_grad(med, i, j):
    dy = np.array([
        med[i, j + 1], med[i, j - 1], med[i + 1, j], 
        med[i - 1, j], med[i + 1, j + 1], med[i + 1, j - 1], 
        med[i - 1, j + 1], med[i - 1, j - 1]]
    ) - med[i, j]

    grad = _MULT_.dot(dy)
    return grad


def reliable_gradient_field(medium, smooth_sigma=1, normalize=True):
    solid = medium > 0
    distance_to_solid = scipy.ndimage.distance_transform_edt(~solid)
    distance_to_void = scipy.ndimage.distance_transform_edt(solid)
    signed_distance = distance_to_solid - distance_to_void

    if smooth_sigma is not None:
        signed_distance = scipy.ndimage.gaussian_filter(signed_distance, smooth_sigma)

    grad = np.array(np.gradient(signed_distance, edge_order=2))
    if normalize:
        nrm = np.sqrt((grad * grad).sum(axis=0))
        nrm[nrm == 0] = 1
    else:
        nrm = 1
    grad = grad / nrm
    return grad


def gradient_field(medium, normalize=False):
    """
    Calculates the normal field of a binary medium.
    :param medium: 2D binary array representing the medium
    :param normalize: boolean argument. If True, then normalizes the gradient to have norm 1 at every coordinate.
    :return: 2 * m * n array grad. grad[0, :, :] gives the 1st coordinate of the gradient at each pixel, and
    grad[1, :, :] gives the 2nd coordinate.
    It uses the following math:
    f(x + dx1) - f(x) = g^T dx1
    f(x + dx2) - f(x) = g^T dx2
    f(x + dx3) - f(x) = g^T dx3
    f(x + dx4) - f(x) = g^T dx4
    y = Ax --> x = (A'A)^(-1)A' y
    x = solve(A, y)
    dy = Dx g ==> g = (Dx^T Dx)^(-1) * Dx^T dy
    Dx = 0  1
         0 -1
         1  0
        -1  0
         1  1
         1 -1
        -1  1
        -1 -1
    g = (f(x + eps) - f(x)) / 4 + (f(x) - f(x - eps)) / 4
      = 1/4 * dy1 + 1/4 * dy2 = <(1/4  1/4), (dy1, dy2) >

    """
    dy = np.array([
        np.roll(medium, shift=-1, axis=1),
        np.roll(medium, shift=1, axis=1),
        np.roll(medium, shift=-1, axis=0),
        np.roll(medium, shift=1, axis=0),
        np.roll(np.roll(medium, shift=-1, axis=0), shift=-1, axis=1),
        np.roll(np.roll(medium, shift=-1, axis=0), shift=1, axis=1),
        np.roll(np.roll(medium, shift=1, axis=0), shift=-1, axis=1),
        np.roll(np.roll(medium, shift=1, axis=0), shift=1, axis=1),
    ]
    )
    dy = np.array([medium - d for d in dy])
    dx = np.array([
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ])
    mult = np.linalg.inv(dx.T.dot(dx)).dot(dx.T)
    grad = np.array([
        np.array([a * b * medium for a, b in zip(mult[i, :], dy)]).sum(axis=0)
        for i in range(2)
    ])

    if normalize:
        norms = np.sqrt(grad ** 2).sum(axis=0)+1 # add +1 for solve the error in this line
        grad = np.array([np.nan_to_num(g / norms, 0) for g in grad])

    return grad

