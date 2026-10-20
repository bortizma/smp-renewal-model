import numpy as np
from core import make_random


def single_circle(num_rows, radius):
    med = np.zeros([num_rows, num_rows]) #square
    for i in range(num_rows):
        for j in range(num_rows):
            if (i - num_rows // 2) ** 2 + (j - num_rows // 2) ** 2 <= radius * radius: 
                med[i, j] = 1
    return med


def single_circle_loc(num_rows,  cent_row, cent_col, radius):
    med = np.zeros([num_rows, num_rows]) 
    for i in range(num_rows):
        for j in range(num_rows):
            if (i - cent_row) ** 2 + (j - cent_col) ** 2 <= radius * radius: 
                med[i, j] = 1
    return med


def square_medium_nonoverlap(num_rows, square_side, density, seed=1, max_attempts=200000):
    med = np.zeros((num_rows, num_rows))
    rand = np.random.RandomState(seed=seed)

    attempts = 0
    while med.sum() / med.size < density:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Could not reach target density without overlapping squares.")

        center = np.array([
            rand.choice(np.arange(num_rows)),
            rand.choice(np.arange(num_rows)),
        ])

        r0 = max(center[0] - square_side // 2, 0)
        r1 = min(center[0] + square_side // 2, num_rows)
        c0 = max(center[1] - square_side // 2, 0)
        c1 = min(center[1] + square_side // 2, num_rows)

        if med[r0:r1, c0:c1].any():
            continue

        med[r0:r1, c0:c1] = 1

    return med


def circle_medium_nonoverlap(num_rows, num_cols, radius, density, seed=1, max_attempts=200000):
    med = np.zeros((num_rows, num_cols))
    rand = make_random(seed)

    jj, ii = np.meshgrid(np.arange(num_cols), np.arange(num_rows))

    attempts = 0
    while med.sum() / med.size < density:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Could not reach target density without overlapping circles.")

        center = np.array([
            rand.choice(np.arange(radius, num_rows - radius)),
            rand.choice(np.arange(radius, num_cols - radius)),
        ])

        mask = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 <= radius ** 2

        if med[mask].any():
            continue

        med[mask] = 1

    return med

def square_medium(num_rows, square_side, density, seed=1):
    """
    medium with random (overlapping) squares
    """
    med = np.zeros([num_rows, num_rows]) 
    rand = np.random.RandomState(seed=seed)
    while True:
        center = np.array([
                rand.choice(np.arange(num_rows)),
                rand.choice(np.arange(num_rows)),
            ])
        med[
            max(center[0] - square_side // 2, 0): min(center[0] + square_side // 2, num_rows - 1),
            max(center[1] - square_side // 2, 0): min(center[1] + square_side // 2, num_rows - 1),
        ] = 1 
        if med.sum() / (num_rows * num_rows) >= density:
            break
    return med


def circle_medium(num_rows, num_cols, radius, density, seed=1):
    """
    medium with random (overlapping) circles
    """

    med = np.zeros([num_rows, num_cols]) 
    rand = make_random(seed)
    jj, ii = np.meshgrid(np.arange(med.shape[1]), np.arange(med.shape[0]))
    while True:
        center = np.array([
                rand.choice(np.arange(num_rows)),
                rand.choice(np.arange(num_cols)),
            ])

        cond = np.sqrt((ii - center[0]) ** 2 + (jj - center[1]) ** 2) <= radius
        
        med[cond] = 1 

        

        if med.sum() / med.size >= density:
            break
    # now gradient
    return med


def medium_given_circles(num_rows, num_cols, centers, radii, ext=2):
    med = np.zeros([num_rows, num_cols])
    jj, ii = np.meshgrid(np.arange(med.shape[1]), np.arange(med.shape[0]))
    grad = np.zeros([2, num_rows, num_cols])
    for cent, r in zip(centers, radii):
        cond = np.sqrt((ii - cent[0]) ** 2 + (jj - cent[1]) ** 2) <= r
        med[cond] = 1
        seg = np.where(cond)
        grad[0, ...][seg] = ii[seg] - cent[0]
        grad[1, ...][seg]  = jj[seg] - cent[1]
        cond_ext = (np.sqrt((ii - cent[0]) ** 2 + (jj - cent[1]) ** 2) <= r + ext) & (~cond)
        seg_ext = np.where(cond_ext)
        grad[0, ...][seg_ext] = -ii[seg_ext] + cent[0]
        grad[1, ...][seg_ext] = -jj[seg_ext] + cent[1]

    return med, grad

