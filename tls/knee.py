"""
@author: Li Xi
@file: knee.py
@time: 2020/12/7 16:18
@desc:
"""
import math

import numpy as np
def calculate_avg_sum(scores, a=0.001):
    for i in range(1, len(scores)):
        scores[i] += scores[i-1]
        scores[i] /= (i+1)
    scores = [(x - min(scores)) / (max(scores) - min(scores)) if (max(scores) - min(scores)) != 0 else 0.0 for x in scores]

    scores = [-math.log(x + a) for x in scores]
    # scores = [x-min(scores) for x in scores]
    return scores

def detect_knee_point(values):
    """
    From:
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    # get coordinates of all the points
    n_points = len(values)
    all_coords = np.vstack((range(n_points), values)).T
    # get the first point
    first_point = all_coords[0]
    # get vector between first and last point - this is the line
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # distance to line is the norm of vec_to_line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # knee/elbow is the point with max distance value
    best_idx = np.argmax(dist_to_line)
    return best_idx
