# optimized_distance.pyx
import numpy as np
cimport numpy as np
from itertools import combinations

# Declare the type for NumPy arrays
ctypedef np.float64_t DOUBLE_t
ctypedef np.int64_t INT_t

# Function to compute total pairwise distances
def compute_group_distance_cython(np.ndarray[DOUBLE_t, ndim=2] distance_matrix, list group):
    cdef int i, j, size = len(group)
    cdef double total_distance = 0.0
    
    for i in range(size):
        for j in range(i + 1, size):
            total_distance += distance_matrix[group[i], group[j]]
    return total_distance

# Function to compute the infimum total pairwise distance
def compute_infimum_distance(np.ndarray[DOUBLE_t, ndim=2] distance_matrix, int group_size):
    """
    Computes the infimum total pairwise distance for groups of given size.
    """
    cdef int n = distance_matrix.shape[0]
    cdef int number_of_groups = n // group_size
    cdef list all_indices = list(range(n))
    cdef list all_groups = list(combinations(all_indices, group_size))
    cdef list group_distances = []
    
    for group in all_groups:
        total_distance = compute_group_distance_cython(distance_matrix, list(group))  # Convert tuple to list
        group_distances.append(total_distance)

    # Sort distances
    group_distances = sorted(group_distances)
    infimum = sum(group_distances[:number_of_groups])
    return infimum
