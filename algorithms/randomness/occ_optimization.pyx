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

# Function to compute total pairwise distances for all groups and return as a dictionary
def compute_distances(np.ndarray[DOUBLE_t, ndim=2] distance_matrix, int group_size):
    """
    Computes the total pairwise distance for all groups of a given size.
    Returns a dictionary with groups as keys and their corresponding distances as values.
    """
    cdef int n = distance_matrix.shape[0]
    cdef int number_of_groups = n // group_size
    cdef list all_indices = list(range(n))
    cdef list all_groups = list(combinations(all_indices, group_size))
    cdef dict group_distances = {}  # Initialize an empty dictionary
    
    for group in all_groups:
        total_distance = compute_group_distance_cython(distance_matrix, list(group))  # Convert tuple to list
        group_distances[tuple(group)] = total_distance  # Use tuple(group) as the key

    return group_distances
