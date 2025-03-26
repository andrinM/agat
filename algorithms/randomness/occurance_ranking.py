import pandas as pd
import numpy as np
import sys
import os
import random
import math

current_dir = os.getcwd()
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Adjust the path to point to src
sys.path.append(src_dir)

from . import occ_optimization
from distance_matrix.distance_matrix import get_distance_matrix

class Occurance_Ranking:
    """
    A class to perform clustering based on occurrence ranking, considering must-link and cannot-link constraints.
    
    Attributes:
        df (DataFrame): The dataset to be clustered.
        group_size (int): The size of each group.
        alpha (int): The number of top groups to consider for member fitness.
        max_iter (int): The maximum number of iterations for optimization.
        n_solutions (int): The number of best solutions to return.
        ml_list (list): A list of must-link constraints.
        cl_list (list): A list of cannot-link constraints.
    """

    def __init__(self, df, group_size, alpha = 0.1, max_iter=100, n_solutions=1, ml_list=None, cl_list=None):
        """
        Initialize the Occurance_Ranking object with the given parameters.
        
        Args:
            df (DataFrame): The dataset to be clustered.
            group_size (int): The desired size for each cluster.
            alpha (int): The top alpha percentage of total possible groups to consider for member fitness.
            max_iter (int, optional): The maximum number of iterations. Default is 100.
            n_solutions (int, optional): The number of best solutions to retrieve. Default is 1.
            ml_list (list, optional): A list of must-link constraints. Default is an empty list.
            cl_list (list, optional): A list of cannot-link constraints. Default is an empty list.
        """
        self.df = df  # Store the dataset
        self.group_size = group_size
        self.alpha = alpha  # Number of top groups to consider for member fitness
        self.max_iter = max_iter  # Maximum number of iterations for optimization
        self.n_solutions = n_solutions  # Number of best solutions to return
        self.ml_list = ml_list if ml_list is not None else []  # Must-link constraints
        self.cl_list = cl_list if cl_list is not None else []  # Cannot-link constraints
        self.distance_matrix = get_distance_matrix(self.df, self.df)

    def fit(self):
        """
        Fit the model by performing clustering using occurrence ranking and considering constraints.
        
        This method performs the following steps:
            1. Calculates the distance matrix for the dataset.
            2. Preprocesses the must-link and cannot-link constraints.
            3. Creates all possible groups based on the distance matrix and group size.
            4. Calculates the member occurrence in the top 'alpha' groups.
            5. Sorts the members based on their occurrence in the top groups.
            6. Greedily creates an initial grouping based on member fitness.
            7. Optimizes the grouping by performing random permutations while respecting constraints.
        
        Returns:
            best_solutions (list): A list of the best solutions found.
        """
            
        n = len(self.df)
        # Step 1: Calculate the distance matrix for the dataset
                
        # Step 2: Preprocess the constraints (must-link and cannot-link)
        ml_graph, cl_graph, neighborhoods = self.preprocess_constraints(ml=self.ml_list,
                                                                        cl=self.cl_list,
                                                                        n=n)
        
        # Step 3: Generate all possible groups based on distance matrix and group size
        all_possible_groups = occ_optimization.compute_distances(self.distance_matrix, self.group_size)

        
        # Step 4: Get the member occurrence in the top alpha groups
        member_occurance = self.get_fittest_members(possible_groups=all_possible_groups,
                                                    alpha=self.alpha)

        # Step 4.1 Check if the length of member_occurance is not equal to n
        while len(member_occurance) != n:
            self.alpha += 0.05  # Increase alpha by 0.05
            # Get the member occurrence again with the updated alpha
            member_occurance = self.get_fittest_members(
                possible_groups=all_possible_groups, alpha=self.alpha)

        

        # Step 5: Sort the members by their occurrence in the top alpha groups (ascending)
        sorted_member_occurance = dict(sorted(member_occurance.items(), key=lambda item: item[1], reverse=False))
        # Step 6: Greedily create an initial grouping based on the sorted member occurrences
        initiale_grouping = self.greedy_grouping(
            ranking = sorted_member_occurance,
            distance_matrix=self.distance_matrix,
            ml_graph=ml_graph,
            cl_graph=cl_graph,
            group_size=self.group_size
        )
        
        # Step 7: Optimize the grouping using random permutations while respecting the constraints
        total_distance, best_solution = self.random_permutation_ml_cl(
            distance_matrix=self.distance_matrix,                       
            ml_graph=ml_graph,
            cl_graph=cl_graph,
            initial_grouping=initiale_grouping,
            n_iterations=self.max_iter,
            
        )
        
        # Extract the BEST grouping out of all best groupings
        #best_distance, best_grouping = best_solutions[0]

        # Transform dictionary of BEST groups into a 2d list
        #best_grouping = list(best_grouping.values())
        # Return the best solutions found
        return total_distance, best_solution


    def random_permutation_ml_cl(self, distance_matrix, ml_graph, cl_graph, initial_grouping, n_iterations):
        """
        Perform a series of random swaps between groups while respecting must-link and cannot-link constraints.

        This method iteratively swaps points between two randomly selected groups and checks if the swap violates
        any constraints, including must-link (ml_graph) and cannot-link (cl_graph) relationships. If the swap improves
        the total intra-group distance (i.e., lowers it), it is accepted; otherwise, it is reverted.

        Args:
            distance_matrix (pandas.DataFrame or numpy.ndarray): A matrix containing pairwise distances between points.
            ml_graph (dict): A dictionary where keys are indices of points, and values are sets of points that must be in the same group.
            cl_graph (dict): A dictionary where keys are indices of points, and values are lists of points that must not be in the same group.
            initial_grouping (dict): A dictionary representing the initial grouping of points, where keys are group indices and values are lists of point indices in each group.
            n_iterations (int): The number of iterations to perform the swapping process.

        Returns:
            tuple: A tuple containing:
                - float: The total distance of all groups after the permutations.
                - dict: The final grouping of points after performing the permutations.

        Notes:
            - The function uses random selection to choose groups and indices for swaps, making the process non-deterministic.
            - The `is_valid_swap` function ensures that swaps respect the must-link and cannot-link constraints defined in `ml_graph` and `cl_graph`.
            - If a swap is invalid or leads to a worse total distance, it is reverted.
            - The function aims to minimize the total pairwise distance within each group, considering the constraints.
        """
        # Convert initial grouping dictionary to a list of groups
        groups = list(initial_grouping.values())
        
        # Helper function to calculate the total distance for a group
        def calculate_group_distance(group):
            total_distance = 0
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    total_distance += distance_matrix[group[i], group[j]]
            return total_distance

        # Helper function to check if the swap violates any constraints
        def is_valid_swap(group1, group2, index1, index2):
            if ml_graph.get(index1) != set() or ml_graph.get(index2) != set():
                return False
            
            # Check cannot-link constraint for both groups
            for neighbor in cl_graph.get(index1, []):
                if neighbor in group2:
                    return False
            for neighbor in cl_graph.get(index2, []):
                if neighbor in group1:
                    return False

            return True

        for _ in range(n_iterations):
            # Randomly select two groups and two indices to swap
            group_idx1, group_idx2 = random.sample(range(len(groups)), 2)
            group1, group2 = groups[group_idx1], groups[group_idx2]
            old_group1 = group1.copy()
            old_group2 = group2.copy()
            index1 = random.choice(group1)
            index2 = random.choice(group2)
            
            # Perform the swap
            group1.remove(index1)
            group2.remove(index2)
            group1.append(index2)
            group2.append(index1)

            # Check if the swap is valid (doesn't violate constraints)
            if is_valid_swap(group1, group2, index1, index2):
                # Calculate the new total distances for both groups
                new_distance_group1 = calculate_group_distance(group1)
                new_distance_group2 = calculate_group_distance(group2)
                
                # If the new distances are better (lower), keep the swap
                total_distance_before = calculate_group_distance(old_group1) + \
                                        calculate_group_distance(old_group2)
                total_distance_after = new_distance_group1 + new_distance_group2
                
                if total_distance_after < total_distance_before:
                    groups[group_idx1] = group1
                    groups[group_idx2] = group2
                    
                else:
                    # Otherwise, revert the swap
                    group1.remove(index2)
                    group2.remove(index1)
                    group1.append(index1)
                    group2.append(index2)
            else:
                # Revert the swap if invalid
                group1.remove(index2)
                group2.remove(index1)
                group1.append(index1)
                group2.append(index2)
        
        # Convert list of groups back into dictionary format
        #final_grouping = {i: groups[i] for i in range(len(groups))}
        
        # Calculate the total distance for all groups
        total_distance = sum(calculate_group_distance(group) for group in groups)
        
        return total_distance, groups


    def get_fittest_members(self, possible_groups, alpha):
        """
        Sorts the dictionary by values and calculates the count of each member
        across the top k tuples (based on the smallest values).

        Parameters:
        - possible_groups: Dictionary with tuples as keys and numeric values.
        - alpha: Number of top alpha groups to consider.

        Returns:
        - A dictionary with indices (member ID) as keys and their counts as values.
        """
        # Calculate how many of the best groups to concider
        alpha = math.ceil(len(possible_groups) * self.alpha)

        # Step 1: Sort the dictionary by its values
        sorted_items = sorted(possible_groups.items(), key=lambda x: x[1])
        
        # Step 2: Extract the top k tuples
        top_alpha_tuples = [key for key, _ in sorted_items[:alpha]]
        
        # Step 3: Count occurrences of each indice (member ID) in the top alpha tuples
        top_alpha_members = {}
        for tup in top_alpha_tuples:
            for num in tup:
                top_alpha_members[num] = top_alpha_members.get(num, 0) + 1
        
        return top_alpha_members

    def greedy_grouping(self, ranking, distance_matrix, ml_graph, cl_graph, group_size):
        """
        Perform greedy grouping of items based on a ranking while considering must-link and cannot-link constraints.

        This method iteratively forms groups of a specified size (`group_size`) by selecting items from a ranked list.
        It prioritizes including must-link (ml_graph) items and avoids including cannot-link (cl_graph) items in the same group.
        If it is not possible to form a group of the specified size due to constraints, the method retries with a random 
        permutation of the ranking list.

        Args:
            ranking (dict): A dictionary of ranked items, where keys are indices and values are the ranks.
            distance_matrix (pandas.DataFrame or numpy.ndarray): A matrix containing pairwise distances between items.
            ml_graph (dict): A dictionary where keys are indices of items, and values are sets of items that must be in the same group.
            cl_graph (dict): A dictionary where keys are indices of items, and values are lists of items that must not be in the same group.
            group_size (int): The desired size of each group.

        Returns:
            dict: A dictionary where keys are group indices, and values are lists of item indices in each group.

        Raises:
            ValueError: If the number of must-link items exceeds the specified group size.
            RuntimeError: If the maximum number of retries is exceeded and the grouping could not be completed successfully.

        Notes:
            - The function starts by forming groups based on the ranking list, while ensuring that must-link items are included and cannot-link items are excluded.
            - If a group cannot be completed due to constraint violations, it attempts to form new groups by randomly permuting the ranking list (up to 30% of the ranking).
            - The process continues until all groups are successfully formed, or a maximum number of retries is reached.
        """
        not_done = True

        while not_done:
            groups = []  # List to hold the resulting groups
            used_indices = set()  # Set to track indices already assigned to groups
            success = True  # Flag to determine if all groups were completed
            
            for x in ranking:
                if x in used_indices:
                    continue

                current_group = set([x])
                used_indices.add(x)

                ml_indices = {idx for idx in ml_graph.get(x, []) if idx not in used_indices}
                if len(current_group) + len(ml_indices) <= group_size:
                    current_group.update(ml_indices)
                    used_indices.update(ml_indices)
                else:
                    raise ValueError("Must-link indices exceed group size. Adjust input data or group size.")

                while len(current_group) < group_size:
                    nearest_indices = np.argsort(np.min(distance_matrix[list(current_group)], axis=0))
                    valid_neighbor_found = False

                    for y in nearest_indices:
                        if y in used_indices or y in current_group:
                            continue
                        if any(y in cl_graph.get(idx, []) for idx in current_group):
                            continue

                        current_group.add(y)
                        used_indices.add(y)
                        ml_indices_y = {idx for idx in ml_graph.get(y, []) if idx not in used_indices}
                        if len(current_group) + len(ml_indices_y) <= group_size:
                            current_group.update(ml_indices_y)
                            used_indices.update(ml_indices_y)
                        else:
                            current_group.discard(y)
                            used_indices.discard(y)
                            break

                        valid_neighbor_found = True
                        if len(current_group) == group_size:
                            break

                    if not valid_neighbor_found:
                        break

                if len(current_group) == group_size:
                    groups.append(list(current_group))
                else:
                    success = False
                    break
            
            if success:
                return {i: groups[i] for i in range(len(groups))}

            # If unsuccessful, randomly permute the first 30% of the ranking list and retry
            retry_count = max(1, int(0.3 * len(ranking)))  # Ensure at least one element is shuffled
            ranking_items = list(ranking.items())  # Convert dict to a list of (key, value) tuples

            # Shuffle only the first `retry_count` elements
            subset = ranking_items[:retry_count]
            np.random.shuffle(subset)  # Shuffle key-value pairs together

            # Rebuild the dictionary
            new_ranking = dict(subset + ranking_items[retry_count:])

            ranking = new_ranking  # Update dictionary
            
            
        
        raise RuntimeError("Max retries exceeded. Grouping could not be completed successfully. Run Again!")
    
    
    def preprocess_constraints(self, ml, cl, n):
        "Create a graph of constraints for both must- and cannot-links"

        """ Represent the graphs using adjacency-lists. This creates two empty dictionaries of size n
        and with keys from 0 to n. Each key is associated with an empty set.
        Keyword arguments:
        ml -- list of must-links
        cl-- list of cannot-links
        n -- number of rows (data points)
        """
        ml_graph, cl_graph = {}, {}
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        """ Key of dict acts as index of a data point. The set (value) represent the neighboors
        of the index. Example: 
        ml = [(1,2),(2,4)]
        ml_graph = {0:set(), 1:{2}, 2:{1,4}, 3:set(), 4:{2}}
        returns adjacency list
        """
        for (i, j) in ml:
            ml_graph[i].add(j)
            ml_graph[j].add(i)

        for (i, j) in cl:
            cl_graph[i].add(j)
            cl_graph[j].add(i)
        
        def dfs(i, graph, visited, component): # Depth First Search Algorithm
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        """ Keyword arguments:
        component -- list of connected nodes (data points)
        neighborhoods -- list of lists. Each within list is a component
        """
        # Run DFS from each node to get all the graph's components
        # and add an edge for each pair of nodes in the component (create a complete graph)
        # See http://www.techiedelight.com/transitive-closure-graph/ for more details
        visited = [False] * n
        neighborhoods = []
        for i in range(n): # traverse each data point (node)
            if not visited[i] and ml_graph[i]: # If ml_graph[i] has values then we get true otherwise false
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
                neighborhoods.append(component) # neighborhoods is a list of lists. Each within list is a component
        
        """ This for loop adds nodes (data points) to the cl_graph if they have a transitive
        inference of a cannot-link constrans. It basically adds some of the must-links to the
        cl_graph, if they violate consistency 
        """
        for (i, j) in cl:
            for x in ml_graph[i]: # i is the key of ml_graph and x is the corresponding value
                add_both(cl_graph, x, j)

            for y in ml_graph[j]:
                add_both(cl_graph, i, y)

            for x in ml_graph[i]:
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)

        """ This for loop checks if any tuple is a must-link AND a cannot-link constraint. 
        If this is the case, an exception gets thrown.
        """
        for i in ml_graph: # iterate over the keys
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise ValueError('Inconsistent constraints between {} and {}'.format(i, j))

        return ml_graph, cl_graph, neighborhoods

   
