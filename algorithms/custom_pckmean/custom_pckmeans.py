import pandas as pd
import numpy as np
import math
import sys
import os
import random
import pulp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import importlib


from algorithms.custom_pckmean.constraints import preprocess_constraints
from distance_matrix.distance_matrix import get_distance_matrix



class custom_PCKMeans:
    def __init__(self, df, group_size, n_clusters=None, max_iter=100, ml_list=None, cl_list=None, lp_time_limit=None ):
        """
        Initialize the custom_PCKMeans instance.

        Parameters:
        - df (pd.DataFrame): The dataset to be clustered.
        - n_clusters (int): The number of clusters to form. Default is 10% of the dataset size (rounded up).
        - max_iter (int): Maximum number of iterations for the clustering algorithm. Default is 100.
        - ml_list (list): Must-link constraints. Default is None.
        - cl_list (list): Cannot-link constraints. Default is None.
        """
        self.df = df  # Store the dataset
        self.group_size = group_size
        self.n_clusters = n_clusters if n_clusters is not None else math.ceil(0.1 * len(df)) + 1
        self.max_iter = max_iter  # Maximum number of iterations
        self.ml_list = ml_list if ml_list is not None else []  # Must-link constraints
        self.cl_list = cl_list if cl_list is not None else []  # Cannot-link constraints
        self.lp_time_limit = lp_time_limit
        

    def fit(self):

        # Scale the hom features
        for hom in self.df.filter(regex="^hom").columns:
            # Get the global minimum and maximum for the hom column
            global_min = self.df[hom].min()
            global_max = self.df[hom].max()

            # Perform MinMax scaling on hom column
            self.df[hom] = (self.df[hom] - global_min) / (global_max - global_min)

        n = self.df.shape[0]

        # Preprocess ml and cl constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml = self.ml_list, cl = self.cl_list, n = n)
        
        # Calculate distance_matrix for total_distance evaluation
        distance_matrix = get_distance_matrix(df1=self.df,
                                              df2=self.df,
                                              scale=False)
        
        if self.ml_list:
            # neighborhood_clusters are all groups with must-links the same size as group_size
            neighborhood_clusters, df_without_neighborhood_members = self.fill_neighborhoods(df = self.df,
                                                                                         neighborhoods=neighborhoods,
                                                                                         group_size = self.group_size,
                                                                                         cl_graph = cl_graph)
            # Run kmean
            unbalanced_final_clusters, new_centroids, distance_matrix_initial_centroids = self.kmeans_with_constraints(df = df_without_neighborhood_members,
                                                                k = self.n_clusters,
                                                                cl_graph= cl_graph,
                                                                max_iter=self.max_iter)
            # Balance cluster_size (divisible by group_size)
            balanced_clusters = self.balance_clusters_cl(unbalanced_final_clusters = unbalanced_final_clusters,
                                                     group_size= self.group_size,
                                                     cl_graph=cl_graph)
            
            # Create distance_matrices, based on balanced clusters
            all_distance_matrices = self.get_all_distance_matrices(balanced_clusters= balanced_clusters)
            # Solve lp for all distance matrices
            lp_solution = self.solve_all(all_distance_matrices, self.group_size, self.lp_time_limit)

            # Formate solution
            final_grouping = [[[balanced_clusters[i][index] for index in group] for group in sublist]
            for i, sublist in enumerate(lp_solution)]
            final_grouping = [inner_list for sublist in final_grouping for inner_list in sublist]
            final_grouping = final_grouping + neighborhood_clusters
            total_distance = self.get_total_distance(distance_matrix=distance_matrix, groups = final_grouping)

            return total_distance, final_grouping
         
        else: # Case, where there are no must-links
            df_without_neighborhood_members = self.df
            unbalanced_final_clusters, new_centroids, distance_matrix_initial_centroids = self.kmeans_with_constraints(df = df_without_neighborhood_members,
                                                                k = self.n_clusters,
                                                                cl_graph= cl_graph,
                                                                max_iter=self.max_iter)
            # Balance cluster_size (divisible by group_size)
            balanced_clusters = self.balance_clusters_cl(unbalanced_final_clusters = unbalanced_final_clusters,
                                                     group_size= self.group_size,
                                                     cl_graph=cl_graph)
            # Create distance_matrices, based on balanced clusters
            all_distance_matrices = self.get_all_distance_matrices(balanced_clusters= balanced_clusters)
            # Solve lp for all distance matrices
            lp_solution = self.solve_all(all_distance_matrices, self.group_size, self.lp_time_limit)

            # Formate solution
            final_grouping = [[[balanced_clusters[i][index] for index in group] for group in sublist]
            for i, sublist in enumerate(lp_solution)]
            final_grouping = [inner_list for sublist in final_grouping for inner_list in sublist]
            total_distance = self.get_total_distance(distance_matrix=distance_matrix, groups = final_grouping)
            
            return total_distance, final_grouping

    def get_all_distance_matrices(self, balanced_clusters):
        all_distance_matrices = []
        for balanced_cluster in balanced_clusters:
            subset_df = self.df.iloc[balanced_cluster]
            distance_matrix = get_distance_matrix(df1=subset_df,df2=subset_df,scale= False)
            all_distance_matrices.append(distance_matrix)
        return all_distance_matrices

    def solve_all(self, distance_matrices, group_size, lp_time_limit):
        """
        Solve multiple Linear Programming (LP) problems for a set of distance matrices using multithreading.

        Args:
            distance_matrices (list): A list of distance matrices, where each matrix represents the distances between data points.
            group_size (int): The size of each group to be formed.
            lp_time_limit (int): The maximum time allowed for each LP solver to run.

        Returns:
            list: A list of group assignments, where each entry corresponds to the solution of an LP problem for a given distance matrix.
        """
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.solve_lp, dm, group_size, lp_time_limit) for dm in distance_matrices]
            for future in futures:
                results.append(future.result())
        return results

    def solve_lp(self, distance_matrix, group_size, lp_time_limit):
        """
        Solve a Linear Programming (LP) problem to minimize the total intra-group distances for a given distance matrix.

        Args:
            distance_matrix (dict): A dictionary representing the pairwise distance matrix between data points.
            group_size (int): The size of each group to be formed.
            lp_time_limit (int): The maximum time allowed for the LP solver to run.

        Returns:
            list or None: A list of groups (each group is a list of indices) if an optimal solution is found; otherwise, None.
        
        Notes:
            - The method uses the CBC solver to solve the LP problem.
            - The problem formulation includes decision variables to assign points to groups and auxiliary variables to represent pairwise group membership.
        """        
        n = len(distance_matrix)
        num_groups = n // group_size

        # Problem definition
        problem = pulp.LpProblem("Minimize_Grouping_Distance", pulp.LpMinimize)

        # Decision variables
        # y[i, g] is 1 if point i is assigned to group g, 0 otherwise
        y = pulp.LpVariable.dicts('y', ((i, g) for i in range(n) for g in range(num_groups)), 
                                lowBound=0, upBound=1, cat=pulp.LpBinary)
        
        # Auxiliary variables
        # z[i, j, g] is 1 if both i and j are in group g, 0 otherwise
        z = pulp.LpVariable.dicts('z', ((i, j, g) for i in range(n) for j in range(n) if i < j for g in range(num_groups)),
                                lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Objective function: minimize the sum of distances for grouped points
        problem += pulp.lpSum(distance_matrix[i, j] * z[i, j, g] 
                            for i in range(n) for j in range(n) if i < j 
                            for g in range(num_groups))

        # Constraint 1: Each point must be in exactly one group
        for i in range(n):
            problem += pulp.lpSum(y[i, g] for g in range(num_groups)) == 1

        # Constraint 2: Each group must have exactly k members
        for g in range(num_groups):
            problem += pulp.lpSum(y[i, g] for i in range(n)) == group_size

        # Constraint 3: Define the auxiliary variable z[i, j, g]
        for i in range(n):
            for j in range(n):
                if i < j:
                    for g in range(num_groups):
                        # z[i, j, g] can only be 1 if both y[i, g] and y[j, g] are 1
                        problem += z[i, j, g] <= y[i, g]
                        problem += z[i, j, g] <= y[j, g]
                        problem += z[i, j, g] >= y[i, g] + y[j, g] - 1

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit= lp_time_limit)
        status = problem.solve(solver)

        # If the problem is solved, extract the results
        if status == pulp.LpStatusOptimal:
            # Create a list to store the groups
            groups = [[] for _ in range(num_groups)]

            # Assign points to groups
            for i in range(n):
                for g in range(num_groups):
                    if pulp.value(y[i, g]) == 1:
                        groups[g].append(i)
                        break
            
            return groups  # Return the groups as a 2D list of indices
        else:
            print(f"Problem Status: {pulp.LpStatus[status]}")
            print("No optimal solution found.")
            return None
        
    def balance_clusters_cl(self, unbalanced_final_clusters, group_size, cl_graph):
        """
        Balances clusters so that every cluster is divisible by group_size, 
        while respecting cannot-link constraints.

        Args:
        - unbalanced_final_clusters: A list of lists where each list represents a cluster.
        - group_size: Integer representing the required group size.
        - cl_graph: Dictionary where keys are indices and values are lists of indices they cannot link with.

        Returns:
        - unbalanced_final_clusters: Balanced list of clusters.
        """
        number_of_balances = 0
        while True:
            give = {}
            take = {}
            cluster_sizes = [len(cluster) for cluster in unbalanced_final_clusters]

            for idx, cluster_size in enumerate(cluster_sizes):
                r = cluster_size % group_size
                if (r != 0) and (r >= group_size / 2):
                    take[idx] = r  # Add a key-value pair
                elif (r != 0) and (r < group_size / 2):
                    give[idx] = r  # Add a key-value pair

            # Take an element from the first give cluster and add it to the first element of the first take cluster
            if give and take:
                # Case 1: Both `give` and `take` are not empty
                give_keys = list(give.keys())
                take_keys = list(take.keys())
                for g_key in give_keys:
                    for t_key in take_keys:
                        if unbalanced_final_clusters[g_key]:  # Ensure the list is not empty
                            chosen_element = random.choice(unbalanced_final_clusters[g_key])

                            # Check for cannot-link constraints
                            if all(
                                chosen_element not in cl_graph.get(member, [])
                                for member in unbalanced_final_clusters[t_key]
                            ):
                                unbalanced_final_clusters[t_key].append(chosen_element)
                                unbalanced_final_clusters[g_key].remove(chosen_element)
                                number_of_balances += 1
                                break
                    else:
                        continue
                    break

            elif not take and give:
                # Case 2: `take` is empty and `give` is not
                give_keys = list(give.keys())
                for g_key in give_keys:
                    for g_key_2 in give_keys:
                        if g_key != g_key_2 and unbalanced_final_clusters[g_key]:
                            chosen_element = random.choice(unbalanced_final_clusters[g_key])

                            # Check for cannot-link constraints
                            if all(
                                chosen_element not in cl_graph.get(member, [])
                                for member in unbalanced_final_clusters[g_key_2]
                            ):
                                
                                unbalanced_final_clusters[g_key_2].append(chosen_element)
                                unbalanced_final_clusters[g_key].remove(chosen_element)
                                number_of_balances += 1
                                break
                    else:
                        continue
                    break

            elif not give and take:
                # Case 3: `give` is empty and `take` is not
                take_keys = list(take.keys())
                for t_key in take_keys:
                    for t_key_2 in take_keys:
                        if t_key != t_key_2 and unbalanced_final_clusters[t_key]:
                            chosen_element = random.choice(unbalanced_final_clusters[t_key])

                            # Check for cannot-link constraints
                            if all(
                                chosen_element not in cl_graph.get(member, [])
                                for member in unbalanced_final_clusters[t_key_2]
                            ):
                                unbalanced_final_clusters[t_key_2].append(chosen_element)
                                unbalanced_final_clusters[t_key].remove(chosen_element)
                                number_of_balances += 1
                                break
                    else:
                        continue
                    break

            elif not give and not take:
                # Case 4: Both `give` and `take` are empty
                break
        # Remove empty clusters (lists)
        unbalanced_final_clusters = [cluster for cluster in unbalanced_final_clusters if cluster]
     

        return unbalanced_final_clusters

    def kmeans_with_constraints(self, df, k, cl_graph, max_iter=100):
        """
        Perform K-Means clustering with additional constraints on the assignment of points to clusters.

        This method initializes centroids using KMeans++, then iteratively updates the centroids and reassigns 
        points to clusters while considering constraints specified in the `cl_graph`. The algorithm stops when
        the clusters stabilize or the maximum number of iterations is reached.

        Args:
            df (pandas.DataFrame): The input data containing the features for clustering.
            k (int): The number of clusters to form.
            cl_graph (dict): A constraint graph specifying the must-link and cannot-link constraints between data points.
            max_iter (int, optional): The maximum number of iterations for the algorithm. Default is 100.

        Returns:
            tuple: A tuple containing:
                - clusters (list): A list of k clusters, where each cluster is a list of point indices.
                - new_centroids (pandas.DataFrame): The new centroids for the clusters after convergence.
                - distance_matrix_initial_centroids (pandas.DataFrame): The distance matrix between points and the centroids.
        
        Notes:
            - The algorithm uses KMeans++ to initialize the centroids and then iteratively refines the clusters.
            - The method assigns points to clusters while respecting constraints in `cl_graph`, ensuring that points 
              with must-link constraints are assigned to the same cluster and points with cannot-link constraints are 
              assigned to different clusters.
            - The algorithm stops either when the clusters stabilize or the maximum number of iterations is reached.
        """
        # Initialize centroids using KMeans++
        distance_matrix = get_distance_matrix(df1=df,df2=df, scale=False)
        original_indices = df.index

        # Creates a list of indices for the inital centroids
        initial_centroids = self.kmeans_plus_plus(distance_matrix=distance_matrix, original_indices = original_indices, k=k)
        # Initialize the distance matrix between all points and the initial centroids
        distance_matrix_initial_centroids = get_distance_matrix(df1=df,
                                                                df2=df.loc[initial_centroids],
                                                                scale=False)
        
        # Assigne points based on the initial centroids
        clusters = self.assign_points_to_centroids_with_constraints(distance_matrix_initial_centroids, original_indices, k, cl_graph)
        # Store the previous cluster assignments to check for convergence
        prev_clusters = None

        for _ in range(max_iter):
            # If clusters haven't changed, break the loop (convergence)
            if clusters == prev_clusters:
                break
            if prev_clusters is not None:
                if set(map(tuple, clusters)) == set(map(tuple, prev_clusters)):
                    break

            prev_clusters = clusters
            
            # Recalculate centroids
            new_centroids = pd.DataFrame(columns=df.columns)
            
            for cluster_points in clusters:
                # Get the new centroid for the cluster
                if len(cluster_points) > 0:
                    points_in_cluster = df.loc[cluster_points]
                    centroid = self.get_cluster_centers(points_in_cluster)    
                    # The code block where you want to suppress the warning
                    new_centroids = pd.concat([new_centroids, centroid], ignore_index=True)
                    
            
            # Update centroids
            distance_matrix_initial_centroids = get_distance_matrix(df1=df,
                                                                    df2=new_centroids,
                                                                    scale=False)
            
            # Reassign points to new centroids considering constraints
            clusters = self.assign_points_to_centroids_with_constraints(distance_matrix_initial_centroids, original_indices, k, cl_graph)
            distance_matrix_initial_centroids = pd.DataFrame(distance_matrix_initial_centroids)
            distance_matrix_initial_centroids.index = df.index
        
        return clusters, new_centroids, distance_matrix_initial_centroids

    def assign_points_to_centroids_with_constraints(self, distance_matrix, original_indices, n_clusters, cl_graph):
        """
        Assigns each point to the closest centroid while ensuring cannot-link constraints are respected.
        
        Args:
        - distance_matrix: A 2D numpy array representing the pairwise distances between points and centroids.
        - original_indices: A list of the original indices of the points in the dataset.
        - k: The number of centroids (clusters).
        - cl_graph: A dictionary where the key is the index of a point, and the value is a set of points that it cannot be assigned to the same cluster with.
        
        Returns:
        - centroid_assignments: A list of lists where each list contains the original indices of the data points 
        closest to a specific centroid.
        """
        # Initialize a list of empty lists for each centroid
        centroid_assignments = [[] for _ in range(n_clusters)]
        
        # Initialize a dictionary to keep track of the points already assigned to each centroid
        centroids_assigned = {i: [] for i in range(n_clusters)}
        
        # For each point, assign it to the closest centroid while respecting the cannot-link constraints
        for i in range(len(distance_matrix)):
            # Get the distances for the i-th point to all centroids
            distances_to_centroids = distance_matrix[i]
            
            # Check if the current point has any cannot-link constraint with existing points in any cluster
            valid_assignment_found = False
            
            while not valid_assignment_found:
                # Find the index of the closest centroid (minimum distance)
                closest_centroid_idx = np.argmin(distances_to_centroids)
                
                # Check if assigning this point to the closest centroid violates any cannot-link constraint
                violates_cannotlink = False
                for assigned_index in centroids_assigned[closest_centroid_idx]:
                    if original_indices[i] in cl_graph.get(assigned_index, set()):
                        violates_cannotlink = True
                        break
                
                if not violates_cannotlink:
                    # If no violation, assign the point to the closest centroid
                    centroid_assignments[closest_centroid_idx].append(original_indices[i])
                    centroids_assigned[closest_centroid_idx].append(original_indices[i])
                    valid_assignment_found = True
                else:
                    # If violated, mark that centroid as invalid and find the second closest centroid
                    distances_to_centroids[closest_centroid_idx] = np.inf  # Disable this centroid
        
        return centroid_assignments

    def kmeans_plus_plus(self, distance_matrix, original_indices, k):
        """
        KMeans++ initialization on a distance matrix with preserved indices.
        
        Parameters:
            distance_matrix (np.ndarray): Square distance matrix (n x n).
            original_indices (np.ndarray): The original indices of the DataFrame.
            k (int): Number of clusters.
        
        Returns:
            list: Indices of initial centroids (from the original DataFrame).
        """
        # Ensure the distance matrix is 2D and square
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("The distance matrix must be a square 2D array.")

        n_points = distance_matrix.shape[0]
        centroids = []

        # Randomly select the first centroid
        first_centroid = np.random.choice(n_points)
        centroids.append(first_centroid)

        for _ in range(1, k):
            # Calculate minimum distances to the existing centroids
            min_distances = np.min(distance_matrix[:, centroids], axis=1)

            # Square the distances as required by K-means++
            squared_distances = min_distances**2

            # Handle the case where all squared distances are zero
            total_distance = np.sum(squared_distances)
            if total_distance == 0:
                # Select randomly from points not already chosen as centroids
                remaining_indices = [i for i in range(n_points) if i not in centroids]
                next_centroid = np.random.choice(remaining_indices)
            else:
                # Select the next centroid with probability proportional to the squared distance
                probabilities = squared_distances / total_distance
                cumulative_probs = np.cumsum(probabilities)
                rand_val = np.random.rand()
                
                # Select the next centroid based on the cumulative probability distribution
                next_centroid = -1  # Default value in case of floating-point inaccuracies
                for i, prob in enumerate(cumulative_probs):
                    if rand_val <= prob:
                        next_centroid = i
                        break

            # Ensure the selected centroid is valid and not already in the list
            if next_centroid not in centroids:
                centroids.append(next_centroid)

        # Map centroids back to original indices using original_indices
        original_centroids = original_indices[centroids].tolist()
        
        
        return original_centroids

    def fill_neighborhoods(self, df, neighborhoods, group_size, cl_graph):
        """
        Assign additional members to existing neighborhoods to ensure the group size is met, while respecting constraints.

        This method calculates the centers of the existing neighborhoods, removes their members from the original dataframe,
        and then uses a distance matrix to assign new members to the neighborhoods, ensuring that each neighborhood reaches 
        the specified group size. The method uses a snake draft selection process to assign points, considering the constraints 
        in `cl_graph`.

        Args:
            df (pandas.DataFrame): The input data containing the features of potential neighborhood members.
            neighborhoods (list): A list of current neighborhoods, where each neighborhood is a list of indices of assigned members.
            group_size (int): The target size for each neighborhood.
            cl_graph (dict): A constraint graph specifying must-link and cannot-link constraints between data points.

        Returns:
            tuple: A tuple containing:
                - neighborhood_clusters (list): A list of updated neighborhoods with assigned members, each neighborhood is a list of point indices.
                - df_without_neighborhood_members (pandas.DataFrame): The dataframe with the newly assigned members removed.
        
        Notes:
            - The function calculates the mean of the existing neighborhoods to determine their centers.
            - It then computes a distance matrix between the remaining unassigned members and the neighborhood centers.
            - The snake draft selection method is used to assign points while respecting the given constraints.
        """
        # Get centers of neighborhoods (mean)
        neighborhood_centers = self.get_neighborhood_centers(self.df, neighborhoods)
        # Flatten neighborhoods into a 1d list
        neighborhood_members = [index for sublist in neighborhoods for index in sublist]

        # Remove indices if they are in a neighborhood
        df_without_neighborhood_members = df.drop(neighborhood_members)

        # Store the original indices (they get removed by creating the distance matrix)
        original_indices = df_without_neighborhood_members.index
        
        
        # Creating distance_matrix to neighborhood_centers
        distance_matrix = pd.DataFrame(get_distance_matrix(df1=df_without_neighborhood_members,
                                                           df2=neighborhood_centers,
                                                           scale=False))

        # Reindex
        distance_matrix.index = original_indices

        # Assigne members to neighborhoods
        neighborhood_clusters = self.snake_draft_selection(distance_matrix= distance_matrix, clusters=neighborhoods, group_size =group_size, cl_graph= cl_graph)

        # Remove assigned members from the original dataframe
        neighborhood_members = [index for sublist in neighborhood_clusters for index in sublist]
        df_without_neighborhood_members = df.drop(neighborhood_members)

        return neighborhood_clusters, df_without_neighborhood_members

    def snake_draft_selection(self, distance_matrix, clusters, group_size, cl_graph):
        """
        Perform a snake draft selection from the distance matrix, considering cannot-link constraints.

        Parameters:
        - distance_matrix: pd.DataFrame, rows are datapoints, columns are clusters.
        - clusters: list of lists, initialized 2D array to store indices for each cluster.
        - group_size: int, the size of each cluster.
        - cl_graph: dict, where keys are indices and values are sets of indices
        that the key index cannot be in the same cluster with.

        Returns:
        - clusters: updated 2D list with selected indices.
        - distance_matrix: updated distance matrix with selected rows removed.
        """
        # Create a copy of the distance matrix to avoid modifying the original
        temp_matrix = distance_matrix.copy()

        # Total number of clusters
        num_clusters = len(clusters)

        # Continue until all clusters have the desired group_size
        while any(len(cluster) < group_size for cluster in clusters):
            # Snake order: left to right for even rounds, right to left for odd rounds
            for cluster_index in range(num_clusters):
                # Reverse order for odd rounds
                if len(clusters[0]) % 2 != 0:
                    cluster_index = num_clusters - 1 - cluster_index

                # Skip this cluster if it's already full
                if len(clusters[cluster_index]) >= group_size:
                    continue

                # Get a sorted list of indices based on distances (closest first)
                sorted_indices = temp_matrix.iloc[:, cluster_index].sort_values().index.tolist()

                # Find the closest valid index that satisfies cannot-link constraints
                for candidate_index in sorted_indices:
                    # Check if the candidate_index violates any cannot-link constraints
                    if not any(
                        candidate_index in cl_graph.get(existing_index, set())
                        for existing_index in clusters[cluster_index]
                    ):
                        # Assign the valid candidate to the cluster
                        clusters[cluster_index].append(candidate_index)

                        # Remove the selected index from the distance matrix
                        temp_matrix = temp_matrix.drop(index=candidate_index)
                        break

        return clusters

    def get_neighborhood_centers(self, df , neighborhoods):
        """Parameters:
        df (pd.DataFrame): DataFrame containing the data points.
        neighborhoods: output of fucntion preprocess_constraints 
        
        Returns:
        neighborhood_centers (df): DataFrame containing the center of each neighborhood. The row index corresponds to the list (neighborhoods)
        index.
        """
        neighborhood_centers = pd.DataFrame(columns=df.columns)  # Initialize with column names

        for neighborhood in neighborhoods:
            subset_df = self.df.loc[neighborhood, :]  # Select rows from df based on neighborhood indices
            center = self.get_cluster_centers(subset_df)  # Calculate the cluster center
            
            # Convert the center (which could be a Series or DataFrame) into a DataFrame and concatenate
            neighborhood_centers = pd.concat([neighborhood_centers, center], ignore_index=True)
            
        return neighborhood_centers

    def get_cluster_centers(self, df):
        """
        Calculate the mean for each column in the given dataframe, considering the specific data type (e.g., homogenous, 
        one-hot encoded, multi-hot encoded).

        This method computes the mean of each feature in the dataframe based on its column type:
        - For homogeneous features (e.g., integer or continuous), the regular arithmetic mean is calculated.
        - For one-hot encoded features, the mode is determined to identify the most common category, with ties broken randomly.
        - For multi-hot encoded features, the mean is calculated based on the majority of the values in each feature, using a threshold of 0.5 
          to determine whether the feature should be marked as 1.

        Args:
            df (pandas.DataFrame): A dataframe containing the features of the data points for which the cluster centers are to be calculated.

        Returns:
            pandas.DataFrame: A DataFrame representing the mean values for each feature in the input dataframe, where each column corresponds 
                              to a feature and contains the calculated mean value for that feature.

        Notes:
            - The function processes different types of features (e.g., 'hom', 'hot', 'mult') based on their encoding scheme.
            - The final output is a DataFrame with the same columns as the input but with mean values for each feature.
        """
        means = {}
        
        for column in df.columns:
            if column.startswith('hom'):
                # Regular mean for integer columns
                means[column] = df[column].mean()
            
            elif column.startswith('hot'):
                # Mean for one-hot encoded columns
                values = df[column].apply(lambda x: np.where(x == 1.0)[0][0] if 1.0 in x else None).dropna()
                array_length = len(df[column].iloc[0])
                result_array = [0] * array_length
                
                if len(values) > 0:
                    mode = values.mode()
                    if len(mode) > 1:
                        selected_index = np.random.choice(mode)
                    else:
                        selected_index = mode[0]
                    
                    result_array[selected_index] = 1
                means[column] = result_array
            
            elif column.startswith('mult'):
                # Mean for multi-hot encoded columns
                summed = np.sum(df[column].tolist(), axis=0)
                row_count = len(df)
                avg = np.round(summed / row_count).astype(int)

                    # Modify logic to use 0.5 as the threshold for setting the value to 1
                avg = (summed / row_count >= 0.5).astype(int)

                    # If all values in the average are 0
                if np.sum(avg) == 0:
                    non_zero_positions = np.where(summed > 0)[0]
                    if len(non_zero_positions) > 0:
                        random_indices = np.random.choice(non_zero_positions, size=1, replace=False)
                        avg[random_indices] = 1
                           
                means[column] = avg
        
        # Convert the means dictionary into a DataFrame
        means_df = pd.DataFrame({col: [val] if isinstance(val, (int, float)) else [val] for col, val in means.items()})
        
        return means_df
    
    def get_total_distance(self, distance_matrix, groups):
        """
        Calculate the total pairwise distance for a set of groups based on a given distance matrix.

        This method computes the total distance by summing the pairwise distances between all points within each group.
        The pairwise distances for each group are derived from the provided distance matrix. The total distance is then 
        accumulated across all groups.

        Args:
            distance_matrix (pandas.DataFrame or numpy.ndarray): A matrix containing pairwise distances between points.
            groups (list of list): A 2D list where each sublist represents a group, containing indices of points assigned to that group.

        Returns:
            float: The total pairwise distance for all the groups combined, representing the sum of distances within each group.

        Notes:
            - The function assumes that `groups` is a list of lists where each inner list contains indices of points belonging to the same group.
            - The distance matrix is expected to be indexed with the same point indices.
        """
        total_distance = 0

        # Iterate over each group in the 2D list of indices
        for group in groups:
            # For each group, calculate the sum of pairwise distances
            group_distance = 0
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Add the distance between every pair of elements in the group
                    group_distance += distance_matrix[group[i], group[j]]
            
            # Add the group distance to the total distance
            total_distance += group_distance

        return total_distance
    



