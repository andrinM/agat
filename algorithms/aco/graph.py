import numpy as np
import pandas as pd 

from distance_matrix.distance_matrix import get_distance_matrix

class Graph:
    """
    Represents a graph where nodes are group members, and edges represent distances between them.
    The graph uses a pheromone matrix to track the desirability of paths between nodes.

    """
    def __init__(self,df, weights = {}):
        """
        Initializes the Graph object with nodes, distances, and pheromones.

        Args:
            df (pd.DataFrame): A DataFrame containing the data for nodes (group members).
            weights (dict, optional): A dictionary specifying weights for computing distances.
                                      Default is an empty dictionary.

        Parameters:
            df (pd.DataFrame): A copy of the input DataFrame.
            num_nodes (int): Number of nodes in the graph.
            distance_matrix (np.ndarray): Matrix of distances computed using the `get_distance_matrix` function.
            pheromones (np.ndarray): Matrix initialized with equal pheromone values (1.0) for all paths.
        """
        self.df = df.copy()
        self.num_nodes = self.df.shape[0]  # Number of nodes (group members)
        self.distance_matrix = get_distance_matrix(df1 = self.df, df2 = self.df, weights = weights)
        # Initialize pheromones for each path between nodes (same size as distances)
        self.pheromones = np.ones((self.num_nodes, self.num_nodes))# Start with equal pheromones






