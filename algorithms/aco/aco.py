import numpy as np
import itertools

from algorithms.aco.ant import Ant

class ACO:
    """
    ACO (Ant Colony Optimization) algorithm to find the optimal grouping solution for the given graph.
    
    The algorithm runs for a specified number of iterations, where specified number of ants explore 
    the graph, form groups, and update pheromones based on the quality of the solutions they find.
    The best solution is tracked and returned at the end of the iterations.

    """
    def __init__(self, graph, num_ants, num_iterations, group_size, ml = {}, cl ={}, decay=0.5, alpha=1.0, beta = 1.0):
        """
        Initializes the ACO algorithm with the given parameters.
        
        Parameters:
            graph (Graph): The graph on which the ACO algorithm will run.
            num_ants (int): The number of ants (solutions) in each iteration.
            num_iterations (int): The number of iterations to run the algorithm.
            group_size (int): The size of each group to be formed.
            ml (list, optional): Pairs of nodes that must be grouped together. Default is empty set.
            cl (list, optional): Pairs of nodes that cannot be grouped together. Default is empty set.
            decay (float, optional): The rate at which pheromones evaporate. Default is 0.5.
            alpha (float, optional): The strength of pheromone influence. Default is 1.0.
            beta (float, optional): The strength of local information influence. Default is 1.0.
        """
        self.graph = graph
        self.df = self.graph.df
        self.num_ants = num_ants  # Number of ants in each iteration
        self.num_iterations = num_iterations  # Number of iteration
        self.group_size = group_size
        self.ml = ml
        self.cl = cl 
        self.decay = decay  # Rate at which pheromones evaporate
        self.alpha = alpha  # Weight for pheromone level
        self.beta = beta  # Wight for local information
        self.best_solution_history = [] # Track best solution found in each iteration
        self.best_path_history = [] # Track best path in each iteration
       
   
    
    def run(self):
        """
        Runs the ACO algorithm for the specified number of iterations.
        
        In each iteration, ants explore the graph, form groups, and update pheromones based on the 
        quality of their solutions. The best solution and path are tracked and returned after all iterations.
        
        Returns:
           tuple: A tuple containing the best path and the best solution found by the algorithm.

        """
        best_path = None
        best_solution = np.inf 
        # Run the algorithm for the specified number of iterations
        for _ in range(self.num_iterations):
            ants = [Ant(self.graph, self.group_size, self.alpha, self.beta, self.ml, self.cl) for _ in range(self.num_ants)]  # Create a group of ants
            for ant in ants:
                ant.group()  # Let each ant form groups
                if best_solution > ant.solution_score:
                    best_solution = ant.solution_score
                    best_path = ant.path    
            self.update_pheromones(ants)  # Update pheromones based on the ants' paths
            # Visualisierung der Pheromone als Heatmap
            #self.plot_pheromone_heatmap(self.graph.pheromones, _)
            self.best_solution_history.append(best_solution)  # Save the best solution for each iteration
            self.best_path_history.append(best_path)
        return best_path, best_solution
  
    def update_pheromones(self, ants):
        """
        Updates the pheromones on the paths after all ants have completed their tours.
        
        Pheromones are evaporated first, and then they are reinforced based on the quality of the ants' 
        solutions. Better solutions result in higher pheromone levels.
        
        Parameters:
            ants (list of Ant): The list of ants that completed their paths.
        """
        self.graph.pheromones *= self.decay  # Reduce pheromones on all paths (evaporation)
        # For each ant, increase pheromones on the paths they took, based on how good their path was
        for ant in ants:
            for group in ant.path:
                combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
                for x, y in combinations:
                    if ant.solution_score == 0: 
                        ant.solution_score = 1e-10
                    self.graph.pheromones[x, y] += 1 / ant.solution_score 
                    self.graph.pheromones[y, x] += 1 / ant.solution_score                   
 
    # Assign groups from best solution to DataFrame after algorithm runs 
    def add_groups(self, path):
        """
        Assigns group labels to the members in the provided DataFrame based on the best path found.
        
        This function adds a new column 'Group' to the DataFrame, where each member is assigned 
        to a group based on the best path solution.
        
        Args:
            path (list of lists): The best path (solution) found by the algorithm, representing 
                                   groups of members.
            df (pandas.DataFrame): The DataFrame that contains the members to be grouped.
        
        Returns:
            pandas.DataFrame: The updated DataFrame with group assignments.
        """

        self.df['Group'] = None
        # Iterate through each group and update the 'Group' column in the Dat3aFrame
        for group_number, group in enumerate(path):
            for index in group:
                self.df.at[index, 'Group'] = group_number
        return self.df      
          