import numpy as np
import itertools
import importlib


class Ant:
    """
    Represents an individual 'ant'  (solution) in an Ant Colony Optimization (ACO) algorithm. 
    The ant explores the graph, forms groups, and constructs solutions based on pheromone levels 
    and distance constraints such as must-links.

    """
    def __init__(self, graph, group_size, alpha = 1, beta = 1, ml = {}, cl = {}):
        """
        Initializes the Ant object, setting the graph, group size and must-links.

        Parameters:
            graph (Graph): The graph representing the problem.
            group_size (int): The size of each group the ant will form.
            alpha (int): weights pheromone level. Default value is 1 
            beta (int): weights local information. Default value is 1 
            ml (list of tuples, optional): Constraints specifying pairs of nodes that must be grouped together.
            cl (list of tuples, optional): Constraints specifying pairs of nodes that can not be grouped together.
        """
        self.graph = graph
        self.group_size = group_size
        self.alpha = alpha
        self.beta = beta
        self.group_num = graph.num_nodes // group_size
        self.ml, self.cl = self.preprocess_constraints(ml, cl, graph.num_nodes)
        self.current_node = None
        # Initialize a path to store nodes grouped into "group_num" groups, each with "group_size" members 
        self.path = self.set_path()
        self.distance = 0   # Start with zero distance traveled
        self.group_distance = np.zeros(self.group_num)
        self.unvisited_nodes = self.set_unvisited_nodes()  # keep track of unvisited nodes 
        self.reachable_nodes = self.unvisited_nodes.copy()
        self.solution_score = np.inf # Set solution_score to inf, to be minimized over iterations

    # Adapted from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
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
                    raise InconsistentConstraintsException('Inconsistent constraints between {} and {}'.format(i, j))

        # Calculate not_neighbours for cl_graph
        visited = [False] * n
        not_neighbours = []
        for i in range(n):
            if not visited[i] and cl_graph[i]:
                component = []
                dfs(i, cl_graph, visited, component)
                not_neighbours.append(component)

        return neighborhoods, not_neighbours
   
    def set_path(self):
        """
        Initializes the path as a list of groups, each containing `group_size` elements.

        Returns:
            list: 
            If must-links are provided,the groups are initialized with the must-linked members,
            otherwise, the groups are initialized as empty lists

        """
        path = [[None] * self.group_size for i in range(self.group_num)]
        if len(self.ml) != 0: 
            for i, member in enumerate(self.ml):
                path[i][:len(member)] = member

        return path 

    def set_unvisited_nodes(self):
        """
        Initializes the set of unvisited nodes by removing those that are part of must-links.

        Returns:
            set: The set of nodes yet to be visited, excluding must-linked nodes.

        """
        unvisited_nodes = set(range(self.graph.num_nodes))
        if len(self.ml) != 0: 
            linked_nodes = {node for link in self.ml for node in link}
            unvisited_nodes -= linked_nodes    
        return unvisited_nodes              

   
    def group(self):
        """
        Forms groups of nodes by assigning each group the appropriate members based on pheromone levels,
        local information and must-link constraints.

        Returns:
            list: The path representing the grouped nodes.
        """ 
        for group in range(self.group_num):
            if all(value is not None for value in self.path[group]): 
                continue
            if self.path[group][0] is None:
                # if cannot-links available start grouping with them
                if self.cl and self.cl[0] and self.cl[0][0] in self.unvisited_nodes:
                    first_sublist = self.cl[0]
                    current_node = first_sublist[0]
                    self.current_node = current_node
                    self.cl[0].remove(current_node)
                    self.unvisited_nodes.remove(self.current_node)
                    self.set_reachable_nodes(first_sublist)
                    self.cl = [e for e in self.cl if e]
                # Randomly choose first group member          
                else: 
                    self.current_node = int((np.random.choice(list(self.unvisited_nodes))))
                    self.unvisited_nodes.remove(self.current_node)
                    self.reachable_nodes.remove(self.current_node)
                self.path[group][0] = self.current_node
            else: 
                self.current_node = self.path[group][self.path[group].index(None) - 1]     
            while None in self.path[group]:
                next_node = self.select_next_node(self.path[group])
                index = self.path[group].index(None)
                self.path[group][index] = next_node
                self.current_node = next_node
            self.reset_reachable_nodes()
        self.calculate_distance()        
        self.distance = self.solution_score = self.group_distance.sum()        
        return self.path
    
    # Select next node based on pheromone levels and distance
    def select_next_node(self, current_group):
        """
        Selects the next node to visit based on pheromone levels and distance to the current node.

        Parameters:
            current_group (list): The current group being formed.

        Returns:
            int: The index of the next node to visit.
        """

        if len(self.unvisited_nodes) == 1: 
            return self.unvisited_nodes.pop()
        else:
            # Array to store selection probabilities 
            probabilities = np.zeros(self.graph.num_nodes)
            for node in self.reachable_nodes:
                loc_information = self.get_loc_information(current_group, node)
                if loc_information == 0: 
                    loc_information = 1e-10 # if 0 divide by small number
                probabilities[node] = self.graph.pheromones[self.current_node][node]**self.alpha / loc_information**self.beta
                assert probabilities[node] != np.inf     
            #if probabilities.sum() != 0:
            assert probabilities.sum() != 0                                   
            probabilities /= probabilities.sum()  # Normalize the probabilities to sum to 1
            next_node = int(np.random.choice(range(self.graph.num_nodes), p=probabilities))
            self.unvisited_nodes.remove(next_node)
            self.reachable_nodes.remove(next_node)
            return next_node 

    def set_reachable_nodes(self, sublist):
        self.reachable_nodes = self.unvisited_nodes.copy() 
        self.reachable_nodes -= set(sublist)      

   
    def reset_reachable_nodes(self):
        self.reachable_nodes = self.unvisited_nodes.copy()        

         
    def get_loc_information(self,current_group,node):
        """
        Calculates the location information for a node with respect to the current group.

        Args:
            current_group (list): The current group of nodes.
            node (int): The node whose local information is to be calculated.

        Returns:
            float: The calculated local information based on distances to other group members.
        """ 
        loc_information = 0
        for member in current_group:
            if member is not None:
                loc_information += self.graph.distance_matrix[member][node]                                    
        return loc_information                        

    def calculate_distance(self):
        """
        Calculates the total distance of the solution by summing distances between all pairs of members within each group.

        """ 
        for i, group in enumerate(self.path): 
            combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
            for x,y in combinations:
                self.group_distance[i] += self.graph.distance_matrix[x][y]