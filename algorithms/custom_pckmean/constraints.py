
from algorithms.custom_pckmean.exceptions import InconsistentConstraintsException



# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
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

    return ml_graph, cl_graph, neighborhoods
