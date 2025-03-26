import pandas as pd 
import numpy as np 
import time
import itertools 
import os 
import sys
import importlib



from distance_matrix.distance_matrix import get_distance_matrix

from algorithms_comparison import optimized_distance

# import algorithms
import algorithms.aco.aco as aco
import algorithms.aco.graph as graph
import algorithms.aco.ant as ant
from algorithms.randomness import occurance_ranking
from algorithms.custom_pckmean import custom_pckmeans

# Relode Modules every time, to account for changes in the algorithms
importlib.reload(occurance_ranking)
importlib.reload(custom_pckmeans)
importlib.reload(aco)
importlib.reload(ant)
importlib.reload(graph)

Graph = graph.Graph
Ant = ant.Ant
ACO = aco.ACO

def calculate_intra_group_distance(grouping, distance_matrix):
    """
    Calculates the total distance within each group by summing distances between all pairs of members.

    """
    keys = [] 
    group_distances = np.zeros(len(grouping))
       
    for i, group in enumerate(grouping): 
        combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
        for x,y in combinations:
            group_distances[i] += distance_matrix[x][y]

    return sorted(group_distances), group_distances        

def calculate_gini_index(grouping, distance_matrix):
    group_distances_sorted, group_distances_unsorted= calculate_intra_group_distance(grouping, distance_matrix)
    num_groups = len(group_distances_sorted)
    cum_prop = np.zeros(num_groups)
    for i in range(num_groups):
        cum_prop[i] = sum(group_distances_sorted[:i+1])
    cum_prop = cum_prop / sum(group_distances_sorted)  
    sum_cum_prop = sum(cum_prop) - 1/2 
    l_curve = (num_groups - 2*sum_cum_prop)/(2*num_groups)
    max_cons = (num_groups - 1)/(2*num_groups)
    gini = l_curve / max_cons 
    return gini, group_distances_unsorted



def run_aco(df, group_size, ml, cl):
    if ml is None:
        ml ={}
    if cl is None:
        cl = {}    
    start_time = time.perf_counter()
    graph = Graph(df)
    aco = ACO(graph, num_ants=30, num_iterations=30, group_size=group_size, beta = 5, ml = ml, cl = cl)
    best_path, best_distance = aco.run()
    run_time = time.perf_counter() - start_time
    return run_time, best_distance, best_path 

def run_custom_pck(df, group_size, must_links, cannot_links):
    start_time = time.perf_counter()
    model_cpck = custom_pckmeans.custom_PCKMeans(df=df,
                                                group_size=group_size,
                                                n_clusters=None, # Default value
                                                max_iter=150,
                                                ml_list=must_links,
                                                cl_list=cannot_links,
                                                lp_time_limit = 360)
    
    best_distance, best_grouping = model_cpck.fit()

    run_time = time.perf_counter() - start_time
    return run_time, best_distance, best_grouping

def run_occurance_ranking(df, group_size, must_links, cannot_links):
    start_time = time.perf_counter()
    model_occ = occurance_ranking.Occurance_Ranking(df=df,
                                                    group_size=group_size,
                                                    alpha=0.1,
                                                    max_iter=10000,
                                                    n_solutions=1,
                                                    ml_list=must_links,
                                                    cl_list=cannot_links)
    best_distance, best_grouping = model_occ.fit()

    run_time = time.perf_counter() - start_time
    return run_time, best_distance, best_grouping


def run_algorithms(df, group_sizes, must_links = None, cannot_links = None, weights = {}):
    """
    Computes the total pairwise distances for all possible groups of a given size 
    from a distance matrix, while adhering to cannot-link constraints.

    Args:
        df: A single data frame 
        group_sizes (list): A list of all possible group sizes
        cl_graph (dict): A dictionary representing cannot-link constraints. Keys 
                        are indices, and values are sets of indices that cannot 
                        be grouped together with the key.
        must_links: List of tuples (x,y) with must-links
        cannot_links: List of tuples (x,y) with cannot-links
        weights: dictionary with weights for each feautre

    Returns:
        results_df: data frame with 'algorithm', 'n_members','group_size', 'time', 'distance','groups', 'group_distances', 'gini-index'
        for each iteration of comibnation of (n_members, group_size)
        avg_results_df: The same as results_df but with average values
    """
    distance_matrix = get_distance_matrix(df1 = df, df2 = df, weights = weights)
    results = []
    avg_results = []
    iterations = 10

    for group_size in group_sizes:
        time_dist_aco = {'run_times': [], 'distances': [], 'gini-index': []}
        time_dist_pck = {'run_times': [], 'distances': [], 'gini-index': []}
        time_dist_ranking = {'run_times': [], 'distances': [], 'gini-index': []}

        # Ignore must-links if the group_size is 2

        # Temporarily set must_links to None only for group_size == 2
        current_must_links = must_links if group_size != 2 else None
        

        for i in range(iterations):
            run_time_aco, distance_aco, grouping_aco = run_aco(df = df,
                                                        group_size = group_size,
                                                        ml = current_must_links,
                                                        cl = cannot_links)
            print(f"ACO: {i} out of {iterations}, group size: {group_size}")

            run_time_pck, distance_pck, grouping_pck = run_custom_pck(df = df,
                                                        group_size = group_size,
                                                        must_links = current_must_links,
                                                        cannot_links = cannot_links)
            print(f"PCK: {i} out of {iterations}, group size: {group_size}")

            run_time_ranking, distance_ranking, grouping_ranking = run_occurance_ranking(df = df,
                                                                     group_size = group_size,
                                                                     must_links = current_must_links,
                                                                     cannot_links = cannot_links)
            print(f"OCC: {i} out of {iterations}, group size: {group_size}")
            
            gini_aco, group_distances_aco = calculate_gini_index(grouping_aco, distance_matrix)
            gini_pck, group_distances_pck = calculate_gini_index(grouping_pck, distance_matrix)
            gini_ranking, group_distances_ranking = calculate_gini_index(grouping_ranking, distance_matrix)

            time_dist_aco['run_times'].append(run_time_aco)
            time_dist_aco['distances'].append(distance_aco)
            time_dist_aco['gini-index'].append(gini_aco)

            time_dist_pck['run_times'].append(run_time_pck)
            time_dist_pck['distances'].append(distance_pck)
            time_dist_pck['gini-index'].append(gini_pck)

            time_dist_ranking['run_times'].append(run_time_ranking)
            time_dist_ranking['distances'].append(distance_ranking)
            time_dist_ranking['gini-index'].append(gini_ranking)
           
            results.extend([
                {'algorithm': 'ACO', 'n_members': len(df),
                 'group_size': group_size, 'time': run_time_aco,
                 'distance': distance_aco,
                 'groups': grouping_aco, 'group_distances': group_distances_aco,
                 'gini-index': gini_aco},
                {'algorithm': 'PCK', 'n_members': len(df),
                 'group_size': group_size, 'time': run_time_pck,
                 'distance': distance_pck, 
                 'groups': grouping_pck, 'group_distances': group_distances_pck,
                 'gini-index': gini_pck },
                {'algorithm': 'Ranking', 'n_members': len(df),
                 'group_size': group_size, 'time': run_time_ranking, 
                 'distance': distance_ranking,
                 'groups': grouping_ranking, 'group_distances': group_distances_ranking,
                 'gini-index': gini_ranking}
            ])
        

        avg_run_time_aco = sum(time_dist_aco['run_times']) / iterations
        avg_run_time_pck = sum(time_dist_pck['run_times']) / iterations
        avg_run_time_ranking = sum(time_dist_ranking['run_times']) / iterations

        avg_distance_aco = sum(time_dist_aco['distances']) / iterations
        avg_distance_pck = sum(time_dist_pck['distances']) / iterations
        avg_distance_ranking = sum(time_dist_ranking['distances']) / iterations

        avg_gini_aco = sum(time_dist_aco['gini-index']) / iterations
        avg_gini_pck = sum(time_dist_pck['gini-index']) / iterations
        avg_gini_ranking = sum(time_dist_ranking['gini-index']) / iterations
        print("Start lower bound")
        lower_bound = optimized_distance.compute_infimum_distance(distance_matrix, group_size)
        print("Start lower end")
        avg_results.extend([
                {'algorithm': 'ACO', 'n_members': len(df),
                'group_size': group_size, 'average_time': avg_run_time_aco,
                'average_distance': avg_distance_aco, 'average_gini-index': avg_gini_aco,
                'lower_bound_distance': lower_bound},
                {'algorithm': 'PCK', 'n_members': len(df),
                'group_size': group_size, 'average_time': avg_run_time_pck,
                'average_distance': avg_distance_pck, 'average_gini-index': avg_gini_pck,
                'lower_bound_distance': lower_bound},
                {'algorithm': 'Ranking', 'n_members': len(df),
                'group_size': group_size, 'average_time': avg_run_time_ranking,
                'average_distance': avg_distance_ranking, 'average_gini-index': avg_gini_ranking,
                'lower_bound_distance': lower_bound}
            ])

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['group_size', 'algorithm'])

    avg_results_df = pd.DataFrame(avg_results)
    avg_results_df = avg_results_df.sort_values(by=['group_size', 'algorithm'])
  
    return results_df, avg_results_df