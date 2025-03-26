import pandas as pd 
import numpy as np 
import time
import itertools 
import importlib


from distance_matrix.distance_matrix import get_distance_matrix

# import algorithms

from algorithms.aco.aco import ACO
from algorithms.aco.ant import Ant 
from algorithms.aco.graph import Graph 
from algorithms.randomness import occurance_ranking
from algorithms.custom_pckmean import custom_pckmeans
from algorithms_comparison.process import run_aco, run_custom_pck, run_occurance_ranking, calculate_gini_index



def run_algorithms(df, group_size, must_links = None, cannot_links = None, weights = {}):
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
    names = df['Name']
    df = df.drop('Name', axis = 1).reset_index(drop=True)
    result_plots = []
    distance_matrix = get_distance_matrix(df1 = df, df2 = df, weights = weights)

    # Temporarily set must_links to None only for group_size == 2
    
    run_time_aco, distance_aco, grouping_aco = run_aco(df = df,
                                                        group_size = group_size,
                                                        ml = must_links,
                                                        cl = cannot_links)
 

    run_time_pck, distance_pck, grouping_pck = run_custom_pck(df = df,
                                                        group_size = group_size,
                                                        must_links = must_links,
                                                        cannot_links = cannot_links)


    run_time_ranking, distance_ranking, grouping_ranking = run_occurance_ranking(df = df,
                                                                     group_size = group_size,
                                                                     must_links = must_links,
                                                                     cannot_links = cannot_links)

            
    gini_aco, group_distances_aco = calculate_gini_index(grouping_aco, distance_matrix)
    gini_pck, group_distances_pck = calculate_gini_index(grouping_pck, distance_matrix)
    gini_ranking, group_distances_ranking = calculate_gini_index(grouping_ranking, distance_matrix)
           
    result_plots.extend([
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

    plots_df = pd.DataFrame(result_plots)
    plots_df = plots_df.sort_values(by=['group_size', 'algorithm'])

  
    return plots_df, grouping_aco, grouping_pck, grouping_ranking


def add_groups(df, grouping):
    df['Group'] = None
        # Iterate through each group and update the 'Group' column in the Dat3aFrame
    for group_number, group in enumerate(grouping):
        for index in group:
            df.at[index, 'Group'] = group_number
    return df 

def process_df(df, grouping_aco, grouping_pck, grouping_ranking):
    # Füge Gruppen zu DataFrame hinzu
    grouped_aco = add_groups(df.copy(), grouping_aco)
    grouped_pck = add_groups(df.copy(), grouping_pck)
    grouped_ranking = add_groups(df.copy(), grouping_ranking)

    # Extrahiere nur die 'Name' und 'Group' Spalten für jede Methode
    group_only_aco = grouped_aco.sort_values(by='Group')
    group_only_pck = grouped_pck.sort_values(by='Group')
    group_only_ranking = grouped_ranking.sort_values(by='Group')

    return group_only_aco, group_only_pck, group_only_ranking
