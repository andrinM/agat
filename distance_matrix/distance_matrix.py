import pandas as pd
import numpy as np  
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial

#TODO 
# skalieren s.d features gleich gewichtet werden
# anpassen, s.d perfect match immer 0 gibt 
# distanzen new: hot_het, hot_hom, het_ninja, het_hot
# add function, such that each feature can be weighted 
# add distance for multiple one_hot 
def get_distance_matrix(df1, df2, weights = {}, scale = True):
    """
    Calculate the total distance matrix for a dataframe based on various metrics.
    Scale = False, means we skip the minmax scaling for the hom features (needed for pck)

    Parameters:
    df1 and df2 (pd.DataFrame): The DataFrame can include different types of features:
    - Homogeneous features (numerical). Columns should start with 'hom_'.
    - Heterogeneous features (numerical). 
        Columns should start with 'het_ninja' (distance for perfect match != 0).
        Columns should start with 'het_hot' (distance for perfect match = 0).
    - Categorical homogeneous features (one-hot encoded, binary). Columns should start with 'hot_hom'.
    - Categorical heterogeneous features (one-hot encoded, binary). Columns should start with 'hot_het.
    - Categorical homogeneous features with multiple choice options (one-hot encoded, binary). 
        Columns should start with 'mult_hot_hom'.
    - Categorical heterogeneous features with multiple choice options (one-hot encoded, binary). 
        Columns should start with 'mult_hot_het'.
    weights (dict, optional): A dictionary where keys are column names and values are ints to weight specified feature.
        Default weights are 1 if not specified.
    max_deviations (dict, optional): A dictionary where keys are column names and values are max_deviations 
    for features with ninja distance.  
    Returns:
    np.array: A matrix representing the summed distances across different feature sets.
    """

    df1 = df1.copy()
    df2 = df2.copy()
    if scale: # Scale hom feautres
        homogeneous_distance = np.zeros((df1.shape[0], df2.shape[0]))
        for hom in df1.filter(regex="^hom").columns:
            # Get the global minimum and maximum for the column of both dataframes
            global_min = min(df1[hom].min(), df2[hom].min())
            global_max = max(df1[hom].max(), df2[hom].max())

            # Perform MinMax scaling for the column in both DataFrames
            df1[hom] = (df1[hom] - global_min) / (global_max - global_min)
            df2[hom] = (df2[hom] - global_min) / (global_max - global_min)
        
            weight = weights.get(hom,1)
            homogeneous_distance += get_homogeneous_distance(df1[hom], df2[hom], weight)
    else: # hom feautres are already scaled
        homogeneous_distance = np.zeros((df1.shape[0], df2.shape[0]))
        for hom in df1.filter(regex="^hom").columns:

            weight = weights.get(hom,1)
            homogeneous_distance += get_homogeneous_distance(df1[hom], df2[hom], weight)


    # Integer based distance
    heterogeneous_distance = np.zeros((df1.shape[0], df2.shape[0]))
    for het_hot in df1.filter(regex="^het_hot").columns:
        weight = weights.get(het_hot,1)
        heterogeneous_distance += get_heterogeneous_hot_distance(df1[het_hot],df2[het_hot],weight)
    np.fill_diagonal(heterogeneous_distance, 0)

    one_hot_het_distance = np.zeros((df1.shape[0], df2.shape[0]))
    for hot_het in df1.filter(regex="^hot_het").columns:
        weight = weights.get(hot_het,1)
        one_hot_het_distance += get_heterogeneous_hot_distance(df1[hot_het],df2[hot_het],weight)
    np.fill_diagonal(one_hot_het_distance, 0)

    one_hot_hom_distance = np.zeros((df1.shape[0], df2.shape[0]))
    for hot_hom in df1.filter(regex="^hot_hom").columns:
        weight = weights.get(hot_hom,1)
        one_hot_hom_distance += get_one_hot_hom_distance(df1[hot_hom],df2[hot_hom],weight)

    multiple_one_hot_hom_distance = np.zeros((df1.shape[0], df2.shape[0]))
    for mult_hot_hom in df1.filter(regex = "^mult_hot_hom").columns: 
        weight = weights.get(mult_hot_hom,1)
        multiple_one_hot_hom_distance += get_multiple_one_hot_hom_distance(df1[mult_hot_hom], df2[mult_hot_hom],weight)

    multiple_one_hot_het_distance = np.zeros((df1.shape[0], df2.shape[0]))
    for mult_hot_het in df1.filter(regex = "^mult_hot_het").columns:
        weight = weights.get(mult_hot_het,1) 
        multiple_one_hot_het_distance += get_multiple_one_hot_het_distance(df1[mult_hot_het],df2[mult_hot_het],weight)
    np.fill_diagonal(multiple_one_hot_het_distance ,0)        
        
    distance_matrix = (
        homogeneous_distance + heterogeneous_distance 
        + one_hot_het_distance + one_hot_hom_distance + multiple_one_hot_hom_distance 
        + multiple_one_hot_het_distance)

    return distance_matrix * (1 / df1.shape[1])

def get_homogeneous_distance(feature1, feature2, weight):
    """
    Calculate the homogeneous distance for a single feature using absolute difference.

    Parameters:
    feature1 (pd.Series): A series from a dataframe 1 to be processed.
    feature2 (pd.Series): A series from a dataframe 2 to be processed.
    Returns:
    np.array: A matrix of pairwise distances.
    """
    metric = partial(abs_difference, weight = weight)
    feature1_reshaped = feature1.to_numpy().reshape(-1, 1)
    feature2_reshaped = feature2.to_numpy().reshape(-1, 1)
    return pairwise_distances(feature1_reshaped, feature2_reshaped, metric=metric)

def abs_difference(x, y, weight):
    return (abs(x-y)) * weight

def get_heterogeneous_hot_distance(feature1, feature2, weight):
    """
    Calculate pairwise distances for a feature using a hot-het metric.

    Parameters:
    feature1 (pd.Series): A series from a dataframe 1 to be processed.
    feature2 (pd.Series): A series from a dataframe 2 to be processed.

    Returns:
    np.array: A matrix of pairwise distances.
    """
    metric = partial(hot_het_metric, weight = weight)
    if isinstance(feature1.iloc[0], (list, np.ndarray)) and isinstance(feature2.iloc[0], (list, np.ndarray)) :
        # Reshaping for one_hot encoded features
        feature1_reshaped = np.stack(feature1.values)
        feature2_reshaped = np.stack(feature2.values)
    else:
        # Reshaping for numerical features
        feature1_reshaped = feature1.to_numpy().reshape(-1, 1)
        feature2_reshaped = feature2.to_numpy().reshape(-1, 1)
    return pairwise_distances(feature1_reshaped, feature2_reshaped, metric = metric)


def hot_het_metric(x, y, weight):
    """
    Parameters:
    x, y (np.array): One-hot encoded vectors to compare.

    Returns:
    int: 1 if vectors are identical, 0 otherwise.
    """
    return int(np.array_equal(x, y))* weight    


def get_one_hot_hom_distance(feature1, feature2, weight):
    """
    Calculate pairwise distances for a one-hot encoded feature using hot-hom metric.

    Parameters:
    feature1 (pd.Series): A series of one-hot encoded lists.
    feature2 (pd.Series): A series of one-hot encoded lists.

    Returns:
    np.array: A matrix of pairwise distances.
    """
    metric = partial(hot_hom_metric, weight = weight)
    feature1_reshaped = np.stack(feature1.values)
    feature2_reshaped = np.stack(feature2.values)
    return pairwise_distances(feature1_reshaped, feature2_reshaped,  metric = metric)

def hot_hom_metric(x, y, weight):
    """
    Parameters:
    x, y (np.array): One-hot encoded vectors to compare.

    Returns:
    int: 1 if different, 0 if identical.
    """
    return int(not np.array_equal(x, y))* weight

def get_multiple_one_hot_hom_distance(feature1, feature2, weight):
    """
    Calculate pairwise distances for one-hot encoded feature with multiple choice using mult_hot_hom_metric.

    Parameters:
    feature1 (pd.Series or np.ndarray): A series or an array of one-hot encoded vectors.
    feature2 (pd.Series or np.ndarray): A series or an array of one-hot encoded vectors.

    Returns:
    np.array: A matrix of pairwise distances.
    """
    metric = partial(mult_hot_hom_metric, weight = weight)
    feature1_reshaped = np.stack(feature1.to_numpy())
    feature2_reshaped = np.stack(feature2.to_numpy())
    return pairwise_distances(feature1_reshaped, feature2_reshaped, metric = metric)

def mult_hot_hom_metric(x, y, weight):
    """
    Counts the differences at each positions within the vectors. 

    Parameters:
    x, y (np.array): One-hot encoded vectors to compare.

    Returns:
    float: The squared, normalized count of non-matching positions between the vectors. 
    """
    return (np.sum(x != y) / len(x)) * weight

def get_multiple_one_hot_het_distance(feature1, feature2, weight):
    """
    Calculate pairwise distances for a series of one-hot encoded vectors using a custom metric.
    This function focuses on heterogeneity by evaluating matches between vectors.

    Parameters:
    feature1 (pd.Series or np.ndarray): A series or an array of one-hot encoded vectors.
    feature2 (pd.Series or np.ndarray): A series or an array of one-hot encoded vectors.

    Returns:
    np.array: A matrix of pairwise distances calculated using the custom heterogeneity metric.
    """
    metric = partial(mult_hot_het_metric, weight = weight)
    feature1_reshaped = np.stack(feature1.to_numpy())
    feature2_reshaped = np.stack(feature2.to_numpy())
    return pairwise_distances(feature1_reshaped, feature2_reshaped, metric = metric)

def mult_hot_het_metric(x, y, weight):
    """
    Counts the similarity at each positions within the vectors. 

    Parameters:
    x, y (np.array): One-hot encoded vectors to compare.

    Returns:
    float: The squared, normalized count of matching positions between the vectors.
    """
    return ((np.sum(x == y) / len(x))) * weight


