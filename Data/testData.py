from faker import Faker
import pandas as pd
import numpy as np
import random
import math

# Create faker instance
fake = Faker()
fake.seed_instance(42)
random.seed(42)
np.random.seed(42)
# Create df with 30 random names
df_names = [fake.name() for _ in range(36)]
df_names = pd.DataFrame(df_names, columns=['Name'])

# Current function!
def generate_dataframe(n_rows, feature_list, random_state = False):
    """
    Generate a DataFrame with n_rows rows and len(feature_list) columns based on the feature_list input.

    Args:
        n_rows (int): Number of rows for the DataFrame.
        feature_list (list): List of tuples (x, y), where x is a feature type and y is an integer.

    Returns:
        pd.DataFrame: Generated DataFrame with specified columns and values.
    """
    if random_state:
        random.seed(42)
        np.random.seed(42)
    # Initialize an empty dictionary to store column data
    data = {}

    # Track counts for enumerating each feature category
    feature_counts = {}

    for feature, y in feature_list:
        # Increment the feature count to ensure proper enumeration
        if feature not in feature_counts:
            feature_counts[feature] = 1
        else:
            feature_counts[feature] += 1

        # Create the column name with enumeration
        col_name = f"{feature}_{feature_counts[feature]}"

        if feature in ["hom_", "het_ninja", "het_hot"]:
            # Random integers from 1 to y
            data[col_name] = np.random.randint(1, y + 1, size=n_rows)

        elif feature in ["hot_hom", "hot_het"]:
            # Arrays of length y with all zeros and one 1 at a random position
            data[col_name] = [np.eye(1, y, k=np.random.randint(0, y)).flatten() for _ in range(n_rows)]

        elif feature in ["mult_hot_het", "mult_hot_hom"]:
            # Arrays of length y with random 0s and 1s
            data[col_name] = [np.random.choice([0, 1], size=y).tolist() for _ in range(n_rows)]

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    return df


def create_random_cl_ml(n_rows, n_links = 0.1):
    """
    Generates two lists of tuples based on a percentage of random integers.

    Parameters:
        n_rows (int): The range of integers (0 to n_rows).
        n_links (float): The percentage of integers to select, as a float between 0 and 1.

    Returns:
        list[tuple[int, int]], list[tuple[int, int]]: Two lists of tuples (x, y) where x != y.
    """
    # Calculate the number of random integers to select
    num_random_integers = math.ceil(n_links * n_rows * 2)  # Multiply by 2 and round up to next even number
    if num_random_integers % 4 != 0:
        num_random_integers += 4 - (num_random_integers % 4)

    # Select random integers without repetition
    random_integers = random.sample(range(n_rows), num_random_integers)

    # Create tuples (x, y) from the selected integers
    tuples = [(random_integers[i], random_integers[i + 1]) for i in range(0, len(random_integers), 2)]

    # Split the tuples into two lists
    mid_index = len(tuples) // 2
    ml_list = tuples[:mid_index]
    cl_list = tuples[mid_index:]
    return [ml_list, cl_list]
    


# IMPORTENT DO NOT DELET!
tuple_list = [("hom_", 3), ("hom_", 4), ("hot_het", 3),("hot_het", 4), ("mult_hot_hom", 4), ("mult_hot_het", 4)]

df_A1 = generate_dataframe(n_rows= 60, feature_list = tuple_list)

df_A2 = generate_dataframe(n_rows= 180, feature_list = tuple_list)

df_A3 = generate_dataframe(n_rows= 420, feature_list = tuple_list)

df_A4 = generate_dataframe(n_rows= 900, feature_list = tuple_list)

# Generated data frames for evaluation:

df_12 = generate_dataframe(n_rows= 12, feature_list = tuple_list)
lst_12_ml_cl = create_random_cl_ml(n_rows=12)

df_36 = generate_dataframe(n_rows= 36, feature_list = tuple_list)
lst_36_ml_cl = create_random_cl_ml(n_rows=36)

df_60 = generate_dataframe(n_rows= 60, feature_list = tuple_list)
lst_60_ml_cl = create_random_cl_ml(n_rows=60)

df_84 = generate_dataframe(n_rows= 84, feature_list = tuple_list)
lst_84_ml_cl = create_random_cl_ml(n_rows=84)

df_96 = generate_dataframe(n_rows= 96, feature_list = tuple_list)
lst_96_ml_cl = create_random_cl_ml(n_rows=96)

df_216 = generate_dataframe(n_rows= 216, feature_list = tuple_list)
lst_216_ml_cl = create_random_cl_ml(n_rows=216)

# List of all data frames to consider during evaluation
df_ev_list = [df_12, df_36, df_60, df_84, df_96, df_216]

# For each df we have a set of ml and cl (they need to be at the same index as their df in df_ev_lst)
lst_ev_ml_cl =[lst_12_ml_cl, lst_36_ml_cl, lst_60_ml_cl, lst_84_ml_cl, lst_96_ml_cl, lst_216_ml_cl]

