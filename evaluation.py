import pandas as pd 
import os

from algorithms_comparison.process import run_algorithms
from Data.testData import df_12
from Data.testData import df_A1
from Data.testData import df_ev_list, lst_ev_ml_cl


group_sizes = {2,3,4}
results = []
avg_results = []
selected_df = 0 #Indicate the position of the desired df: df_ev_list = [df_12, df_36, df_60, df_84, df_96, df_216]. Important for the lst_ev_ml_cl

# If you want to run a single n_members config use True, for a whole run use False
single_run = True

# This loop only works, if we run all togheter
if single_run == False:
    if __name__ == '__main__':
        for i, df in enumerate(df_ev_list[:4]):
            num = len(df) # For naming csv

            # lst_ev_ml_cl is a list of lists. Each n_member has two inner list, one for ml (index 0) and for cl (iondex 1).
            df_res, df_avg = run_algorithms(df = df,
                                            group_sizes = group_sizes,
                                            must_links=lst_ev_ml_cl[i][0],
                                            cannot_links=lst_ev_ml_cl[i][1])
            results.append(df_res)
            avg_results.append(df_avg)

        df_results = pd.concat(results, ignore_index = True)
        df_avg_results = pd.concat(avg_results, ignore_index = True)

        df_results = df_results.sort_values(by=['n_members','group_size', 'algorithm'])
        df_avg_results = df_avg_results.sort_values(by=['n_members','group_size', 'algorithm'])


        directory = "algorithms_comparison/csv_files"

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f'results_{num}.csv')
        df_results.to_csv(file_path, index=False)
        file_path = os.path.join(directory, f'avg_results_{num}.csv')
        df_avg_results.to_csv(file_path, index=False)

if single_run == True:
    if __name__ == '__main__':
        df = df_ev_list[selected_df]
        num = len(df) # For naming csv

        # lst_ev_ml_cl is a list of lists. Each n_member has two inner list, one for ml (index 0) and for cl (iondex 1).
        df_res, df_avg = run_algorithms(df = df,
                                        group_sizes = group_sizes,
                                        must_links=lst_ev_ml_cl[i][0],
                                        cannot_links=lst_ev_ml_cl[i][1])
        
        results.append(df_res)
        avg_results.append(df_avg)

        df_results = pd.concat(results, ignore_index = True)
        df_avg_results = pd.concat(avg_results, ignore_index = True)

        df_results = df_results.sort_values(by=['n_members','group_size', 'algorithm'])
        df_avg_results = df_avg_results.sort_values(by=['n_members','group_size', 'algorithm'])


        directory = "algorithms_comparison/csv_files'"

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f'2results_{num}.csv')
        df_results.to_csv(file_path, index=False)
        file_path = os.path.join(directory, f'2avg_results_{num}.csv')
        df_avg_results.to_csv(file_path, index=False)