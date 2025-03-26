# An Algorithmic Approach to Solve the Grouping Problem 

## Overview 

This repository presents algorithmic solutions to the grouping problem, a combinatorial optimization challenge. The objective is to partition a set of objects into optimal groups while adhering to predefined constraints, such as fixed group sizes and must-link/cannot-link conditions. These constraints enforce mandatory co-membership or exclusion between specific individuals. This work was developed as part of a bachelor's thesis.

## Algorithms 
In this repository, three algorithms are implemented that demonstrates different approaches to tackling the grouping problem:
- Ant Colony Optimization: A probabilistic optimization algorithm.
- Custom PCK-Mean: A customized PCKMeans clustering algorithm
- Occurrence Ranking: Algorithm defined from scratch

## Features 
- Reproducible Evaluations: Scripts and data are provided to replicate the evaluations conducted in the thesis
- Interactive Prototype: A minimal prototype using Streamlit allows users to form groups with their own datasets. 

## Getting Started 

1. Clone Repository:
```python
git clone https://gitlab.inf.unibe.ch/SEG/students/theses/viola-camille-andrin-mueller/agat.git 
```
2. Install Dependencies:
```python
pip install -r requirements.txt
```
3. Build Cython Extensions:
Run the following command to build the Cython extension for occ_optimization:  
```python
cd algorithms/randomness 

python setup.py build_ext --inplace 
```
Run the following command to build the Cython extension for optimized_distances:

```python
cd algorithms_comparison

python setup.py build_ext --inplace 
```
4. Run Streamlit: 
Run the following command to run the Streamlit App:

```python
python main.py home
```

5. Run Evaluation: 

Run the following command to run the evaluation: 

```python
python main.py evaluation 

```

## Evaluation Settings 
The evaluations can be configured in the `evaluation.py` file, by addapting following inputs:

```python
goup_sizes = {2,3,4} #line 10
selected_df = 0 #line 13
single_run = True #line 16
```

- group_sizes: List of group sizes considered in the evaluation.

- single_run: if ```True``` each algorithm is only run once, else 10 iterations for each configurations is run 

- selected_df: selects only ```df_ev_list[selected_df]``` from ```df_ev_list``` if single_run is set to ```True```


## CSV File Format 
The CSV file for Streamlit must have a first column named "Name". After that, features can be individually selected from the table below: 

| FeatureID     | Feature Type                  | Feature Class | # Answers | Example Input |
|---------------|-------------------------------|---------------|-----------|---------------|
| hom           | discrete numerical, interval  | homogeneous   | single    | 1             |
| het           | discrete numerical, interval  | heterogeneous | single    | 1             |
| hot_hom       | discrete categorical, nominal | homogeneous   | single    | [1,0,0]       |
| hot_het       | discrete categorical, nominal | heterogeneous | single    | [1,0,0]       |
| multi_hot_hom | discrete categorical, nominal | homogeneous   | multiple  | [1,1,0]       |
| multi_hot_hom | discrete categorical, nominal | heterogeneous | multiple  | [1,1,0]       |

Important: 
- Column names must match the FeatureID 
- If a feature appears multiple times, it must be named in the format FeatureID_num (e.g., hom_1, hom_2)
- The input for each feature must match the format of the example for that specific feature
- All list inputs must be in integer format (e.g. 1 instead of 1.0)

The directory `example_csv_files` contains two example files that can be uploaded via streamlit. 

## Authors
Viola Meier \
Andrin Müller
