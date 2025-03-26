import streamlit as st
import pandas as pd
import ast
from ast import literal_eval
import numpy as np

from gui.algorithm_runner import run_algorithms, add_groups, process_df 

st.set_page_config(page_title="Grouping Assistant")
st.title("Grouping Assistant")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # string lists to np array
    data = data.apply(lambda col: col.map(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('[') else x))

    st.session_state["data"] = data
    st.write("CSV file successfully loaded!")

    expand_df = st.expander("Show File")
    expand_df.write(data)

    st.sidebar.title("Configurations")

    def find_divisors(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    row_count = len(st.session_state['data'])
    divisors = find_divisors(row_count)
    st.sidebar.subheader("Group Size")
    st.session_state['group_size'] = st.sidebar.selectbox('Select Group Size', divisors)

    st.sidebar.subheader("Must-Links")
    must_links = st.sidebar.text_input("Enter set of must-links, e.g., {(1, 2), (0, 10)}")

    st.sidebar.subheader("Cannot-Links")
    cannot_links = st.sidebar.text_input("Enter set of cannot-links, e.g., {(1, 2), (0, 10)}")

    st.sidebar.subheader("Weights for Features")
    weights = {column: st.sidebar.number_input(f'Weight for {column}', min_value=0.0, max_value=None, value=1.0, step=0.1, key=column)
               for column in st.session_state['data'].columns[1:]}

    if st.sidebar.button("Start Grouping"):
        st.session_state['must-links'] = ast.literal_eval(must_links) if must_links else None
        st.session_state['cannot-links'] = ast.literal_eval(cannot_links) if cannot_links else None

        plots_df, grouping_aco, grouping_pck, grouping_ranking = run_algorithms(
            df=st.session_state["data"],
            group_size=st.session_state['group_size'],
            must_links=st.session_state['must-links'],
            cannot_links=st.session_state['cannot-links'],
            weights=weights
        )

        df_aco, df_pck, df_ranking = process_df(st.session_state["data"], grouping_aco, grouping_pck, grouping_ranking)

        compare = st.expander("Compare Results")
        with compare:
            st.image("algorithms_comparison/comparison_plots/time_3_comparison.png", caption="Example Plot", use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Result ACO", "Result CPCK", "Result OCCR"])
        tab1.write(df_aco)
        tab2.write(df_pck)
        tab3.write(df_ranking)
