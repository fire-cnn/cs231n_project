""" Prompt creation from tabular data
"""

import pandas as pd
import numpy as np


def prompting(
    df,
    prompt_type,
    column_name_map,
    columns_of_interest,
    id_var,
    template=None,
    cols_template=None,
):
    """Create text prompts from tabular dataset

    This function create text prompts for each row of a tabular dataset.
    There are two options for prompting: 1. "bank", which is an itemized
    list of variables and values, and 2. "template", which is uses a very
    descriptive text template (defined by the user) to put all the wanted
    variables.

    For the template, the should be a format-able string:
    This observation has an x of {}. Y is {}. Z has the following value {}.
    and `cols_template` should have the order of the string.

    Args:
        - df: DataFrame tabular data
        - prompt_type str: a prompt style (bank or template)
        - column_name_map: a dict with a mapping between the column names
        and the desired names in text.
        - id_var str: column in data to build identifier in dictionary
        - columns_of_interest list: a list with the subset of columns to
        use from the data.
        - cols_template list: List with the columns in the order of the
        template placeholders.

    Returns:
        Dict of strings
    """

    if prompt_type == "template" and template is None:
        raise ValueError(f"Template exepcted!")

    if template is not None and cols_template is None:
        raise ValueError(f"You need to pass column list for the prompt")

    # Subset and round dataframe
    df_filtered = df.filter(columns_of_interest)

    for col_name in df_filtered:
        if df_filtered[col_name].dtype == np.float32:
            df_filtered[col_name] = df_filtered[col_name].astype(float)
        elif df_filtered[col_name].dtype == np.float64:
            df_filtered[col_name] = df_filtered[col_name].astype(float)

    df_filtered = df_filtered.round(2)

    # Create prompt
    if prompt_type == "bank":
        dict_rows = {}
        for idx, row in df_filtered.iterrows():
            row_string = []
            for col in row.keys():
                if col == id_var:
                    pass
                else:
                    if isinstance(row[col], float):
                        val = round(row[col], 2)
                    else:
                        val = row[col]

                    colname = column_name_map[col]

                    s = f" - {colname}: {val}"
                    row_string.append(s)

            row_str = "\n".join(row_string)
            dict_rows[row[id_var]] = row_str

    elif prompt_type == "template":
        dict_rows = {}
        for idx, row in df_filtered.iterrows():
            row_subset = row[cols_template]
            s = template.format(*row_subset)
            dict_rows[row[id_var]] = s

    return dict_rows
