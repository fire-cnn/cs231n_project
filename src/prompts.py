""" Prompt creation from tabular data
"""

import numpy as np
from tqdm import tqdm


def prompting(
    df,
    prompt_type,
    column_name_map,
    columns_of_interest,
    id_var,
    final_prompt,
    label_column,
    special_tokens=None,
    round_dec=3,
    add_response=True,
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
        - final_prompt str: Final prompt for the text model. Usually the question
          we want to answer.
        - template str: A text template.
        - special_tokens tuple: A tuple with start, end, and padding tokens
        - label_column str: Column in df storing the label.
        - columns_of_interest list: a list with the subset of columns to
          use from the data.
        - round_dec int: Rounding decimals to this level
        - cols_template list: List with the columns in the order of the
          template placeholders.
        - add_response: Should the prompt include the response? This is useful
          to do datasets for predition vs. training.

    Returns:
        Dict of strings
    """

    if special_tokens is None:
        start, end, pad = ("<startoftext>", "<endoftext>", "<pad>")
    elif special_tokens is False:
        start, end, pad = ("", "", "")

    if prompt_type == "template" and template is None:
        raise ValueError("Template exepcted!")

    if template is not None and cols_template is None:
        raise ValueError("You need to pass column list for the prompt")

    # Subset and round dataframe
    df_filtered = df.filter(columns_of_interest)

    # Hacky, but change labels so they can be informative for our purposes
    df_filtered = df_filtered.replace(
        {label_column: {"Yes": "burned", "No": "unburned"}}
    )

    for col_name in df_filtered:
        if df_filtered[col_name].dtype == np.float32:
            df_filtered[col_name] = df_filtered[col_name].astype(float)
        elif df_filtered[col_name].dtype == np.float64:
            df_filtered[col_name] = df_filtered[col_name].astype(float)

    df_filtered = df_filtered.round(round_dec)

    # Create prompt
    if prompt_type == "bank":
        dict_rows = {}
        for idx, row in tqdm(
            df_filtered.iterrows(),
            total=df_filtered.shape[0],
            desc=f"Transforming tabular to {prompt_type} text prompts",
        ):
            row_string = []
            for col in row.keys():
                if col == id_var or col == label_column:
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
            if add_response:
                row_str = f"{start}{row_str}\n {final_prompt} {row[label_column]}{end}"
            else:
                row_str = f"{start}{row_str}\n {final_prompt}{end}"

            dict_rows[row[id_var]] = row_str

    elif prompt_type == "template":
        dict_rows = {}
        for idx, row in tqdm(
            df_filtered.iterrows(),
            total=df_filtered.shape[0],
            desc=f"Transforming tabular to {prompt_type} text prompts",
        ):
            row_subset = row[cols_template]
            template = template.rstrip()
            s = template.format(*row_subset)
            if add_response:
                s = f"{start}{s}. {final_prompt} {row[label_column]}{end}"
            else:
                # This is a hack, the prompt needs and starting space character to fit.
                s = f"{start}{s}{end}"

            dict_rows[row[id_var]] = s

    return dict_rows
