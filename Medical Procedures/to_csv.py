import pandas as pd
import numpy as np
import re


def merge_nan_rows(df: pd.DataFrame):
    out_df = df.copy()
    i_last = None
    for i_row, row in out_df.iterrows():
        if pd.isna(i_row):
            for i_col, col in row.iteritems():
                if isinstance(out_df.loc[i_last][i_col], str):
                    out_df.loc[i_last][i_col] += (' ' + col)
        else:
            i_last = i_row

    out_df = out_df[out_df.index.notnull()]
    out_df.index = out_df.index.astype(int)

    return out_df


data = pd.DataFrame(pd.read_excel('data/procedures.xlsx', usecols="B:S", header=5, index_col="Index"))
data = merge_nan_rows(data)

data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def str_to_float_col(input_str: str):
    if type(input_str) != str:
        return input_str

    if input_str == '':
        return np.nan
    else:
        try:
            return float(input_str.replace(',', '.'))
        except ValueError:
            # print(f'{input_str}')
            return np.inf


for col in ['D-dimer', 'Urea', 'Creatinine', 'CRP']:
    data[col] = data[col].apply(str_to_float_col, convert_dtype=float)

data['Ultrasound of veins a/o'] = data['Ultrasound of veins a/o'].apply(
    lambda x: 'Не выявлено' if not pd.isna(x) and re.search("не\s+выявлено|NaN", x, re.IGNORECASE) is not None else x)

data.to_csv("data/procedures.csv", index_label='Index')

# %%
