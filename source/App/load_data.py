import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, header=0, sep=',')
    new_column_names = [col.split(" (")[0] for col in data.columns[:-1]] + ['target']
    data.columns = new_column_names
    data = convert_columns_to_numeric(data, data.columns[:-1])
    return data

def convert_columns_to_numeric(data, columns):
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('int64')
    return data