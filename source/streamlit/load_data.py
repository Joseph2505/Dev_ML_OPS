import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, header=0, sep=',')
    new_column_names = [col.split(" (")[0] for col in data.columns[:-1]] + ['target']
    data.columns = new_column_names
    for col in new_column_names[:-1]:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

    return data