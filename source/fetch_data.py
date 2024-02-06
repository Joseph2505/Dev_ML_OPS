import pandas as pd

def fetch_data(filepath):
    data = pd.read_csv(filepath, header=0, sep=',')
    return data