import pandas as pd
from io import StringIO
import os
from load_data import load_data, convert_columns_to_numeric

def test_convert_columns_to_numeric():
    test_data = StringIO("""A,B,C\n1,2,3\n4,five,6""")
    df = pd.read_csv(test_data, sep=",")
    
    df_converted = convert_columns_to_numeric(df, ['A', 'B', 'C'])
    
    assert df_converted['B'].dtype == 'int64'
    assert df_converted['B'][1] == 0

def test_load_data():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, '..', '..', 'dataset', 'transfusion.data')
    print("file_path:", file_path)
    print("current_directory:", current_directory)
    df_loaded = load_data(file_path)
    assert 'target' in df_loaded.columns
