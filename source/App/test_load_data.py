import unittest
import pandas as pd
from io import StringIO

from App import load_data as ld

class TestDataFunctions(unittest.TestCase):
    
    def test_convert_columns_to_numeric(self):
        test_data = StringIO("""A,B,C\n1,2,3\n4,five,6""")
        df = pd.read_csv(test_data, sep=",")
        
        df_converted = ld.convert_columns_to_numeric(df, ['A', 'B', 'C'])
        
        self.assertEqual(df_converted['B'].dtype, 'int64')
        self.assertEqual(df_converted['B'][1], 0) 
        
    def test_load_data(self):
        df_loaded = ld.load_data('../../dataset/transfusion.data')
        self.assertIn('target', df_loaded.columns)

if __name__ == '__main__':
    unittest.main() 