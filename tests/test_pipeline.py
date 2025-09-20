import unittest, os, pandas as pd
from telecom_project_analysis import load_data
class TestPipeline(unittest.TestCase):
    def test_load_data_csv(self):
        df = pd.DataFrame({'a':[1,2,3]})
        df.to_csv('temp_test_telecom.csv', index=False)
        loaded = load_data('temp_test_telecom.csv')
        self.assertEqual(loaded.shape[0], 3)
        os.remove('temp_test_telecom.csv')
if __name__ == "__main__":
    unittest.main()
