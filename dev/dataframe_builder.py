""""DataFrame builder"""

import pandas as pd

class DataFrameBuilder:
    
    HEADER="https://bc-dataton2020.s3.amazonaws.com/dataton_all_data/header.txt"
    
    def __init__(self, dataframe):
        self.original_dataframe = self._assign_columns(dataframe)
        
    def _assign_columns(self, dataframe):
        column_names = pd.read_csv(DataFrameBuilder.HEADER).columns.to_list()
        dataframe.columns = column_names
        return dataframe
        
    def handle_missing_data(self):
        pass
    
    def process_columns(self):
        pass
    
    def remove_columns(self):
        pass
    
    def create_columns(self):
        pass
    
    def build(self):
        pass
        