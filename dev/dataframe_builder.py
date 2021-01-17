""""DataFrame builder"""

import pandas as pd

class DataFrameBuilder:
    
    HEADER="https://bc-dataton2020.s3.amazonaws.com/dataton_all_data/header.txt"
    NUMERIC_COLUMNS=[
        "ingreso_segurida_social",
        "mora_max"
    ]
    STRING_COLUMNS=[]
    BOOLEAN_COLUMNS=[]
    DROP_COLUMNS=[
        "fecha_nacimiento",
        "profesion",
        "ocupacion"
    ]
    
    def __init__(self, dataframe, keep_original=False):
        self.original_dataframe = self._assign_columns(dataframe.copy())
        self.cleaned_dataframe = None
        self.keep_original = keep_original
        
    def _assign_columns(self, dataframe):
        column_names = pd.read_csv(DataFrameBuilder.HEADER).columns.to_list()
        dataframe.columns = column_names
        return dataframe
    
    # Reemplazar \N por NA
    def cleanNA(self, dataframe):
        
        for column in dataframe.columns:
            if column in DataFrameBuilder.NUMERIC_COLUMNS :
                dataframe[column] = dataframe[column].replace("\\N", np.nan).astype('float')
        return dataframe
    
    # Manejo de datos faltantes
    def handle_missing_data(self, dataframe):
        pass
    
    # Modificacion de columnas existentes
    def process_columns(self, dataframe):
        
        # Procesamiento columnas demograficas
        # Procesamiento columnas financieras
        # Procesamiento columnas de riesgo
        dataframe['rep_calif_cred'] = np.where(
                                        dataframe['rep_calif_cred'] == "SIN INFO","NA",
                                        dataframe['rep_calif_cred']
        )
        
        dataframe['mora_max'] = np.where(
                                   dataframe['mora_max'] < 30, "Entre 0 y 30 dias",
                                   np.where(
                                       dataframe['mora_max'] < 60, "Entre 31 y 60 dias",
                                       np.where(
                                           dataframe['mora_max'] > 60, "Mas de 60", "NA")
                                   )
        )
        return dataframe
    
    # Para eliminar las columnas que no vamos a usar
    def remove_columns(self, dataframe):
        
        return dataframe.drop(DataFrameBuilder.DROP_COLUMNS, axis=1)
    
    # Para creacion de columnas nuevas
    def create_columns(self):
        pass
    
    # Guardar Dataframe
    def save_dataframe(self):
        pass
    
    # En esta funcion va todo el flujo
    def build(self, to_s3=False):
        
        # Borrar variables
        sliced_dataframe = self.remove_columns(self.original_dataframe)
        # Missing values
        na_cleaned_dataframe = self.cleanNA(sliced_dataframe)
        # Procesamiento de columnas
        cleaned_dataframe = self.process_columns(na_cleaned_dataframe)
        self.cleaned_dataframe = cleaned_dataframe
        
        # Eliminar de memoria el dataframe original
        if not self.keep_original:
            self.original_dataframe = None
        
        # Guardado (En local o S3)
        
        return self.cleaned_dataframe
        