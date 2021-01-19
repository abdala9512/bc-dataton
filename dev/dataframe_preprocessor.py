""""DataFrame Preprocessor"""
import pandas as pd
import numpy as np

class DataFramePreProcessor:
    
    def __init__(self, dataframe, test=False):
        self.test = test
        self.original_dataframe = dataframe.copy()
        self.modeling_dataframe = None
    
    
    def handleMissingData(self, dataframe):
        dataframe['ingreso_final'] = dataframe['ingreso_final'].fillna(0)
        dataframe['ind'] = dataframe['ind'].fillna(0)
        return dataframe
    
    # Borrar filas deacuerdo a cierta logica de negocio
    def rowFilter(self, dataframe):
        return dataframe[
            (dataframe['edad'] < 80) &
            (dataframe['gasto_familiar'] >= 0) &
            (dataframe['gasto_familiar'] < 10000000) &
            (dataframe['cant_oblig_tot_sf'] < 13) &
            (dataframe['cuota_de_consumo'] >= 0)
 
        ] 
    
    def oneEncodeVariables(self):
        pass
    
    def newVars(self, dataframe):
        pct_vars = [
            'cuota_cred_hipot',
            'cuota_tarjeta_de_credito',
            'cuota_de_consumo',
            'cuota_rotativos',
            'cuota_sector_real_comercio',
            'cuota_de_sector_solidario',
            'cuota_tc_bancolombia',
            'cuota_libranza_sf'
        ]
        
        cat_vars = [
            'mora_max'
        ]
        
        dummified = []
        for var in cat_vars:
            dummified.append(
                pd.get_dummies(dataframe[var], drop_first=True, prefix=var)
            )
        
        dummified = pd.concat(dummified, axis=1)
        dataframe = pd.concat([dataframe.drop(cat_vars, axis=1),dummified], axis=1)
        for var in pct_vars:
            dataframe[f"{var}_pct"] = dataframe[var] / dataframe['ingreso_final'] 
            dataframe[f"{var}_pct"] = dataframe[f"{var}_pct"].replace(dict.fromkeys([np.nan, np.inf], 0))
            
        dataframe['genero'] = np.where(dataframe['genero'] == 'M', 0, 1)
        dataframe['ind_mora_vigente'] = np.where(dataframe['ind_mora_vigente'] == "S", 1, 0)
        dataframe['convenio_lib'] = np.where(dataframe['convenio_lib'] == 'S', 1, 0)
        dataframe['ingreso_calculado'] =  dataframe['ingreso_segurida_social']  + dataframe['ingreso_nompen'] 
        dataframe['ind_annio'] = dataframe['periodo'].apply(lambda x: 1 if str(x)[:4] == '2020' else 0)
        if not self.test:
            dataframe['log_gasto_familiar'] = np.log(dataframe['gasto_familiar'] +1 ) 
        return dataframe
    
    def process(self):
        
        complete_df = self.handleMissingData(self.original_dataframe)
        if not self.test:
            filtered_df = self.rowFilter(complete_df)
            grown_df    = self.newVars(filtered_df)
        else:
            grown_df    = self.newVars(complete_df)
        self.modeling_dataframe = grown_df
        
        return self.modeling_dataframe
        