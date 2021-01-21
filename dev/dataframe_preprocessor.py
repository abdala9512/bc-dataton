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
            (dataframe['edad'] < 100) & # Imputar a percentil 99
            #(dataframe['gasto_familiar'] >= 0) &
            (dataframe['gasto_familiar'] < 30000000) &
            # (dataframe['cant_oblig_tot_sf'] < 13) &
            (dataframe['cuota_de_consumo'] >= 0) & # Mas bien transformar a cero
            (dataframe['ingreso_final'] < 50000000) &
            (dataframe['cuota_cred_hipot'] >= 0)
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
            'cuota_libranza_sf',
            'cupo_total_tc',
            'cupo_tc_mdo'
        ]
        
        cat_vars = [
            'mora_max', # Tal vez haya que encontrar el umbral correcto
            'estado_civil',
            'rep_calif_cred',
            'ocupacion',
            'tipo_vivienda',
            'ocupacion',
            'nivel_academico',
            'pol_centr_ext'
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
            
        for var in pct_vars:
            dataframe[f"{var}_log"] = np.log(dataframe[var]+1)
         
        dataframe['genero'] = np.where(dataframe['genero'] == 'M', 0, 1)
        dataframe['ind_mora_vigente'] = np.where(dataframe['ind_mora_vigente'] == "S", 1, 0)
        dataframe['convenio_lib'] = np.where(dataframe['convenio_lib'] == 'S', 1, 0)
        dataframe['ingreso_calculado'] =  dataframe['ingreso_segurida_social']  +  \
                                          dataframe[['ingreso_nompen', 'ingreso_nomina']].max(axis=1) 
        dataframe['ind_annio'] = dataframe['periodo'].apply(lambda x: 1 if str(x)[:4] == '2020' else 0)
        dataframe['ind_covid'] = np.where(
            dataframe['periodo'].isin([202004,202003,202005]), 1,0)
        if not self.test:
            dataframe['gasto_familiar'] = np.where(dataframe['gasto_familiar'] < 0, 0, dataframe['gasto_familiar'])
            dataframe['log_gasto_familiar'] = np.log(dataframe['gasto_familiar'] +1 ) 
            
        dataframe['ingreso_final'] = np.log(dataframe['ingreso_final']+1)
        dataframe['cupo_total_tc'] = np.log(dataframe['cupo_total_tc']+1)
        dataframe['ingreso_nomina'] = np.log(dataframe['ingreso_nomina']+1)
        dataframe['ingreso_segurida_social'] = np.log(dataframe['ingreso_segurida_social']+1)
        dataframe['ingreso_nompen'] = np.log(dataframe['ingreso_nompen']+1)
        dataframe['ingreso_calculado'] = np.log(dataframe['ingreso_calculado']+1)
        
         

        dataframe['obl_total_pct'] = dataframe['cuota_cred_hipot_pct'] + \
                                     dataframe['cuota_tarjeta_de_credito_pct'] +\
                                     dataframe['cuota_de_consumo_pct'] + \
                                     dataframe['cuota_rotativos_pct'] + \
                                     dataframe['cuota_sector_real_comercio_pct'] + \
                                     dataframe['cuota_de_sector_solidario_pct'] + \
                                     dataframe['cuota_libranza_sf_pct']
                                
        dataframe['ingreso_corr'] = dataframe[['ingreso_final', 'ingreso_calculado']].max(axis=1)
        dataframe['ingreso_corr_log'] = np.log(dataframe['ingreso_corr']+1)
        
        dataframe['interact_ing_gen']  = dataframe['genero'] * dataframe['ingreso_corr']
        dataframe['interact_ing_ed']  = dataframe['edad'] * dataframe['ingreso_corr']
        dataframe['interact_cup_gen']  = dataframe['genero'] * dataframe['cupo_total_tc']
        dataframe['interact_cup_ed']  = dataframe['edad'] * dataframe['cupo_total_tc']
        
        dataframe['ingreso_cero'] = np.where(dataframe['ingreso_corr'] == 0, 1, 0)
        dataframe['ingreso_geo_alto']  = np.where(dataframe['ingreso_corr'] < 14.90, 1, 0) # ALgo mas tecnico
        dataframe['es_ciudad_principal'] = np.where(
            dataframe['departamento_residencia'].isin(['BOGOTA D.C.', 'ANTIOQUIA', 'VALLE', 'CUNDINAMARCA']), 1,0)
        
        dataframe['pc10'] = np.where(dataframe['ingreso_corr'] <= np.quantile(dataframe['ingreso_corr'],0.1), 1, 0)
        dataframe['pc90'] = np.where(dataframe['ingreso_corr'] >= np.quantile(dataframe['ingreso_corr'],0.9), 1, 0)

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
        