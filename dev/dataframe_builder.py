""""DataFrame builder"""

import pandas as pd
import numpy as np

class DataFrameBuilder:
    
    HEADER="https://bc-dataton2020.s3.amazonaws.com/dataton_all_data/header.txt"
    NUMERIC_COLUMNS=[
        "edad",
        "ingreso_segurida_social",
        "mora_max",
        "ingreso_nomina",
        "ind",
        "ingreso_final",
        "cuota_cred_hipot",
        "saldo_prom3_tdc_mdo"
    ]
    DROP_COLUMNS=[
        "fecha_nacimiento",
        "profesion",
        "ocupacion",
        "codigo_ciiu",
        "ciudad_residencia",
        "ciudad_laboral",
        "departamento_laboral",
        "nivel_academico",
        "tipo_vivienda",
        "categoria",
        "rechazo_credito",
        "cartera_castigada",
        "cant_moras_30_ult_12_meses",
        "cant_moras_60_ult_12_meses",
        "cant_moras_90_ult_12_meses",
        "ctas_embargadas",
        "tiene_ctas_embargadas",
        "pension_fopep",
        "tiene_cred_hipo_1",
        "tiene_cred_hipo_2",
        "cant_cast_ult_12m_sr",
        "tenencia_tc",
        "tiene_consumo",
        "tiene_crediagil",
        "pol_centr_ext",
        "tiene_ctas_activas"
    ]
    
    def __init__(self, dataframe, keep_original=False, test=False):
        self.test=test
        self.original_dataframe = self._assign_columns(dataframe.copy())
        self.cleaned_dataframe = None
        self.keep_original = keep_original

        
    def _assign_columns(self, dataframe):
        column_names = pd.read_csv(DataFrameBuilder.HEADER).columns.to_list()
        if self.test:
            column_names.remove("gasto_familiar")
            column_names.insert(0, "id_registro")
        dataframe.columns = column_names
        return dataframe
    
    # Manejo de datos faltantes
    # Reemplazar \N por NA
    def cleanNA(self, dataframe):
        
        for column in dataframe.columns:
            if column in DataFrameBuilder.NUMERIC_COLUMNS :
                dataframe[column] = dataframe[column].replace("\\N", np.nan).astype('float')
        return dataframe
    
    # Modificacion de columnas existentes
    def process_columns(self, dataframe):
        
        # Procesamiento columnas demograficas
        dataframe['edad'] =  dataframe['edad'].round().fillna(method='ffill').astype('int') 
        dataframe['departamento_residencia'] = dataframe['departamento_residencia'].str.strip()
        dataframe['estado_civil'] = np.where(
                    dataframe['estado_civil'] == "SOLTERO", "SOL",
                    np.where(
                        dataframe['estado_civil'] == "CASADO", "CAS",
                        np.where(
                            dataframe['estado_civil'] == "UNION LIBRE", "UL",
                                np.where(
                                    dataframe['estado_civil'] == "NO INFORMA", "NI",
                                        np.where(
                                            dataframe['estado_civil'] == "DIVORCIADO", "DIV",
                                            np.where(
                                                dataframe['estado_civil'] == "VIUDO", "VIU",
                                                    np.where(
                                                        dataframe['estado_civil'] == "\\N", "NI",
                                                        dataframe['estado_civil']
                                                        )
                                                )
                                            )
                                    )
                            )
                        )
                    )
        ########## Procesamiento columnas financieras
        dataframe['convenio_lib'] = np.where(dataframe['convenio_lib'] == "\\N", "N", "S")
        #dataframe['tiene_consumo'] = np.where(dataframe['tiene_consumo'] == "\\N", "N", "S")
        #dataframe['tenencia_tc'] = np.where(dataframe['tenencia_tc'] == "NO", "N", "S")
        dataframe['cat_ingreso'] = np.where(
                                        dataframe['cat_ingreso'] == "\\N","OTROS",
                                        dataframe['cat_ingreso']
        )
        
        dataframe['cuota_cred_hipot'] = dataframe['cuota_cred_hipot'].fillna(0)
        dataframe['cant_oblig_tot_sf'] = pd.Series(np.where(
                                            dataframe['cant_oblig_tot_sf'] == "\\N", "0",
                                            dataframe['cant_oblig_tot_sf']
        )).astype("int")
        
        dataframe['ingreso_nomina'] = dataframe['ingreso_nomina'].fillna(0)
        dataframe['ingreso_segurida_social'] = dataframe['ingreso_segurida_social'].fillna(0)
        
        dataframe['ctas_activas'] = pd.Series(np.where(dataframe['ctas_activas'] =="\\N", "0",
                                             dataframe['ctas_activas']
                                            )).astype("int")
        dataframe['nro_tot_cuentas'] = pd.Series(np.where(dataframe['nro_tot_cuentas'] =="\\N", "0",
                                             dataframe['nro_tot_cuentas']
                                            )).astype("int")
        ########### Procesamiento columnas de riesgo
        dataframe['ind_mora_vigente'] = np.where(
                                        dataframe['ind_mora_vigente'] == '\\N', "NApl",
                                        dataframe['ind_mora_vigente']
        )
        dataframe['rep_calif_cred'] = np.where(
                                        dataframe['rep_calif_cred'] == "SIN INFO","NApl",
                                        dataframe['rep_calif_cred']
        )
        
        dataframe['mora_max'] = np.where(
                                   dataframe['mora_max'] < 30, "Entre 0 y 30 dias",
                                   np.where(
                                       dataframe['mora_max'] < 60, "Entre 31 y 60 dias",
                                       np.where(
                                           dataframe['mora_max'] > 60, "Mas de 60", "NApl")
                                   )
        )
        
        dataframe['cant_mora_30_tdc_ult_3m_sf'] = np.where(
                                                    dataframe['cant_mora_30_tdc_ult_3m_sf'] == "\\N", "NApl",
                                                    np.where(
                                                        dataframe['cant_mora_30_tdc_ult_3m_sf'] == "0",
                                                            "SIN MORA", "CON MORA")
        )
        
        dataframe['cant_mora_30_consum_ult_3m_sf'] = np.where(
                                                    dataframe['cant_mora_30_consum_ult_3m_sf'] == "\\N", "NApl",
                                                    np.where(
                                                        dataframe['cant_mora_30_consum_ult_3m_sf'] == "0",
                                                            "SIN MORA", "CON MORA")
        )

        return dataframe
    
    # Para eliminar las columnas que no vamos a usar
    def remove_columns(self, dataframe):
        return dataframe.drop(DataFrameBuilder.DROP_COLUMNS, axis=1)
    
    # Para creacion de columnas nuevas
    def create_columns(self):
        # CREACION CUENTAS PASIVO Y CUENTAS ACTIVAS CON BANCOLOMBIA
        # POSIBLE: SUMAR LOS CUPOS DE TC
        
        pass
    
    def filter_rows(self, dataframe):
        # BORRAR CUENTAS ACTIVAS > 10
        # BORRAR OBLIGACIONES 10+
        pass
    
    # Guardar Dataframe
    def save_dataframe(self, dataframe, path):
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
        