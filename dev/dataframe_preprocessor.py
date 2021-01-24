""""DataFrame Preprocessor"""
import pandas as pd
import numpy as np

from scipy.special import boxcox1p, inv_boxcox1p
from scipy import stats
from scipy.stats import norm, skew
class DataFramePreProcessor:
    
    def __init__(self, dataframe, filter_threshold=50000, test=False):
        self.test = test
        self.original_dataframe = dataframe.copy()
        self.modeling_dataframe = None
        self.filter_threshold = filter_threshold
    
    
    def handleMissingData(self, dataframe):
        dataframe['ingreso_final'] = dataframe['ingreso_final'].fillna(0)
        return dataframe
    
    
    def columnFilter(self, dataframe):
        to_drop = [
            "Unnamed: 0",
            "id_cli",
            "pol_centr_ext",
            "ult_actual",
            "cant_mora_30_tdc_ult_3m_sf",
            "cant_mora_30_consum_ult_3m_sf",
            "ind",
            "cat_ingreso",
            "departamento_residencia",
            "ocupacion",
            "estado_civil",
            "tipo_vivienda",
            "nivel_academico",
            "rep_calif_cred",
            "tiene_ctas_activas",
            "mora_max",
            "ind_mora_vigente",
            "genero",
            "convenio_lib"
            
        ]
        
        dataframe = dataframe.drop(to_drop,axis=1, errors='ignore')
        return dataframe
    
    
    # Borrar filas deacuerdo a cierta logica de negocio
    def rowFilter(self, dataframe):
        cut = 0.99
        return dataframe[
                (dataframe['gasto_familiar'] > self.filter_threshold) &
                (dataframe['gasto_familiar'] < np.quantile(dataframe['gasto_familiar'], cut)) &
                (dataframe['ingreso_final'] < np.quantile(dataframe['ingreso_final'], cut)) &
                (dataframe['cupo_total_tc'] < np.quantile(dataframe['cupo_total_tc'], cut)) & # Percentil 99%
                (dataframe['cuota_tc_bancolombia'] < np.quantile(dataframe['cuota_tc_bancolombia'], cut)) & # percentil 99.99%
                (dataframe['cuota_de_vivienda'] < np.quantile(dataframe['cuota_de_vivienda'], cut)) &# Percentil 99.99%
                (dataframe['cuota_de_consumo'] < np.quantile(dataframe['cuota_de_consumo'], cut)) & # percentil 99%
                (dataframe['cuota_rotativos'] < np.quantile(dataframe['cuota_rotativos'], cut))& # percentil 99.99%
                (dataframe['cuota_tarjeta_de_credito'] < np.quantile(dataframe['cuota_tarjeta_de_credito'], cut)) & 
                (dataframe['cuota_de_sector_solidario'] < np.quantile(dataframe['cuota_de_sector_solidario'], cut)) &
                (dataframe['cuota_sector_real_comercio'] < np.quantile(dataframe['cuota_sector_real_comercio'], cut)) &# Percentil 99.5%
                (dataframe['cuota_libranza_sf'] < np.quantile(dataframe['cuota_libranza_sf'], cut)) & # Percentil 99
                (dataframe['ingreso_segurida_social'] < np.quantile(dataframe['ingreso_segurida_social'], cut)) & # percentil 99.9
                (dataframe['ingreso_nomina'] < np.quantile(dataframe['ingreso_nomina'], cut)) &
                (dataframe['saldo_prom3_tdc_mdo'] < np.quantile(dataframe['saldo_prom3_tdc_mdo'], cut)) &
                (dataframe['saldo_no_rot_mdo'] < np.quantile(dataframe['saldo_no_rot_mdo'], cut)) &
                (dataframe['cuota_cred_hipot'] < np.quantile(dataframe['cuota_cred_hipot'], cut)) &
                (dataframe['mediana_nom3'] < np.quantile(dataframe['mediana_nom3'], cut)) &
                (dataframe['mediana_pen3'] < np.quantile(dataframe['mediana_pen3'], cut)) &
                (dataframe['cuota_tc_mdo'] < np.quantile(dataframe['cuota_tc_mdo'], cut)) &
                (dataframe['ingreso_nompen'] < np.quantile(dataframe['ingreso_nompen'], cut)) &
                (dataframe['cant_oblig_tot_sf'] <= 15) &
                (dataframe['ctas_activas'] < 5) &
                (dataframe['nro_tot_cuentas'] < 5) 
            ] 
    
    def oneEncodeVariables(self):
        pass
    
    def processVars(self, dataframe):
        
        # Transformacion demograficas
        dataframe['edad'] = np.where(dataframe['edad'] < 18, 18,
                                    np.where(dataframe['edad'] > 80, 80, dataframe['edad']
                                    ))

        # Transformacion finacieras
        dataframe['ingreso_calculado'] =  dataframe['ingreso_segurida_social']  +  \
                                          dataframe[['ingreso_nompen', 'ingreso_nomina']].max(axis=1)
        dataframe['ingreso_corr'] = dataframe[['ingreso_final', 'ingreso_calculado']].max(axis=1)
        
        dataframe['cuota_cred_hipot'] = dataframe[['cuota_cred_hipot', 'cuota_de_vivienda']].max(axis=1)
        
        dataframe['cuota_de_consumo'] =np.where(dataframe['cuota_de_consumo'] < 0, 0, dataframe['cuota_de_consumo'])
        dataframe['cuota_cred_hipot'] =np.where(dataframe['cuota_cred_hipot'] < 0, 0, dataframe['cuota_cred_hipot'])                                        
                
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
        
        for var in pct_vars:
            dataframe[f"{var}_pct"] = dataframe[var] / dataframe['ingreso_corr']  * 100
            dataframe[f"{var}_pct"] = dataframe[f"{var}_pct"].replace(dict.fromkeys([np.nan, np.inf], 0))
    
        dataframe['obl_total_pct'] = dataframe['cuota_cred_hipot_pct'] + \
                                     dataframe['cuota_tarjeta_de_credito_pct'] +\
                                     dataframe['cuota_de_consumo_pct'] + \
                                     dataframe['cuota_rotativos_pct'] + \
                                     dataframe['cuota_sector_real_comercio_pct'] + \
                                     dataframe['cuota_de_sector_solidario_pct'] + \
                                     dataframe['cuota_sector_real_comercio_pct'] 
        
                                    
        dataframe['total_cuota'] = dataframe['cuota_cred_hipot'] + \
                                   dataframe['cuota_tarjeta_de_credito'] + \
                                   dataframe['cuota_de_consumo'] + \
                                   dataframe['cuota_rotativos'] + \
                                   dataframe['cuota_de_sector_solidario'] + \
                                   dataframe['cuota_libranza_sf'] + \
                                   dataframe['cuota_tarjeta_de_credito'] + \
                                   dataframe['cuota_tc_bancolombia']
        
        dataframe['total_cupo'] = dataframe['cupo_total_tc'] + dataframe['cupo_tc_mdo']
        
        # Variables de interaccion
        dataframe['interact_ing_gen']  = dataframe['genero'] * dataframe['ingreso_corr']
        dataframe['interact_ing_ed']  = dataframe['edad'] * dataframe['ingreso_corr']

        # variables al cuadrado        
        dataframe['total_cuota_2']  =dataframe['total_cuota']**2
 
        

        dataframe['cuota_pct_cupo'] = (dataframe['cuota_tarjeta_de_credito'] + dataframe['cuota_tc_bancolombia']) / dataframe['total_cupo']

        
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.fillna(0, inplace=True)

        
        if not self.test:
            dataframe['log_gasto_familiar'] = np.log1p(dataframe['gasto_familiar']) 
            numeric_feats = dataframe.drop(['gasto_familiar',
                                            'log_gasto_familiar'], axis=1).dtypes[
                    (dataframe.dtypes == "float64")].index
        else:
            numeric_feats = dataframe.dtypes[
                    (dataframe.dtypes == "float64")].index
            
        skewed_feats = dataframe[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness = skewness[abs(skewness) > 0.75]       
        skewed_features = skewness.index
        lam = 0.15
        PT_transformer = PowerTransformer()
        for feat in skewed_features:
             dataframe[feat] = boxcox1p(dataframe[feat], lam)        
            #dataframe[feat] = PT_transformer.fit_transform(dataframe[feat].values.reshape(-1,1))
            
            
        cat_vars = []
        if cat_vars:
            dummified = []
            for var in cat_vars:            
                dummified.append(
                    pd.get_dummies(dataframe[var], drop_first=True, prefix=var)
                )

            dummified = pd.concat(dummified, axis=1)
            dataframe = pd.concat([dataframe.drop(cat_vars, axis=1),dummified], axis=1)
            
        ## Final cleaning
        
        dataframe.drop(["ingreso_final", "ingreso_calculado",
                        "cuota_de_vivienda", "gasto_familiar",
                        #"cupo_total_tc",
                        "cupo_total_tc_pct", "cupo_tc_mdo_pct", "cupo_disponible",
                        #"cupo_tc_mdo",
                        #"cupo_total_tc",
                        "ingreso_nompen", "ingreso_nomina", "ingreso_segurida_social",
                        "tenencia_tc", # Alta correlacion con cuota tarjeta de credito
                        "ctas_activas", # Alta correlacion con nto_tot_cuentas
                        "cuota_cred_hipot_pct", #altta correlacion con cuota credito hipotecario
                        "cuota_de_consumo_pct", #Alta correlacion con cuta de consumo
                        "cuota_rotaticos_pct", # Alta correlacion con cuota de rotativos
                        "cuota_tarjeta_de_credito_pct",
                        "cuota_de_sector_solidario_pct",
                        "cuota_sector_real_comercio_pct",
                        "cuota_libranza_sf_pct",
                        "cuota_tc_bancolombia_pct",
                        "cuota_rotativos_pct",
                        # REVISAR BIEN
                        "saldo_no_rot_mdo",
                        "total_cuota",
                        "total_cupo",
                        "convenio_lib",
                        "ind_mora_vigente",
                        #'cuota_cred_hipot',
                        #'cuota_tarjeta_de_credito',
                        #'cuota_de_consumo',
                        #'cuota_rotativos',
                        #'cuota_sector_real_comercio',
                        #'cuota_de_sector_solidario',
                        #'cuota_tc_bancolombia',
                        #'cuota_libranza_sf',
                        "tiene_consumo",
                        #"cuota_tc_mdo",
                        "tiene_crediagil",
                        "nivel_academico",
                        "total_cupo",
                        "nro_tot_cuentas",
                        "mediana_nom3",
                        "mediana_pen3",
                        "cant_oblig_tot_sf",
                        "saldo_prom3_tdc_mdo",
                        "cuota_pct_cupo"
                       ], axis=1, inplace=True,  errors='ignore')

        return dataframe
    
    def process(self):
        
        complete_df = self.columnFilter(
                self.handleMissingData(self.original_dataframe)
        )
        if not self.test:
            filtered_df = self.rowFilter(complete_df)
            grown_df    = self.processVars(filtered_df)
        else:
            grown_df    = self.processVars(complete_df)
        self.modeling_dataframe = grown_df
        
        return self.modeling_dataframe
        