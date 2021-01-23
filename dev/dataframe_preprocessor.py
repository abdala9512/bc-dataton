""""DataFrame Preprocessor"""
import pandas as pd
import numpy as np

from scipy.special import boxcox1p, inv_boxcox1p
from scipy import stats
from scipy.stats import norm, skew
class DataFramePreProcessor:
    
    def __init__(self, dataframe, test=False):
        self.test = test
        self.original_dataframe = dataframe.copy()
        self.modeling_dataframe = None
    
    
    def handleMissingData(self, dataframe):
        dataframe['ingreso_final'] = dataframe['ingreso_final'].fillna(0)
        dataframe['ind'] = dataframe['ind'].fillna(0)
        dataframe['tipo_vivienda'] = dataframe['tipo_vivienda'].fillna("NO INFORMA")
        dataframe['categoria'] = dataframe['categoria'].fillna("6")
        dataframe['estado_civil'] =dataframe['estado_civil'].fillna("NI")
        #dataframe['departamento_residencia'] = dataframe['departamento_residencia'].fillna("SIN INFORMACION")
        dataframe['ocupacion'] = dataframe['ocupacion'].fillna("Otro")
        dataframe['ind_mora_vigente']  = dataframe['ind_mora_vigente'].fillna("N") 
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
            #ocupacion",
            #estado_civil",
            #tipo_vivienda",
            #nivel_academico",
            #rep_calif_cred",
            "tiene_ctas_activas"
        ]
        
            
        dataframe = dataframe.drop(to_drop,axis=1, errors='ignore')
        return dataframe
    
    
    # Borrar filas deacuerdo a cierta logica de negocio
    def rowFilter(self, dataframe):
        return dataframe[
            #(dataframe['edad'] < 100) & # Imputar a percentil 99
            (dataframe['gasto_familiar'] > 40000) &
            #(dataframe['edad'] >=20) &
            (dataframe['gasto_familiar'] < 30000000) &
            # (dataframe['cant_oblig_tot_sf'] < 13) &
            (dataframe['cuota_de_consumo'] >= 0) & # Mas bien transformar a cero
            (dataframe['ingreso_final'] < 50000000) &
            (dataframe['cuota_cred_hipot'] >= 0) &
            (dataframe['cupo_total_tc'] < 50000000) & # Percentil 99%
            (dataframe['cuota_tc_bancolombia'] < 10000000) & # percentil 99.99%
            (dataframe['cuota_de_vivienda'] < 10000000) &# Percentil 99.99%
            (dataframe['cuota_de_consumo'] < 10000000) & # percentil 99%
            (dataframe['cuota_rotativos'] < 10000000)& # percentil 99.99%
            (dataframe['cuota_tarjeta_de_credito'] < 10000000) & 
            (dataframe['cuota_de_sector_solidario'] < 10000000) &
            (dataframe['cuota_sector_real_comercio'] < 13000000) &# Percentil 99.5%
            (dataframe['cuota_libranza_sf'] < 5000000) & # Percentil 99
            (dataframe['ingreso_segurida_social'] < 25000000) & # percentil 99.9
            (dataframe['ingreso_nomina'] < 20000000) &
            (dataframe['saldo_prom3_tdc_mdo'] < 30000000) &
            (dataframe['saldo_no_rot_mdo'] < 300000000) &
            (dataframe['cuota_cred_hipot'] < 10000000) &
            (dataframe['mediana_nom3'] < 20000000) &
            (dataframe['mediana_pen3'] < 11000000) &
            (dataframe['cuota_tc_mdo'] < 30000000) &
            (dataframe['ingreso_nompen'] < 3000000) &
            (dataframe['cant_oblig_tot_sf'] <= 15) &
            (dataframe['ctas_activas'] < 5) &
            (dataframe['nro_tot_cuentas'] < 5) 
          #  ~(dataframe['departamento_residencia'].isin(['MADRID', 'ESTADO DE LA FLORIDA', 'VAUPES']))
        ] 
    
    def oneEncodeVariables(self):
        pass
    
    def processVars(self, dataframe):
        
        # Transformacion demograficas
        dataframe['genero'] = np.where(dataframe['genero'] == 'M', 0, 1)
        dataframe['edad'] = np.where(dataframe['edad'] < 18, 18,
                                    np.where(dataframe['edad'] > 80, 80, dataframe['edad']
                                    ))
        #dataframe['educacion_grupo'] = np.where(
        #    dataframe['nivel_academico'].isin(['PRIMARIO', 'UNIVERSITARIO', 'ESPECIALIZACION']),1,0
        #)

        dataframe['tipo_vivienda'] =  np.where(dataframe['tipo_vivienda'] == "\\N", "NO INFORMA", dataframe['tipo_vivienda'])
        dataframe['categoria']=np.where(dataframe['categoria']=='\\N', "6",dataframe['categoria'])
        dataframe['categoria']=dataframe['categoria'].astype(float).astype(int)
        dataframe['ocupacion']=np.where(dataframe['ocupacion'].isin(
                ["\\N", "Agricultor", "Ganadero", 'Vacío', "Ama de casa", "Sin Ocupacion Asignada"]), "Otro",
                                       np.where(dataframe['ocupacion'] == "Pensionado", "Jubilado", dataframe['ocupacion']
                                               ))
        
        #dataframe['departamento_residencia'] = \
        #    np.where(
        #        dataframe['departamento_residencia'].isin(['\\N', 'NARIÑO', 'NARI#O', 'VAUPES','MADRID', 'ESTADO DE LA FLORIDA']),
        #        "SIN INFORMACION", dataframe['departamento_residencia'] )
        #dataframe['es_ciudad_principal'] = np.where(
        #    dataframe['departamento_residencia'].isin(['BOGOTA D.C.', 'ANTIOQUIA', 'VALLE', 'CUNDINAMARCA']), 1,0)        
        
        dataframe['ind_mora_vigente'] = np.where(dataframe['ind_mora_vigente'] == "S", 1, 0)
        dataframe['convenio_lib'] = np.where(dataframe['convenio_lib'] == 'S', 1, 0)
        # Transformacion finacieras
        dataframe['ingreso_calculado'] =  dataframe['ingreso_segurida_social']  +  \
                                          dataframe[['ingreso_nompen', 'ingreso_nomina']].max(axis=1)
        dataframe['ingreso_corr'] = dataframe[['ingreso_final', 'ingreso_calculado']].max(axis=1)
        
        dataframe['cuota_cred_hipot'] = dataframe[['cuota_cred_hipot', 'cuota_de_vivienda']].max(axis=1)
                
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
        dataframe['ingreso_cero'] = np.where(dataframe['ingreso_corr'] == 0, 1, 0)
        
        # Variables de interaccion
        dataframe['interact_ing_gen']  = dataframe['genero'] * dataframe['ingreso_corr']
        dataframe['interact_ing_ed']  = dataframe['edad'] * dataframe['ingreso_corr']
        dataframe['interact_cup_gen']  = dataframe['genero'] * dataframe['total_cupo']
        dataframe['interact_cup_ed']  = dataframe['edad'] * dataframe['total_cupo']
        dataframe['interact_obl_gen'] = dataframe['genero'] * dataframe['obl_total_pct']
        dataframe['interact_obl_ed']  = dataframe['edad'] * dataframe['obl_total_pct']    

        #dataframe['ingreso_geo_alto']  = np.where(dataframe['ingreso_corr'] < 14.90, 1, 0) # ALgo mas tecnico

    
        #dataframe['pc25'] = np.where(dataframe['ingreso_corr'] <= np.quantile(dataframe['ingreso_corr'],0.25), 1, 0)
        #dataframe['pc75'] = np.where(dataframe['ingreso_corr'] >= np.quantile(dataframe['ingreso_corr'],0.75), 1, 0)

        # variables al cuadrado
        
        #dataframe['edad_2'] = dataframe['edad']**2
        dataframe['total_cuota_2']  =dataframe['total_cuota']**2
        #dataframe['total_cupo_2']  =dataframe['total_cupo']**2
        #dataframe['obl_total_pct_2'] = dataframe['obl_total_pct']**2
        #dataframe['ingreso_corr2'] = dataframe['ingreso_corr']**2 
        
        
        dataframe['cupo_pct'] = dataframe['total_cupo']/dataframe['ingreso_corr']
        dataframe['cupo_disponible'] = dataframe['total_cupo'] - dataframe['cuota_tarjeta_de_credito'] - \
                                       dataframe['cuota_tc_bancolombia']
        dataframe['liquidez'] = dataframe['cupo_disponible'] + dataframe['ingreso_corr']
        #dataframe['liquidez_c'] = dataframe['total_cupo'] + dataframe['ingreso_corr']
        dataframe['cuota_pct_cupo'] = (dataframe['cuota_tarjeta_de_credito'] + dataframe['cuota_tc_bancolombia']) / dataframe['total_cupo']
        #dataframe['ind_corregido'] = dataframe['ingreso_corr'] - dataframe['total_cuota'] - dataframe['ingreso_corr']*0.1
        #dataframe['ratio_cupo'] = dataframe['cupo_tc_mdo']/dataframe['cupo_tc_mdo']
        
        
        
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
        for feat in skewed_features:
            dataframe[feat] = boxcox1p(dataframe[feat], lam)        
        
        cat_vars = [
            'mora_max',
            'estado_civil',
            'rep_calif_cred',
            'ocupacion',
            'tipo_vivienda',
            'nivel_academico',
            #'departamento_residencia'
        ]
        
        dummified = []
        for var in cat_vars:            
            dummified.append(
                pd.get_dummies(dataframe[var], drop_first=True, prefix=var)
            )
        
        dummified = pd.concat(dummified, axis=1)
        dataframe = pd.concat([dataframe.drop(cat_vars, axis=1),dummified], axis=1)
        
        ## Final cleaning
        
        dataframe.drop(["ingreso_final", "ingreso_calculado", "cuota_de_vivienda"], axis=1, inplace=True)
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.fillna(0, inplace=True)

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
        