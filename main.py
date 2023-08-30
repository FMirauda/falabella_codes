# Carga de datos

import sys
import utils as ut
import utils_sc as usc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pytz
from google.cloud import storage
import pickle
from sklearnneuralprophet import SklearnNeuralProphet
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

print('pandas:', pd.__version__)
print('numpy:', np.__version__)


today = datetime.datetime.now(pytz.timezone('America/Santiago')).date()

correlativo_datos = ut.dt_to_str(today)
fecha_inicio_datos = ut.dt_to_str(today - datetime.timedelta(days=(365 * 3 -5)))
fecha_fin_datos =  ut.dt_to_str(today - datetime.timedelta(days=2))

# correlativo_datos = 'test'
# fecha_inicio_datos = '2019-04-28'
# fecha_fin_datos = '2022-04-27'

n_test_days = 14
last_test_day = fecha_fin_datos
# last_val_day = '2022-03-20'

locales = ['2000', '9990', '9913']

drop_sundays = False
periods = 28
picking_manual = False
path_picking_manual = 'data/picking_manual_20220502.xlsx'

# Creación de series de tiempo

### Importando datos de las tablas
qry1 = "select * from `tc-sc-bi-bigdata-corp-tsod-dev.shipping_capacity.data_from_fleet_soc`"
df = ut.read_bq(qry1)
df['fecha_pactada'] = pd.to_datetime(df.fecha_pactada)
# ut.cusdis(df)

qry2 = "select * from `tc-sc-bi-bigdata-corp-tsod-dev.shipping_capacity.data_from_fleet`"
rg = ut.read_bq(qry2)
rg['fecha_pactada'] = pd.to_datetime(rg.fecha_pactada)
# rg['day_of_week'] = rg.fecha_pactada.dt.day_of_week
# rg.set_index('fecha_pactada', inplace=True)
# ut.cusdis(rg)

# Ambos valores inclusive
start_ratio_date = '2021-01-15'
end_ratio_date = '2021-12-31'

# fr -> merged folios & registros
fr = df.drop(columns='unidades').merge(rg.drop(columns='unidades'),
                                       left_on = ['fecha_pactada', 'reparto', 'comuna', 'size'],
                                       right_on = ['fecha_pactada', 'reparto', 'Comuna', 'size'])
fr = fr[(fr.fecha_pactada >= start_ratio_date) & (fr.fecha_pactada <= end_ratio_date)]

ratios = fr.groupby(['reparto', 'comuna', 'size'], as_index=False)[['folios', 'registros']].sum()
ratios['ratio'] = ratios.registros / ratios.folios
# ut.cusdis(ratios)

### Estimación de folios

mg_cols = ['reparto', 'comuna', 'size', 'ratio']

rrg = rg.merge(ratios[mg_cols], how='left', 
               left_on=['reparto', 'Comuna', 'size'], 
               right_on=['reparto', 'comuna', 'size'])

rrg['folios'] = rrg.registros / rrg.ratio

# ut.cusdis(rrg)

df.set_index('fecha_pactada', inplace=True)
rrg.set_index('fecha_pactada', inplace=True)

### Concatenación de ambas series de tiempo

concat_cols = ['reparto', 'comuna', 'size', 'folios']
rg_concat = rrg[rrg.index < start_ratio_date].drop(columns='comuna').rename(columns={'Comuna': 'comuna'})[concat_cols]

df_concat = df[df.index >= start_ratio_date][concat_cols]
df = pd.concat([rg_concat, df_concat])
df['folios'] = df.folios.astype('float32')

## Forecast de ventas de retail
qry3 = "select * from `tc-sc-bi-bigdata-corp-tsod-dev.shipping_capacity.forecast_ventas_retail`"
ventas = ut.read_bq(qry3)
ventas['process_date'] = pd.to_datetime(ventas.process_date)
ventas['predicted_date'] = pd.to_datetime(ventas.predicted_date)
# ut.cusdis(ventas)

## Programa de eventos (cyber y pandemia)
qry4 = "select * from `tc-sc-bi-bigdata-corp-tsod-dev.shipping_capacity.cronograma_eventos`"
eventos = ut.read_bq(qry4)
eventos['date'] = pd.to_datetime(eventos.date)
eventos.rename(columns={'date': 'fecha_pactada'}, inplace=True)

# ut.cusdis(eventos)

## Picking planificado
# celda ya modificada para picking desde serie armada para este proyecto
file = open('querys/cota_picking_reparto.sql', 'r')
qry2 = file.read()
file.close()
pckp = ut.read_bq(qry2)
pckp['capacity_date'] = pd.to_datetime(pckp.capacity_date)
# ut.cusdis(pckp)

## Parchando "a mano" el picking planificado
def formato_picking_semanal(pk, col):
    date = pd.to_datetime(col)
    df = pd.DataFrame()

    for i in range(7):
        temp = pk[['reparto', 'local', col]].copy()
        temp.columns = ['reparto', 'local', 'picking_general']
        temp['fecha'] = date
        df = pd.concat([df, temp])
        date += datetime.timedelta(1)
    
    return df

def formato_picking_manual(pk, reg_rep=None, reg_loc=None, reg_esp=None):
    
    df = pd.DataFrame()
    week_cols = pk.columns[pk.columns.str.startswith('20')]
    
    for col in week_cols:
        temp = formato_picking_semanal(pk, col)
        df = pd.concat([df, temp], ignore_index=True)
    
    df['day_of_week'] = df.fecha.dt.day_of_week
    reg_esp['fecha'] = pd.to_datetime(reg_esp.fecha)
    
    # Aplicando reglas específicas
    df['p_base'] = 1
    df = df.merge(reg_rep, how='left')
    df = df.merge(reg_loc, how='left')
    df = df.merge(reg_esp, how='left')
    df['p_final'] = df.p_especifico.combine_first(df.p_local.combine_first(df.p_reparto.combine_first(df.p_base)))
    
    df['picking'] = df.picking_general * df.p_final
    
    # formatos finales
    df['reparto'] = df.reparto.astype(str)
    df.rename(columns={'fecha': 'capacity_date'}, inplace=True)
    df = df[['local', 'reparto', 'capacity_date', 'picking']]
    
    return df

if picking_manual:
    
    pk = pd.read_excel(path_picking_manual)
    reg_rep = pd.read_excel('data/reglas_reparto.xlsx')
    reg_loc = pd.read_excel('data/reglas_local.xlsx')
    reg_esp = pd.read_excel('data/picking_especifico.xlsx')
    pkm = formato_picking_manual(pk, reg_rep=reg_rep, reg_loc=reg_loc, reg_esp=reg_esp)
    
    pkm['capacity_date'] = pd.to_datetime(pkm.capacity_date)
    pkm['reparto'] = pkm.reparto.astype(str)

    min_pk = pkm.capacity_date.min()
    min_pk

    pkm = pkm.groupby(['reparto', 'capacity_date'], as_index=False).picking.sum()

    pckp = pd.concat([pckp[pckp.capacity_date < min_pk], pkm], ignore_index=True)

#   ut.cusdis(pkm)

## Cargando detalle de lanes
qry5 = "select * from `tc-sc-bi-bigdata-corp-tsod-dev.shipping_capacity.lane_used_capacity`"
lanes = ut.read_bq(qry5)

lanes['Capacity_date'] = pd.to_datetime(lanes.Capacity_date)
lanes['used_capacity'] = lanes.used_capacity.astype(int)
lanes['day_of_week'] = lanes.Capacity_date.dt.day_of_week
lanes.set_index('Capacity_date', inplace=True)
# ut.cusdis(lanes)

comunas = lanes.zone.str.split('-', expand=True)
lanes['comuna'] = comunas[1]
lanes[2] = comunas[2]
# ut.cusdis(lanes)

lanes = lanes[lanes[2].isna()]
lanes.drop(columns=2, inplace=True)

## Seleccionando comunas

lanes = lanes[lanes.orgn_facility_alias_id.isin(locales)]

rank = lanes.groupby('comuna', as_index=False).used_capacity.sum().sort_values('used_capacity', ascending=False, ignore_index=True)
# ut.show(rank, rows=True)

rank = rank[rank.used_capacity > 40]
comunas_rank = rank.comuna.values
# ut.cusdis(rank)

# display(comunas_rank)

# comunas_extra = np.array(['Extra Urbano', 'LA DEHESA', 'PEÑALOLEN', 'TIL TIL', 'Urbano', 'ÑUÑOA'])
comunas_extra_2 = np.array(['LO BARNECHEA (LA DEHESA)', 'PENALOLEN', 'TIL-TIL', 'NUNOA'])

# Entrenamiento

## Configuraciones

asymmetric_loss_1 = usc.AsymmetricHuberLoss(teta_l = .85, teta_r = 0.15)
asymmetric_loss_3 = usc.AsymmetricHuberLoss(teta_l = .95, teta_r = 0.05)

parameters_6 = {
    'n_lags': [28, 56],
    'p_hist': [1, 0.8],
    'ar_sparsity': [.8, 1],
    'loss_func': ['Huber', asymmetric_loss_1, asymmetric_loss_3]
}

parameters_dev = {
    'n_lags': [periods, periods*2]
}

# print('Combinaciones grilla 1:', usc.n_params_comb(parameters))
print('Combinaciones grilla 6:', usc.n_params_comb(parameters_6))

# Entrenamiento final

# Recordar revisar n_dev, grilla de parámetros y n jobs!!

# Para operar sobre todas las st, setear en None
n_dev = 1


models = {}
errors = []

print(df.dtypes)

for local in locales[:n_dev]:
    
    models[local] = {}
    
    comunas_local = df[(df.reparto == local) & (~df.comuna.isna())].comuna.unique()
    comunas_local = np.intersect1d(np.concatenate([comunas_rank, comunas_extra_2]), comunas_local) # corregido
    

    for comuna in comunas_local[:n_dev]:

        df_local_comuna =  df[(df.reparto == local) & (df.comuna == comuna)]
        if df_local_comuna.shape[0] > 100: # comprobando si existen registros suficientes como para forcastear
            
            models[local][comuna] = {}
            tamanos_local_comuna = df_local_comuna['size'].unique()

            for tamano in tamanos_local_comuna[:n_dev]:
            # for tamano in ['BT']:
                

                df_local_comuna_tamano = df_local_comuna[df_local_comuna['size'] == tamano]
                if df_local_comuna_tamano.shape[0] > periods*2:

                            print(local, comuna, tamano)
        
                            st = usc.get_st(df, local, comuna, tamano, fill_sundays_na=True, drop_sundays=drop_sundays,
                                            fecha_inicio_datos=fecha_inicio_datos, fecha_fin_datos=fecha_fin_datos)

                            ventas_cd = ventas[ventas.reparto == local][['predicted_date', 'prediction']]
                            ventas_cd.columns = ['fecha_pactada', 'forecast_ventas']
                            min_ventas = ventas_cd.fecha_pactada.min()

                            pckp_cd = pckp[pckp.reparto == local][['capacity_date', 'picking']]
                            pckp_cd.columns = ['fecha_pactada', 'picking_planificado']
                            min_picking = pckp_cd.fecha_pactada.min()

                            st = st.merge(ventas_cd, how='outer', left_on='fecha_pactada', right_on='fecha_pactada')
                            st = st.merge(eventos, how='left', left_on='fecha_pactada', right_on='fecha_pactada')
                            st = st.merge(pckp_cd, how='outer', left_on='fecha_pactada', right_on='fecha_pactada')

                            st = st[st.fecha_pactada >= min_picking]
                            st = st[st.fecha_pactada >= min_ventas]

                            st.sort_values('fecha_pactada', ignore_index=True, inplace=True)

                            names = {'fecha_pactada': 'ds', 'folios':'y'}
                            st.rename(columns=names, inplace=True)
                            st.drop(columns=['day_of_week', 'covid'], inplace=True)

                            # st['picking_planificado'] = st.picking_planificado.astype('float')
                            num_cols = st.drop(columns='ds').columns
                            st[num_cols] = st[num_cols].astype('float32')
                            print('st dtypes', st.dtypes)
                            st_train_test = st[st.ds <= last_test_day]

                            index_to_ds = st_train_test.ds.to_dict()
                            splits = usc.time_series_splitter(st_train_test.index, n_splits=4, test_size=14)
                            scorer = make_scorer(usc.asymmetric_mape, greater_is_better=False, index_to_ds=index_to_ds, pon_sub=5, pon_sob=1)

                            model = SklearnNeuralProphet(n_forecasts=28,
                                                         d_hidden = 16,
                                                         num_hidden_layers = 4,
                                                         seasonality_mode = 'additive',
                                                         growth = 'linear',
                                                         events=['cyber', 'black_friday', 'feriado', 'feriado_irrenunciable'],
                                                         future_regressor=['picking_planificado', 'forecast_ventas'],
                                                         # holidays='Chile'
                                                        )

                            # fct = GridSearchCV(model, param_grid=parameters_6, scoring=scorer, cv=splits, n_jobs=1)
                            fct = GridSearchCV(model, param_grid=parameters_dev, scoring=scorer, cv=splits, n_jobs=1)
                            
                            _ = fct.fit(st_train_test, st_train_test.y)

                            models[local][comuna][tamano] = fct
                            errors.append(fct.best_score_)

### Guardando modelos en Cloud Storage

client = storage.Client()
bucket = client.get_bucket('shipping_capacity_models')
blob = bucket.blob('models_'+correlativo_datos+'.pickle')
with blob.open(mode='wb') as file:
    pickle.dump(models, file)

# Generando predicciones

preds = pd.DataFrame()

for local in models:
    for comuna in models[local]:
        for tamano in models[local][comuna]:
            
            print(local,comuna,tamano)
            
            st = usc.get_st(df, local, comuna, tamano, fill_sundays_na=True, drop_sundays=drop_sundays,
                            fecha_inicio_datos=fecha_inicio_datos, fecha_fin_datos=fecha_fin_datos)

            ventas_cd = ventas[ventas.reparto == local][['predicted_date', 'prediction']]
            ventas_cd.columns = ['fecha_pactada', 'forecast_ventas']
            min_ventas = ventas_cd.fecha_pactada.min()

            pckp_cd = pckp[pckp.reparto == local][['capacity_date', 'picking']]
            pckp_cd.columns = ['fecha_pactada', 'picking_planificado']
            min_picking = pckp_cd.fecha_pactada.min()

            st = st.merge(ventas_cd, how='outer', left_on='fecha_pactada', right_on='fecha_pactada')
            st = st.merge(eventos, how='left', left_on='fecha_pactada', right_on='fecha_pactada')
            st = st.merge(pckp_cd, how='outer', left_on='fecha_pactada', right_on='fecha_pactada')

            st = st[st.fecha_pactada >= min_picking]
            st = st[st.fecha_pactada >= min_ventas]

            st.sort_values('fecha_pactada', ignore_index=True, inplace=True)

            names = {'fecha_pactada': 'ds', 'folios':'y'}
            st.rename(columns=names, inplace=True)
            st.drop(columns=['day_of_week', 'covid'], inplace=True)

            num_cols = st.drop(columns='ds').columns
            st[num_cols] = st[num_cols].astype('float32')

            st_pred = st[(st.ds > last_test_day) & (st.ds <= pd.to_datetime(last_test_day) + pd.Timedelta(days=periods))]
            
            model = models[local][comuna][tamano]
            pred = model.predict(st_pred)
            
            pred = pred.to_frame().reset_index()
            pred.columns = ['fecha', 'forecast']
            pred['local'] = local
            pred['comuna'] = comuna
            pred['tamano'] = tamano
            
            preds = pd.concat([preds, pred], ignore_index=True)

preds['fecha_prediccion'] = datetime.datetime.now()
# ut.cusdis(preds)

from google.cloud import bigquery

schema = [
    bigquery.SchemaField("fecha", "DATE"),
    bigquery.SchemaField("forecast", "FLOAT"),
    bigquery.SchemaField("local", "STRING"),
    bigquery.SchemaField("comuna", "STRING"),
    bigquery.SchemaField("tamano", "STRING"),
    bigquery.SchemaField("fecha_prediccion", "DATETIME")
]

pandas_schema = [
    {'name': "fecha", 'type': "DATE"},
    {'name': "forecast", 'type': "FLOAT"},
    {'name': "local", 'type': "STRING"},
    {'name': "comuna", 'type': "STRING"},
    {'name': "tamano", 'type': "STRING"},
    {'name': "fecha_prediccion", 'type': "DATETIME"}
]

def create_table(dataset, table_name, schema, partition_field):
    client = bigquery.Client()
    project = client.project
    dataset_ref = bigquery.DatasetReference(project, dataset_id = dataset)

    table_ref = dataset_ref.table(table_name)

    table = bigquery.Table(table_ref, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field=partition_field,  # name of column to use for partitioning
    ) 

    table = client.create_table(table)

    print(
        "Created table {}, partitioned on column {}".format(
            table.table_id, table.time_partitioning.field
        )
    )

dataset = 'shipping_capacity'
table = 'predicciones'

create_new_table = False
if create_new_table:
    create_table(dataset, table, schema, 'fecha_prediccion')

preds.to_gbq(dataset + '.' + table, project_id='tc-sc-bi-bigdata-corp-tsod-dev', if_exists='append', table_schema=pandas_schema)


print("Proceso ejecutado correctamente")