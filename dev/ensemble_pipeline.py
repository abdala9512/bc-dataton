dates = [
    '201902', '201903', '201904', '201905', '201907', '201908', '201909',
    '201910', '201911', '202001', '202002', '202003', '202004', '202005',
    '202007', '202008', '202009', '202010', '202011'
]

test_df = pd.read_csv("test_cleaned.csv")
test_df['cuota_de_consumo'] = np.where(test_df['cuota_de_consumo'] <0 ,0, test_df['cuota_de_consumo'])
test_df_modeling = DataFramePreProcessor(test_df, test=True)
test_df_modeling.process()

last_predictions_list = []
y_tests = []
y_preds = []
mapes   = []
models = []
p = 0.2
for date in dates:
    print(f"Periodo {date}:")
    raw_dataframe = pd.read_csv(
             f"train_{date}_cleaned.csv",
             header=0,
             skiprows=lambda i: i>0 and random.random() > p
    ).drop("Unnamed: 0", axis=1)
    # raw_dataframe=raw_dataframe[raw_dataframe['gasto_familiar'] >0 ] 
    print(f"     Total rows in original_data {raw_dataframe.shape[0]}" )
    fe_dataframe = DataFramePreProcessor(raw_dataframe)
    fe_dataframe.process()
    X = fe_dataframe.modeling_dataframe[feature_cols]
    y = fe_dataframe.modeling_dataframe['gasto_familiar']
    print(f"     Total rows in transformed_data {fe_dataframe.modeling_dataframe.shape[0]}" )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=111)
    xgb_reg.fit(X_train, y_train)
    models.append(xgb_reg)
    y_pred = xgb_reg.predict(X_test)
    y_tests.append(y_test)
    y_preds.append(y_pred)
    mape = mean_absolute_percentage_error(y_pred, y_test)
    mapes.append(mape)
    print(f"     MAPE {date}: ", mape )
    
    month_test = test_df_modeling.modeling_dataframe[test_df_modeling.modeling_dataframe['periodo'] == int(date)].reset_index()
    final_prediction=xgb_reg.predict(month_test[feature_cols])
    
    submission = pd.concat([month_test['id_registro'],pd.Series(final_prediction)], axis=1)
    submission.columns = ["id_registro", "gasto_familiar"]
    submission['gasto_familiar'] = submission['gasto_familiar'].round(4)
    submission['gasto_familiar'] = np.where(submission['gasto_familiar'] > 10000000, 10000000, submission['gasto_familiar'])
    
    last_predictions_list.append(submission)
