import optuna 
import joblib
from optuna import Trial
from optuna.samplers import TPESampler


class ModelTuning:
    
    
    def __init__(self, X, y, algorithm):
        self.X = X
        self.y = y
        self.algorithm = algorithm
        self.best_params= None
    
    def objective(self, trial: Trial, X, y, algorithm) -> float:
        
        if algorithm == 'xgb':
                params = {
                    "n_estimators" : trial.suggest_int('n_estimators', 0, 200),
                    'max_depth':trial.suggest_int('max_depth', 2, 20),
                    'reg_alpha':trial.suggest_int('reg_alpha', 0, 8),
                    'reg_lambda':trial.suggest_int('reg_lambda', 0, 8),
                    'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
                    'gamma':trial.suggest_int('gamma', 0, 5),
                    'learning_rate':trial.suggest_loguniform('learning_rate',0.01,0.5),
                    'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                    'nthread' : -1
                }
                model_ = xgb.XGBRegressor(**params)
                
        elif algorithm == 'catboost':
            params = {
                "n_estimators" : trial.suggest_int('n_estimators', 0, 200),
                "learning_rate": trial.suggest_loguniform('learning_rate',0.01,0.5),
                "depth": trial.suggest_int('depth', 6, 10),
                "l2_leaf_reg": trial.suggest_int('l2_leaf_reg', 2, 30),
                "bagging_temperature": trial.suggest_discrete_uniform('bagging_temperature',0.1,1,0.01)
            }
            model_ = CatBoostRegressor(**params, verbose=False)
        elif algorithm == 'lasso':
            params = {
                 "alpha": trial.suggest_loguniform("alpha", 0.01, 100)
            }
            model_ =Lasso(**params)
        elif algorithm == 'ridge':
            params = {
                 "alpha": trial.suggest_loguniform("alpha", 0.01, 100)
            }
            model_ =Ridge(**params)
        else:
            params = {
            "alpha": trial.suggest_loguniform("alpha", 0.01, 100),
            "l1_ratio": trial.suggest_loguniform("l1_ratio", 0.01, 1)
            }
            model_ = ElasticNet(**params)
            
        # joblib.dump(study, 'study.pkl')
    
        train_X,test_X,train_y,test_y = train_test_split(X, y, test_size = 0.30,random_state = 101)
        model_.fit(train_X,train_y)
        predictions = model_.predict(test_X)

        return mean_absolute_percentage_error(predictions,test_y)
    
    
    def getHyperParameters(self):
        
        study = optuna.create_study(direction='minimize',sampler=TPESampler())
        study.optimize(lambda trial : self.objective(trial, self.X, self.y, self.algorithm),n_trials= 2)
        study.best_trial.params
        print('Best trial {}: score {},\nparams {}'.format(self.algorithm,study.best_trial.value,study.best_trial.params))
        self.best_params=study.best_trial.params