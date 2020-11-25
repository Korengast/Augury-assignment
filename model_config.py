from enum import Enum

import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from MLP import MLP_Classifier


class Model_Config(Enum):
    RANDOM_FOREST = ('random_forest',
                     RandomForestClassifier,
                     {
                         'n_estimators': [10, 50, 100, 500, 1000]
                     })
    XGB = ('xgb',
           xgboost.XGBClassifier,
           {
               'learning_rate': [0.1, 0.3, 0.5],
               'gamma': [0, 0.25, 0.5],
               'max_depth': [3, 6, 10],
               'min_child_weight': [1, 5, 10]
           })
    LOGISTIC_REGRESSION = ('logistic_regression',
                           LogisticRegression,
                           {
                               'penalty': ['l2'],
                                'C': [1, 5, 10]
                           })
    MLP = ('mlp',
           MLP_Classifier,
           {
               # 'layer_sizes': [(512,), (256, 16), (64, 8, 4)],
               # 'n_epochs': [500, 1000, 2000],
               # 'learning_rate': [0.1, 0.01, 0.001]

               'n_epochs': [200, 500],
               'layer_sizes': [(32,), (8, 4)],
               'learning_rate': [0.01, 0.001]
           })

    def __init__(self, model_name, model_class, param_grid):
        self.model_name = model_name
        self.model_class = model_class
        self.param_grid = param_grid

    @classmethod
    def all(cls):
        return [
            cls.LOGISTIC_REGRESSION,
            cls.RANDOM_FOREST,
            cls.XGB,
            cls.MLP
        ]
