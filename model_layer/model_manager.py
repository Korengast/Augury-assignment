from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost
import numpy as np
import pandas as pd
from typing import Tuple

from data_layer.data import Data
from data_layer.data_processor import Data_Processor
from model_layer.estimator_config import Estimator_Config
from model_layer.model import Model
from util import get_sample_weight


class Estimator_Stats:
    def __init__(self):
        self.data_process_type = None
        self.is_weight_class = None
        self.hyper_params = None
        self.validation_score = None


class Model_Manager:
    _CV_FOR_TUNING = 2

    def __init__(self, data: Data, fitted_estimator=None):
        self.__data = data
        self.models = [Model(estimator_config) for estimator_config in Estimator_Config.all()]

        self.__fitted_estimator = fitted_estimator

    def choose_data_process(self):
        y = self.__data.get_train_ground_truth(self.__fitted_estimator)
        df = self.__data.features_train

        x_train, x_validation, y_train, y_validation = train_test_split(df, y, test_size=0.25)

        for model in self.models:
            highest_score = 0
            best_processor = None
            for drop_correlated in (True, False):
                for use_pca in (True, False):
                    for standartize in (True, False):
                        data_processor = Data_Processor(drop_correlated, use_pca, standartize)
                        model.data_processor = data_processor
                        model.fit(x_train, y_train)
                        preds = model.predict(x_validation)
                        if f1_score(y_validation, preds) > highest_score:
                            highest_score = f1_score(y_validation, preds)
                            best_processor = data_processor
            model.data_processor = best_processor

    def decide_if_weight_class(self):
        y = self.__data.get_train_ground_truth(self.__fitted_estimator)
        train_df = self.__data.features_train
        sample_weight = get_sample_weight(y)
        x_train, x_validation, y_train, y_validation, weight_train, _ = self.__split_for_balanced_test(train_df, y, sample_weight, test_size=0.2)

        for model in self.models:
            highest_score = 0
            to_weight_class_decision = None
            for to_weight_class in (True, False):
                model.to_weight_class = to_weight_class
                model.fit(x_train, y_train, weight_train)
                preds = model.predict(x_validation)
                if accuracy_score(y_validation, preds) > highest_score:
                    highest_score = accuracy_score(y_validation, preds)
                    to_weight_class_decision = to_weight_class
            model.to_weight_class = to_weight_class_decision

    def hyperparams_tuning(self):
        y = self.__data.get_train_ground_truth(self.__fitted_estimator)
        x = self.__data.features_train.values
        sample_weight = get_sample_weight(y)

        for model in self.models:
            print(f"Tuning hyper parameters for {model.model_name}...")
            param_grid = model.param_grid
            if model.model_name == Estimator_Config.MLP.estimator_name:
                estimator = model.build(input_size=x.shape[1])
            else:
                estimator = model.build()
            hyperparams_tuner = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=self._CV_FOR_TUNING)
            hyperparams_tuner.fit(x, y, sample_weight=sample_weight)
            hyper_params = {}
            best_estimator = hyperparams_tuner.best_estimator_
            for param_name in param_grid:
                print(f"{model.model_name.upper()}_{param_name} : {getattr(best_estimator, param_name)}")
                hyper_params[param_name] = getattr(best_estimator, param_name)

            model.hyper_params = hyper_params

    def evaluate_models(self):
        y = self.__data.get_train_ground_truth(self.__fitted_estimator)
        df = self.__data.features_train
        sample_weight = get_sample_weight(y)

        x_train, x_validation, y_train, y_validation, weight_train, _ = train_test_split(df, y, sample_weight, test_size=0.25)

        for model in self.models:
            model.fit(x_train, y_train, weight_train)
            preds = model.predict(x_validation)
            accuracy = accuracy_score(y_validation, preds)
            f1 = f1_score(y_validation, preds)
            print(f"{model.model_name} validation accuracy: {accuracy}")
            print(f"{model.model_name} validation f1: {f1}")
            model.validation_accuracy = accuracy



    def __split_for_balanced_test(self, train_df, y, sample_weight, test_size):
        sample_weight = np.array(sample_weight)
        test_portion = test_size
        test_abs_size = int(test_portion * len(y))
        indices_of_0 = np.where(y == 0)[0]
        indices_of_1 = np.where(y == 1)[0]

        if min(len(indices_of_0), len(indices_of_1)) <= test_abs_size / 2:
            raise ValueError(f"Not enough data to have balanced test split. Decrease test portion")

        n_0 = n_1 = test_abs_size//2
        if n_0 + n_1 < test_abs_size:
            n_1 += 1

        test_indices = list(np.random.choice(indices_of_0, size=n_0, replace=False))
        test_indices += list(np.random.choice(indices_of_1, size=n_1, replace=False))
        test_indices.sort()

        all_indices = np.array(list(range(len(y))))
        train_indices = np.array([idx for idx in all_indices if idx not in test_indices])

        x_train = train_df.values[train_indices, :]
        x_test = train_df.values[test_indices, :]

        y_train = y[train_indices]
        y_test = y[test_indices]

        train_weights = sample_weight[train_indices]
        test_weights = sample_weight[test_indices]

        return x_train, x_test, y_train, y_test, train_weights, test_weights

