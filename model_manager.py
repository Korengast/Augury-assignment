from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost
import numpy as np
from typing import Tuple

from MLP import MLP_Classifier
from data import Data
from data_processor import Data_Processor
from model_config import Model_Config


class Estimator_Stats:
    def __init__(self):
        self.data_process_type = None
        self.is_weight_class = None
        self.hyper_params = None
        self.validation_score = None


class Model_Manager:
    _CV_FOR_TUNING = 2

    def __init__(self, data_processor: Data_Processor):
        self.__data_processor = data_processor
        self.estimators_stats = {model_config: Estimator_Stats() for model_config in Model_Config.all()}

        # self.__build()

    # def __build(self):
        # self.__choose_data_process()
        # self.__tune_hyperparams()
        # self.__evaluate_ensembl()
        # self.__fit_ensemble()

    def choose_data_process(self):
        y = self.__data_processor.y_train
        for model_config in self.estimators_stats:
            data_type_scores = {'all_features': 0,
                                'all_features_standartized': 0,
                                'no_corr_features': 0,
                                'no_corr_features_standartized': 0,
                                'pca_features': 0}
            for data_type in data_type_scores:
                x = getattr(self.__data_processor, data_type)['train']
                x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=777)
                if model_config == Model_Config.MLP:
                    estimator = model_config.model_class(input_size=x_train.shape[1])
                else:
                    estimator = model_config.model_class()
                estimator.fit(x_train, y_train)
                preds = estimator.predict(x_validation)
                data_type_scores[data_type] = f1_score(y_validation, preds)
            print(model_config.name)
            for k in data_type_scores:
                print(f"{k}: {data_type_scores[k]}")
            self.estimators_stats[model_config].data_process_type = max(data_type_scores, key=data_type_scores.get)

    def decide_if_weight_class(self):
        y = self.__data_processor.y_train
        for model_config in self.estimators_stats:
            data_process_type = self.estimators_stats[model_config].data_process_type
            x = getattr(self.__data_processor, data_process_type)['train']
            weight_per_class = self.__get_weight_per_class()
            results = {True: None, False: None}
            for is_weight_class in results:
                if is_weight_class:
                    sample_weight = [weight_per_class[i] for i in y]
                else:
                    sample_weight = [1] * len(y)
                x_train, x_validation, y_train, y_validation, weights_train, _ =\
                    train_test_split(x, y, sample_weight, test_size=0.25, random_state=777)
                if model_config == Model_Config.MLP:
                    estimator = model_config.model_class(input_size=x_train.shape[1])
                else:
                    estimator = model_config.model_class()

                estimator.fit(x_train, y_train, weights_train)
                preds = estimator.predict(x_validation)
                results[is_weight_class] = f1_score(y_validation, preds)
            print(model_config.name)
            print(results)
            self.estimators_stats[model_config].is_weight_class = max(results, key=results.get)

    def hyperparams_tuning(self):
        y = self.__data_processor.y_train
        for model_config in self.estimators_stats:
            data_process_type = self.estimators_stats[model_config].data_process_type
            if self.estimators_stats[model_config].is_weight_class:
                weight_per_class = self.__get_weight_per_class()
                sample_weight = [weight_per_class[i] for i in y]
            else:
                sample_weight = [1] * len(y)
            x = getattr(self.__data_processor, data_process_type)['train']
            print(f"Tuning hyper parameters for {model_config.model_name}...")
            param_grid = model_config.param_grid
            if model_config == Model_Config.MLP:
                estimator = model_config.model_class(input_size=x.shape[1])
            else:
                estimator = model_config.model_class()
            hyperparams_tuner = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=self._CV_FOR_TUNING)
            hyperparams_tuner.fit(x, y, sample_weight=sample_weight)

            hyper_params = {}
            best_estimator = hyperparams_tuner.best_estimator_
            for param_name in param_grid:
                print(f"{model_config.model_name.upper()}_{param_name} : {getattr(best_estimator, param_name)}")
                hyper_params[param_name] = getattr(best_estimator, param_name)
            self.estimators_stats[model_config].hyper_params = hyper_params

    def evaluate_estimators(self):
        y = self.__data_processor.y_train
        for model_config in self.estimators_stats:
            data_process_type = self.estimators_stats[model_config].data_process_type
            if self.estimators_stats[model_config].is_weight_class:
                weight_per_class = self.__get_weight_per_class()
                sample_weight = [weight_per_class[i] for i in y]
            else:
                sample_weight = [1] * len(y)

            x = getattr(self.__data_processor, data_process_type)['train']
            x_train, x_validation, y_train, y_validation, weights_train, _ = \
                train_test_split(x, y, sample_weight, test_size=0.25, random_state=777)

            hyper_params = self.estimators_stats[model_config].hyper_params
            if model_config == Model_Config.MLP:
                estimator = model_config.model_class(input_size=x.shape[1], **hyper_params)
            else:
                estimator = model_config.model_class(**hyper_params)
            estimator.fit(x_train, y_train, weights_train)
            preds = estimator.predict(x_validation)
            accuracy = accuracy_score(y_validation, preds)
            print(f"{model_config.name} validation accuracy: {accuracy}")
            self.estimators_stats[model_config].validation_score = accuracy

    def __get_weight_per_class(self):
        total = sum(self.__data_processor.y_count.values())
        return {
            0: self.__data_processor.y_count[0] / total,
            1: self.__data_processor.y_count[1] / total,
        }


# d = Data()
# dp = Data_Processor(d)
# m = Model_Manager(dp)
# m.choose_data_process()
# m.decide_if_weight_class()
# m.hyperparams_tuning()
# m.evaluate_estimators()
