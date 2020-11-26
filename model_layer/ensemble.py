from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from data_layer.data import Data
from model_layer.model import Model
from model_layer.model_manager import Model_Manager
from util import get_sample_weight


class Ensemble:

    def __init__(self, models: List[Model]):
        self.__models = models
        self.__model_weights = self.__calc_model_weights()

    def fit(self, x, y, sample_weight=None):
        for model in self.__models:
            model.fit(x, y, sample_weight)

    def predict(self, x):
        preds = np.array([0.0] * x.shape[0])
        for model in self.__models:
            model_preds = model.predict(x)
            weight = self.__model_weights[model]
            preds += weight * model_preds
        return preds.round()

    def __calc_model_weights(self):
        scaler = MinMaxScaler(feature_range=(0.5, 1))
        all_scores = np.array([[model.validation_accuracy for model in self.__models]])
        scaler.fit(all_scores.T)
        total = sum(scaler.transform(all_scores.T))[0]

        model_weights = {}
        for model in self.__models:
            accuracy = np.array([[model.validation_accuracy]])
            model_weights[model] = scaler.transform(accuracy)[0][0] / total
        return model_weights






