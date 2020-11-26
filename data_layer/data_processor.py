from collections import Counter
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_layer.data_analyzer import Data_Analyzer


class Data_Processor:
    _N_PCA_COMPONENTS = 5

    def __init__(self, to_drop_correlated=True, use_pca=False, to_standartize=False):
        self.__to_drop_correlated = to_drop_correlated
        self.__use_pca = use_pca
        self.__to_standartize = to_standartize

        self.__correlated_features = []
        self.__pca = PCA(n_components=self._N_PCA_COMPONENTS)
        self.__scaler = StandardScaler()

    def process(self, x_train=None, x_test=None):
        train_df = pd.DataFrame(x_train) if x_train is not None else None
        test_df = pd.DataFrame(x_test) if x_test is not None else None

        if self.__to_drop_correlated:
            if train_df is not None:
                self.__correlated_features = Data_Analyzer.calc_correlations(train_df)
                train_df.drop(self.__correlated_features, axis=1, inplace=True)
            if test_df is not None:
                test_df.drop(self.__correlated_features, axis=1, inplace=True)

        if self.__use_pca:
            feature_names = [f"PCA_Feature {i}" for i in range(self._N_PCA_COMPONENTS)]
            if train_df is not None:
                self.__pca.fit(train_df)
                train_df = pd.DataFrame(self.__pca.transform(train_df), columns=feature_names)
            if test_df is not None:
                test_df = pd.DataFrame(self.__pca.transform(test_df), columns=feature_names)

        if self.__to_standartize:
            if train_df is not None:
                self.__scaler.fit(train_df)
                train_df = pd.DataFrame(self.__scaler.transform(train_df), columns=train_df.columns)
            if test_df is not None:
                test_df = pd.DataFrame(self.__scaler.transform(test_df), columns=test_df.columns)

        x_train = None
        x_test = None
        if train_df is not None:
            x_train = train_df.values
        if test_df is not None:
            x_test = test_df.values

        if x_train is None:
            return x_test
        if x_test is None:
            return x_train

        return x_test, x_train

