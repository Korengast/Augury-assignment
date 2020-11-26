from collections import Counter

import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

from data_layer.data import Data


class Data_Analyzer:
    _CORRELATION_THRESHOLD = 0.95

    def __init__(self, data: Data):
        self.__data = data

        self.__calc_y_distribution()
        self.__check_data_types()
        self.__check_for_missing_values()
        self.__check_for_zeros_rate()
        self.__look_for_noninformative_features()
        self.calc_correlations(self.__data.features_train, to_plot=True)

    @classmethod
    def calc_correlations(cls, df, to_plot=False):
        corr_mat = df.corr()

        highly_correlated = set()
        upper_triangle = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        for feature in upper_triangle.index:
            one_feature_correlations = upper_triangle.loc[feature]
            highly_correlated.update(
                one_feature_correlations.loc[abs(one_feature_correlations) > cls._CORRELATION_THRESHOLD].index)
        if to_plot:
            sn.heatmap(corr_mat, annot=False)
            plt.show()
            print(
                f"{len(highly_correlated)} features were dropped due to more than {cls._CORRELATION_THRESHOLD} correlation")
        return highly_correlated

    def __calc_y_distribution(self):
        self.y_count = Counter(self.__data.get_train_ground_truth())
        total = sum(self.y_count.values())
        plt.bar(self.y_count.keys(), np.array(list(self.y_count.values())) / total)
        plt.show()

    def __check_data_types(self):
        data_types = self.__data.features_train.dtypes
        non_float = data_types[data_types != float]
        print(f"Non-float features: {non_float}")

    def __check_for_missing_values(self):
        n_missing_values = self.__data.features_train.isnull().sum().sum() + \
                           self.__data.features_train.isna().sum().sum()
        print(f"There are {n_missing_values} missing values in features_train")
        n_missing_values = self.__data.features_test.isnull().sum().sum() + \
                           self.__data.features_test.isna().sum().sum()
        print(f"and {n_missing_values} missing values in features_test")

    def __check_for_zeros_rate(self):
        df = self.__data.features_train
        zeros_rate_per_feature = df[df == 0].count(axis=0) / len(df.index)
        feature_with_max = zeros_rate_per_feature.idxmax()
        max_rate = zeros_rate_per_feature.max()
        print(f"The feature '{feature_with_max}' has the highest zeros rate of {max_rate}")

    def __look_for_noninformative_features(self):
        non_informative_features = set()
        description = self.__data.features_train.describe()
        for feature in self.__data.features_train.columns:
            if description[feature]['std'] == 0:
                non_informative_features.add(feature)
        print(f"There are {len(non_informative_features)} non-informative features (std=0)")
        return non_informative_features


