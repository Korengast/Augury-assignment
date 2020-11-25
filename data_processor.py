from collections import Counter

import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import Data


class Data_Processor:
    _CORRELATION_THRESHOLD = 0.95
    _N_PCA_COMPONENTS = 5

    def __init__(self, data: Data):
        self.__data = data

        self.all_features = None
        self.no_corr_features = None

        self.all_features_standartized = None
        self.no_corr_features_standartized = None

        self.pca_features = None
        self.pca_features_standartized = None

        self.class_weights = None

        self.y_train = None
        self.y_test = None

        self.__build()

    def __build(self):
        self.__analyses_and_check()
        self.__build_all_features()
        self.__build_non_corr_features()
        self.__build_pca_features()

        self.y_train = self.__data.get_train_ground_truth()
        self.y_test = self.__data.get_test_ground_truth()

    def __analyses_and_check(self):
        self.__calc_y_distribution()
        self.__check_data_types()
        self.__check_for_missing_values()
        self.__check_for_zeros_rate()
        self.__look_for_noninformative_features()

    def __build_all_features(self):
        self.all_features = {'train': self.__data.features_train.values,
                             'test': self.__data.features_test.values}
        scaler = StandardScaler()
        train_standartized = scaler.fit_transform(self.all_features['train'])
        test_standartized = scaler.transform(self.all_features['test'])
        self.all_features_standartized = {'train': train_standartized,
                                          'test': test_standartized}

    def __build_non_corr_features(self):
        correlated_features = self.__calc_correlations()
        self.no_corr_features = {'train': self.__data.features_train.drop(correlated_features, axis=1).values,
                                 'test': self.__data.features_test.drop(correlated_features, axis=1).values}
        scaler = StandardScaler()
        train_standartized = scaler.fit_transform(self.no_corr_features['train'])
        test_standartized = scaler.transform(self.no_corr_features['test'])
        self.no_corr_features_standartized = {'train': train_standartized,
                                              'test': test_standartized}

        print(self.__data.features_train.drop(correlated_features, axis=1).describe())

    def __build_pca_features(self):
        pca = PCA(n_components=self._N_PCA_COMPONENTS)
        train_pca = pca.fit_transform(self.all_features['train'])
        test_pca = pca.transform(self.all_features['test'])
        self.pca_features = {'train': train_pca,
                             'test': test_pca}

        scaler = StandardScaler()
        train_standartized = scaler.fit_transform(self.pca_features['train'])
        test_standartized = scaler.transform(self.pca_features['test'])
        self.pca_features_standartized = {'train': train_standartized,
                                          'test': test_standartized}

    def __calc_y_distribution(self):
        self.y_count = Counter(self.__data.get_train_ground_truth())
        total = sum(self.y_count.values())
        # plt.bar(self.y_count.keys(), np.array(list(self.y_count.values())) / total)
        # plt.show()

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

    def __calc_correlations(self):
        df = self.__data.features_train
        corr_mat = df.corr()
        # sn.heatmap(corr_mat, annot=False)
        # plt.show()

        highly_correlated = set()
        upper_triangle = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))
        for feature in upper_triangle.index:
            one_feature_correlations = upper_triangle.loc[feature]
            highly_correlated.update(one_feature_correlations.loc[abs(one_feature_correlations) > self._CORRELATION_THRESHOLD].index)
        print(f"{len(highly_correlated)} features were dropped due to more than {self._CORRELATION_THRESHOLD} correlation")
        return highly_correlated


d = Data()
dp = Data_Processor(d)
# bp = 0
