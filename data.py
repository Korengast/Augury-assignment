import urllib.request as urllib2
import zipfile
from io import BytesIO
import pandas as pd
from sklearn import preprocessing
import numpy as np


class Data:

    def __init__(self):
        response = urllib2.urlopen(
            'https://github.com/augurysys/machine_learning_assignment/raw/master/augury_ml_assignment_2018.zip')
        augury_ml_assignment_zip = response.read()

        zip_file_strio = BytesIO(augury_ml_assignment_zip)
        zip_file = zipfile.ZipFile(zip_file_strio)

        feature_train_csv_data = BytesIO(zip_file.read('features_train.csv'))
        self.features_train = pd.read_csv(feature_train_csv_data)

        label_train_csv_data = BytesIO(zip_file.read('labels_train.csv'))
        self.labels_train = pd.read_csv(label_train_csv_data)

        feature_test_csv_data = BytesIO(zip_file.read('features_test.csv'))
        self.features_test = pd.read_csv(feature_test_csv_data)

        label_test_csv_data = BytesIO(zip_file.read('labels_test_true.csv'))
        self.labels_test = pd.read_csv(label_test_csv_data)

        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels_test.values)

    def get_train_ground_truth(self, smart_annotators_choosing=False):
        y_train_annotators = self.__extract_annotators_mat()
        votes_counter = y_train_annotators.sum(axis=1)
        all_voted_the_same = len(votes_counter[votes_counter == 5]) + len(votes_counter[votes_counter == 0])
        almost_all_voted_the_same = len(votes_counter[votes_counter >= 4]) + len(votes_counter[votes_counter <= 1])
        print(f"{round(100 * all_voted_the_same / len(votes_counter), 2)}% all annotated the same")
        print(f"{round(100 * almost_all_voted_the_same / len(votes_counter), 2)}% all annotated the same but 1")
        if not smart_annotators_choosing:
            # Use majority vote
            return y_train_annotators.mean(axis=1).astype(float).round()
        else:
            return y_train_annotators.mean(axis=1).astype(float).round()  # fixme

    def get_test_ground_truth(self):
        return self.label_encoder.transform(self.labels_test.values)

    def __extract_annotators_mat(self):
        y_train_annotators = np.zeros_like(self.labels_train.values)
        for annotator in range(y_train_annotators.shape[1]):
            y_train_annotators[:, annotator] = \
                self.label_encoder.transform(self.labels_train.values[:, annotator])
        return y_train_annotators


# d = Data()
# x_train, y_train, x_test, y_test = d.get_model_data()
