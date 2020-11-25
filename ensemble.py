from typing import Dict

from data import Data
from data_processor import Data_Processor
from model_config import Model_Config
from model_manager import Estimator_Stats, Model_Manager


class Ensemble:

    def __init__(self, estimator_stats: Dict[Model_Config, Estimator_Stats], data_processor: Data_Processor):
        self.__estimator_stats = estimator_stats

    def evaluate(self):
        pass

    def fit(self):
        pass

    def predict(self, x):
        pass


d = Data()
dp = Data_Processor(d)
m = Model_Manager(dp)
m.choose_data_process()
m.decide_if_weight_class()
m.hyperparams_tuning()
m.evaluate_estimators()
e = Ensemble(m.estimators_stats, dp)
e.evaluate()
e.fit()
y_train_pred = e.predict(x_train)
y_test_pred = classifier.predict(x_test)
print("Accuracy on training set: {:4.2f}"
      .format(accuracy_score(y_train, y_train_pred)))
print("Accuracy on test set: {:4.2f}"
      .format(accuracy_score(y_test, y_test_pred)))



