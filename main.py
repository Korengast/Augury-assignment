from sklearn.metrics import accuracy_score

from data_layer.data import Data
from data_layer.data_analyzer import Data_Analyzer
from model_layer.ensemble import Ensemble
from model_layer.model_manager import Model_Manager
from util import get_sample_weight

if __name__ == '__main__':
    data = Data()
    Data_Analyzer(data)

    models_manager = Model_Manager(data)
    models_manager.choose_data_process()
    models_manager.decide_if_weight_class()
    models_manager.hyperparams_tuning()
    models_manager.evaluate_models()

    ensemble = Ensemble(models_manager.models)
    x_train = data.features_train.values
    y_train = data.get_train_ground_truth()
    x_test = data.features_test.values
    y_test = data.get_test_ground_truth()
    sample_weight = get_sample_weight(y_train)
    ensemble.fit(x_train, y_train, sample_weight)

    y_train_pred = ensemble.predict(x_train)
    y_test_pred = ensemble.predict(x_test)
    print("Accuracy on training set: {:4.2f}"
          .format(accuracy_score(y_train, y_train_pred)))
    print("Accuracy on test set: {:4.2f}"
          .format(accuracy_score(y_test, y_test_pred)))

    models_manager = Model_Manager(data, fitted_estimator=ensemble)
    models_manager.choose_data_process()
    models_manager.decide_if_weight_class()
    models_manager.hyperparams_tuning()
    models_manager.evaluate_models()

    ensemble2 = Ensemble(models_manager.models)
    y_train = data.get_train_ground_truth(fitted_estimator=ensemble)
    sample_weight = get_sample_weight(y_train)
    ensemble2.fit(x_train, y_train, sample_weight)

    sample_weight = get_sample_weight(y_train)
    ensemble2.fit(x_train, y_train, sample_weight)
    y_train_pred = ensemble2.predict(x_train)
    y_test_pred = ensemble2.predict(x_test)
    print("Accuracy on training set: {:4.2f}"
          .format(accuracy_score(y_train, y_train_pred)))
    print("Accuracy on test set: {:4.2f}"
          .format(accuracy_score(y_test, y_test_pred)))
