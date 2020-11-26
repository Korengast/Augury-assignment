from data_layer.data_processor import Data_Processor
from model_layer.estimator_config import Estimator_Config


class Model:

    def __init__(self, estimator_config: Estimator_Config):
        self.estimator_config = estimator_config
        self.model_name = estimator_config.estimator_name
        self.param_grid = estimator_config.param_grid
        self.data_processor = Data_Processor()
        self.to_weight_class = None
        self.hyper_params = {}
        self.validation_accuracy = None

        self.estimator = None

    def fit(self, x_train, y_train, sample_weight=None):
        x_train = self.data_processor.process(x_train=x_train)
        self.build(input_size=x_train.shape[1])
        if not self.to_weight_class and sample_weight is not None:
            sample_weight = [1] * len(sample_weight)
        self.estimator.fit(x_train, y_train, sample_weight)

    def predict(self, x_test):
        x_test = self.data_processor.process(x_test=x_test)
        return self.estimator.predict(x_test)

    def build(self, input_size=None):
        if self.estimator_config == Estimator_Config.MLP:
            self.estimator = self.estimator_config.estimator_class(input_size=input_size, **self.hyper_params)
        else:
            self.estimator = self.estimator_config.estimator_class(**self.hyper_params)
        return self.estimator
