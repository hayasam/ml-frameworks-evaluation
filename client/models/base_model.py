import abc

class EvaluationModel(abc.ABC):
    @abc.abstractmethod
    def initialize_weights(self, random_state: int):
        pass
    
    @abc.abstractmethod
    def train_on_data(self, data):
        pass

    @abc.abstractmethod
    def get_data_params(self):
        pass

    @abc.abstractmethod
    def start_training(self):
        pass

    def use_device(self, device_type: str):
        """Makes the model be run on CPU or GPU"""
        pass

    def save(self, evaluation_type: str, run: int):
        pass

    @abc.abstractmethod
    def get_params_str(self):
        return "A string containing parameters for this model"
