import abc

class EvaluationModel(abc.ABC):
    @abc.abstractmethod
    def initialize_weights(self, random_state: int):
        pass

    def use_device(self, device_type: str):
        """Makes the model be run on CPU or GPU"""
        pass

    def save(self, evaluation_type: str, run: int):
        pass

    def get_params_str(self):
        return "A string containing parameters for this model"