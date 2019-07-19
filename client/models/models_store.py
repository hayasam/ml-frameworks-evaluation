from .base_model import EvaluationModel
import importlib

class ModelStore(object):
    def __init__(self):
        # TODO: Build a dict containing possible values
        # Could use dynamic values based on directory content
        pass
    @staticmethod
    def get_model_for_name(library: str, name: str, **kwargs) -> EvaluationModel:
        qualified_module_name = '{}_models'.format(library).lower()
        module = __import__('{}.{}'.format('models', qualified_module_name), fromlist=[name])
        cl = getattr(module, name)
        return cl(**kwargs)
