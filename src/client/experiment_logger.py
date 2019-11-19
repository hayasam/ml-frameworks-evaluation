import logging
from pathlib import Path

DEFAULT_LOG_DIR = '.'


class ExperimentLogger(object):
    def __init__(self, experiment_name, **kwargs):
        self.name = experiment_name
        self.current_run = 0
        self.base_logger = logging.getLogger()
        self.bp = Path(kwargs.get('log_dir', DEFAULT_LOG_DIR))
        assert self.bp.exists()
        # Need to cast to str because of python 3.5
        self._training_log_handler = logging.FileHandler(str(self.bp / '{}.training.log'.format(self.name)))
        
        self.train_logger = self.base_logger.getChild('training')
        self.train_logger.addHandler(self._training_log_handler)
        self.train_logger.setLevel(logging.DEBUG)
        
        self.parameters_logger = self.base_logger.getChild('parameter')
        # Need to cast to str because of python 3.5
        self._parameter_log_handler = logging.FileHandler(str(self.bp / '{}.parameters.log'.format(self.name)))
        self.parameters_logger.addHandler(self._parameter_log_handler)
        self.parameters_logger.setLevel(logging.DEBUG)

        self.metrics_logger = self.base_logger.getChild('metrics')
        # Need to cast to str because of python 3.5
        self.metrics_log_handler = logging.FileHandler(str(self.bp / '{}.metrics.log'.format(self.name)))
        self.metrics_logger.addHandler(self.metrics_log_handler)
        self.metrics_logger.setLevel(logging.DEBUG)

        self.data_logger = self.base_logger.getChild('data')
        # Need to cast to str because of python 3.5
        self.data_log_handler = logging.FileHandler(str(self.bp / '{}.data.log'.format(self.name)))
        self.data_logger.addHandler(self.data_log_handler)
        self.data_logger.setLevel(logging.DEBUG)

        self.base_logger.setLevel(logging.DEBUG)
        self.base_log_handler = logging.StreamHandler()
        self.base_log_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(message)s'))
        self.base_logger.addHandler(self.base_log_handler)


    def train(self, *args, **kwargs):
        self.train_logger.debug('run {}'.format(self.current_run))
        self.train_logger.debug(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        self.parameters_logger.debug('run {}'.format(self.current_run))
        self.parameters_logger.debug(*args, **kwargs)

    def metrics(self, *args, **kwargs):
        self.metrics_logger.debug('run {}'.format(self.current_run))
        self.metrics_logger.debug(*args, **kwargs)
    
    def data(self, *args, **kwargs):
        self.data_logger.debug('run {}'.format(self.current_run))
        self.data_logger.debug(*args, **kwargs)
    
    def status(self, *args, **kwargs):
        self.base_logger.debug(*args, **kwargs)
