import logging
from pathlib import Path


class MetricsLoggerStore(object):
    active_loggers = dict()
    def __init__(self, **kwargs):
        self.base_path = Path(kwargs.get('base_path', '.'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.base_logger = logging.getLogger()

    def get_logger(self, experiment_name: str, **kwargs):
        if experiment_name in MetricsLoggerStore.active_loggers:
            return MetricsLoggerStore.active_loggers[experiment_name]
        print('Logger not found, creating')
        experiment_logger = self.base_logger.getChild(experiment_name)
        logger_options = {
            'handler': logging.FileHandler(filename=str(self.base_path / '{}_metrics.log'.format(experiment_name)), mode='a'),
            'formatter': logging.Formatter('%(message)s'),
            'level': logging.DEBUG
        }
        experiment_logger.addHandler(logger_options['handler'])
        logger_options['handler'].setFormatter(logger_options['formatter'])
        experiment_logger.setLevel(logger_options['level'])

        MetricsLoggerStore.active_loggers[experiment_name] = experiment_logger
        return experiment_logger
