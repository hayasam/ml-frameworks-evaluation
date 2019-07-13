import pickle
import numpy as np
from pathlib import Path

DEFAULT_MINIMAL_SEED_LEN = 1_000
DEFAULT_MIN_SEED_VALUE = 0
DEFAULT_MAX_SEED_VALUE = 65536

def init(value_min, value_max, value_count):
    return np.random.randint(value_min, value_max, value_count)

class SeedController(object):
    """Mapping str -> [int]"""
    def __init__(self, **kwargs):
        self.seed_len = kwargs.get('seed_len', DEFAULT_MINIMAL_SEED_LEN)
        self.min_seed_value = kwargs.get('min_val', DEFAULT_MIN_SEED_VALUE)
        self.max_seed_value = kwargs.get('max_val', DEFAULT_MAX_SEED_VALUE)
        if self.seed_len < DEFAULT_MINIMAL_SEED_LEN:
            raise ValueError('Seed length (seed_len) needs to be greater than 1,000')
        min_seed_val, max_seed_val = np.iinfo(np.int).min, np.iinfo(np.int).max
        self._mapping = dict()

    @classmethod
    def from_saved_file(cls, saved_file, **kwargs_non_exsiting):
        if not Path(saved_file).exists():
            return cls(**kwargs_non_exsiting)
        with open(saved_file, 'rb') as cf:
            return pickle.load(cf)

    def dump(self, file_destination, overwrite=False):
        if Path(file_destination).exists() and not overwrite:
            raise ValueError('File {} exists, set overwrite to True')
        with open(file_destination , 'wb') as output_file:
            pickle.dump(self, output_file)

    def get_seeds(self, experiment_name: str):
        if experiment_name in self._mapping:
            return self._mapping[experiment_name]
        self._mapping[experiment_name] = init(self.min_seed_value, self.max_seed_value, self.seed_len)
        return self._mapping[experiment_name]
