import pickle
from pathlib import Path

import numpy as np


def init(value_min, value_max, value_count):
    return np.random.randint(value_min, value_max, value_count)

class SeedController(object):
    """Mapping str -> [int]"""
    def __init__(self, **kwargs):
        self.seed_len = kwargs['seed_len']
        self.min_seed_value = kwargs['min_val']
        self.max_seed_value = kwargs['max_val']
        # TODO: Hardcoded
        if self.seed_len < 1000:
            raise ValueError('Seed length (seed_len) needs to be greater than 1,000')
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

    def get_random_states(self, seed_identifier: str):
        if seed_identifier in self._mapping:
            return self._mapping[seed_identifier]
        self._mapping[seed_identifier] = init(self.min_seed_value, self.max_seed_value, self.seed_len)
        return self._mapping[seed_identifier]