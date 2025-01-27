import pickle
from pathlib import Path

import numpy as np



class SeedController(object):
    """Mapping str -> [int]"""
    def __init__(self, meta_seed, use_single_seed=True, **kwargs):
        self.seed_len = kwargs['seed_len']
        self.min_seed_value = kwargs['min_val']
        self.max_seed_value = kwargs['max_val']
        # To make sure the random seeds generation is deterministic
        self.meta_seed = meta_seed
        self.use_single_seed = use_single_seed
        # TODO: Hardcoded
        if self.seed_len < 1000:
            raise ValueError('Seed length (seed_len) needs to be greater than 1,000')
        self._mapping = dict()
        self._underlying = dict()

    @classmethod
    def from_saved_file(cls, meta_seed, saved_file, **kwargs_non_exsiting):
        if Path(saved_file).exists():
            with open(saved_file, 'rb') as cf:
                candidate = pickle.load(cf)
            if candidate.meta_seed == meta_seed:
                return candidate
            else:
                print('Meta seed is different from the saved one, return a new seed controller')
                # Do nothing and let the final return be called
        return cls(meta_seed=meta_seed, **kwargs_non_exsiting)

    def create_seed(self, seed_identifier):
        _vec = np.random.randint(self.min_seed_value, self.max_seed_value, self.seed_len)
        self._underlying[seed_identifier] = _vec
        if self.use_single_seed:
            self._mapping[seed_identifier] = np.repeat(_vec[0], self.seed_len)
        else:
            self._mapping[seed_identifier] = _vec.copy()

    def dump(self, file_destination, overwrite=False):
        if Path(file_destination).exists() and not overwrite:
            raise ValueError('File {} exists, set overwrite to True')
        with open(file_destination , 'wb') as output_file:
            pickle.dump(self, output_file)

    def get_random_states(self, seed_identifier: str):
        if seed_identifier in self._mapping:
            return self._mapping[seed_identifier]
        self.create_seed(seed_identifier)
        return self._mapping[seed_identifier]
