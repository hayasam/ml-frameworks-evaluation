import os

import numpy as np
from dotenv import load_dotenv

load_dotenv(verbose=True)

# General options that won't be changed in general
_DEFAULT_MINIMAL_SEED_LEN = 1_000
_DEFAULT_MIN_SEED_VALUE = 0
_DEFAULT_MAX_SEED_VALUE = np.iinfo(np.uint32).max

DEFAULT_MINIMAL_SEED_LEN = os.getenv('DEFAULT_MINIMAL_SEED_LEN', _DEFAULT_MINIMAL_SEED_LEN)
DEFAULT_MIN_SEED_VALUE = os.getenv('DEFAULT_MIN_SEED_VALUE', _DEFAULT_MIN_SEED_VALUE)
DEFAULT_MAX_SEED_VALUE = os.getenv('DEFAULT_MAX_SEED_VALUE', _DEFAULT_MAX_SEED_VALUE)

# Options that are more likely to change
_DEFAULT_METRICS_LOG_PATH = '.logs'
_DEFAULT_SEED_FILE = '.seed_control.pickle'
_DEFAULT_SERVER_LOG_FILE = 'server.log'

SEED_CONTROLLER_FILE = os.getenv('SEED_CONTROLLER_FILE', _DEFAULT_SEED_FILE)
METRICS_LOG_BASE_PATH = os.getenv('METRICS_LOG_BASE_PATH', _DEFAULT_METRICS_LOG_PATH)
SERVER_LOG_FILE = os.getenv('SERVER_LOG_FILE', _DEFAULT_SERVER_LOG_FILE)
