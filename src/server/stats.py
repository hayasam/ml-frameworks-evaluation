from collections import defaultdict
from typing import Union, Optional

import numpy as np
import scipy.stats


def aggregate_file_metrics(file_path, metrics: Union[str, list], n_samples: int, metrics_position: Optional[dict]=None):
    _metrics = [metrics] if isinstance(metrics, str) else metrics
    out = defaultdict(lambda: np.zeros((n_samples, )))
    
    with open(file_path, 'r') as lf:
        line = lf.readline()
        while line != '':
            # aggregate_line_metrics
            splits = [metric_kv.split(': ') for metric_kv in line.split('-')]
            if metrics_position is None:
                metrics_position = {m: next((i for i, s in enumerate(splits) if m == s[0].strip())) for m in _metrics+['run']}
            run = int(splits[metrics_position['run']][1])
            for m in _metrics:
                metric_pos = metrics_position[m]
                metric_value = splits[metrics_position[m]][1]
                out[m][run] = float(metric_value)
            line = lf.readline()
    return out[metrics] if isinstance(metrics, str) else out

def print_pair_metrics_from_files(experiment_1_file: str, experiment_2_file: str, n_samples: int, metrics: list):
    positions = {'run': 0, 'accuracy': 1, 'precision': 2, 'recall': 3, 'f1': 4}
    _metrics = [metrics] if isinstance(metrics, str) else metrics
    # Read metrics from file
    metrics_experiment_1 = aggregate_file_metrics(experiment_1_file, _metrics, n_samples, positions)
    metrics_experiment_2 = aggregate_file_metrics(experiment_2_file, _metrics, n_samples, positions)
    for m in _metrics:
        print('-- Metric {} --'.format(m))
        m_1, m_2  = metrics_experiment_1[m], metrics_experiment_2[m]
        # Our metrics are continuous, so correction for continuity?
        w, p_w = scipy.stats.wilcoxon(m_1, m_2, correction=False)
        mn, p_mn = scipy.stats.mannwhitneyu(m_1, m_2, use_continuity=False)
        tv, p_t = scipy.stats.ttest_ind(m_1, m_2)
        print('Wilcoxon p-value of ', p_w)
        print('Mann-Whitney p-value of ', p_mn)
        print('Student p-value of ', p_t)
