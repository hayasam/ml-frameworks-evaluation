from collections import namedtuple

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

# TODO: Put this in its own package
MetricsDTO = namedtuple('MetricsDTO', 'accuracy precision recall f1_score')

# TODO: Create class and maybe override repr / str?
def create_metrics_dto(predictions, target) -> MetricsDTO:
    np_pred = predictions
    if isinstance(np_pred, list):
        np_pred = np.array(np_pred)
    np_pred, np_target = np_pred.ravel(), target.ravel()
    acc, pr, rec, f1 = accuracy_score(y_true=np_target, y_pred=np_pred), precision_score(y_true=np_target, y_pred=np_pred, average='macro'), recall_score(y_true=np_target, y_pred=np_pred, average='macro'), f1_score(y_true=np_target, y_pred=np_pred, average='macro')
    return MetricsDTO(accuracy=acc, precision=pr, recall=rec, f1_score=f1)

def metrics_dto_str(metrics_dto: MetricsDTO) -> str:
    s = 'accuracy: {} - precision: {} - recall: {} - f1: {}'.format(metrics_dto.accuracy, metrics_dto.precision, metrics_dto.recall, metrics_dto.f1_score)
    return s
