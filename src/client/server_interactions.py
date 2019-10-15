import numpy as np
from metrics_dto import MetricsDTO
from ml_evaluation_ipc_communication import EvaluationRunIdentifier

def send_metrics_for_run(socket, run_identifier: EvaluationRunIdentifier, seed: int, run: int, metrics_dto: MetricsDTO):
    metrics_obj = create_calculated_metrics_message(run_identifier, run=run, seed=seed, metrics=metrics_dto)
    socket.send_pyobj(metrics_obj)
    print('Sending metrics object:', metrics_obj)
    # Receive response
    obj = socket.recv_pyobj()
    if not obj:
        # TODO: Custom exception
        raise Exception('Metrics were not synced')


def create_calculated_metrics_message(run_identifier: EvaluationRunIdentifier, run: int, seed: int, metrics: MetricsDTO):
    obj = {'type': 'metrics', 'run_identifier': run_identifier,'run': run, 'seed': seed, 'value': metrics._asdict()}
    return obj


def prepare_data_for_run(socket, run_identifier: EvaluationRunIdentifier, run: int, current_client_seed: int,  data_params: dict):
    data_query_obj = create_data_query(run_identifier=run_identifier, run=run, current_client_seed=current_client_seed, data_params=data_params)
    print('Sending data query object:', data_query_obj)
    socket.send_pyobj(data_query_obj)
    msg = socket.recv_pyobj()
    train_data, test_data = msg
    return train_data, test_data

def create_data_query(run_identifier: EvaluationRunIdentifier, run: int, current_client_seed: int, data_params: dict):
    obj = {'type': 'data', 'run_identifier': run_identifier, 'run': run, 'seed': current_client_seed, **data_params}
    return obj


def request_seed(socket, run_identifier: EvaluationRunIdentifier):
    req = {'type': 'seed', 'run_identifier': run_identifier}
    print('Sending seed query object:', req)
    socket.send_pyobj(req)
    seed_info = socket.recv_pyobj()
    return seed_info
