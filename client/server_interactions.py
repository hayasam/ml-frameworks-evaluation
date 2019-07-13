import numpy as np
from metrics_dto import MetricsDTO


def send_metrics_for_run(socket, experiment_name: str, seed: int, run: int, metrics_dto: MetricsDTO):
    metrics_obj = create_calculated_metrics_message(experiment_name=experiment_name, run=run, challenge='mnist', seed=seed, metrics=metrics_dto)
    socket.send_pyobj(metrics_obj)
    # Receive response
    obj = socket.recv_pyobj()
    if not obj:
        # TODO: Custom exception
        raise Exception('Metrics were not synced')


def create_calculated_metrics_message(experiment_name: str, run: int, challenge: str, seed: int, metrics: MetricsDTO):
    # TODO: Just send the DTO (implies making a middle package)
    obj = {'type': 'metrics', 'experiment_name': experiment_name, 'challenge': challenge, 'run': run, 'seed': seed, 'value': metrics._asdict()}
    return obj


# TODO: Create a clean interface object (ex: DTO)
def create_data_query(experiment_name: str, run: int, challenge: str, seed: int, data_params: dict):
    obj = {'type': 'data', 'experiment_name': experiment_name, 'challenge': challenge, 'run': run, 'seed': seed, **data_params}
    return obj

def prepare_data_for_run(socket, experiment_name: str, run: int, seed: int,  data_params: dict):
    socket.send_pyobj(create_data_query(challenge='mnist', experiment_name=experiment_name, run=run, seed=seed, data_params=data_params))
    msg = socket.recv_pyobj()
    train_data, test_data = msg
    # print(train_data[0].shape, train_data[1].shape)
    return train_data, test_data


def request_seed(socket, experiment_name):
    req = {'type': 'seed', 'experiment_name': experiment_name}
    socket.send_pyobj(req)
    seed_info = socket.recv_pyobj()
    return seed_info
