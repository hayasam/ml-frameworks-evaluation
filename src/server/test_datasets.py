from challenges import get_challenges
from data_server import get_data_for_challenge_seed, dataset_to_numpy, cut_data_for_experiment, shuffle_dataset
SERVER_LOGGER = {'debug': lambda x: print(x)}
def test_challenge(challenge_name: str):
    seed = 0
    run = 0
    # challenges = get_challenges('/docker/volumes/data/_data')
    challenges = get_challenges('./data', download=True)
    the_challenge = challenges[challenge_name]
    print(the_challenge)
    train_loader, test_loader = the_challenge.get_subset(run=run, seed=seed, train_batch_size=64, test_batch_size=64)
    np_train = dataset_to_numpy(train_loader)
    np_test = dataset_to_numpy(test_loader)
    # shuffled_train = cut_data_for_experiment(shuffle_dataset(np_train[0], np_train[1], seed), 0.5)
    # shuffled_test = cut_data_for_experiment(shuffle_dataset(np_test[0], np_test[1], seed), 0.5)
    print(np_train[0].shape, np_train[1].shape, np_test[0].shape, np_test[1].shape)
    return np_train, np_test

if __name__ == "__main__":
    # print('MNIST Challenge')
    # test_challenge('mnist')
    # print('CIFAR10 Challenge')
    # test_challenge('cifar')
    print('ImageNet Challenge')
    test_challenge('imagenet')
