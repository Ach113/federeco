from server import run_server, sample_clients
from dataset import Dataset
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='federeco',
        description='federated recommendation system',
    )

    parser.add_argument('-d', '--dataset', default='movielens', metavar='dataset',
                        choices=['movielens', 'pinterest'],
                        help='which dataset to use, default "movielens"')
    parser.add_argument('-p', '--path', default='pretrained/ncf.h5', metavar='path',
                        help='path where trained model is stored, default "pretrained/ncf.h5"')
    return parser.parse_args()


def main():
    args = parse_arguments()
    # instantiate the dataset based on passed argument
    dataset = Dataset(args.dataset)
    # run the server to load the existing model or train & save a new one
    trained_model = run_server(dataset, num_clients=20, num_rounds=400, path=args.path)
    # pick random client & generate recommendations for them
    client = sample_clients(dataset, 1)[0]
    recommendations = client.generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k=10)
    print('Recommendations for user id: ', client.client_id)
    print(recommendations)


if __name__ == '__main__':
    main()
