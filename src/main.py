from server import run_server, initialize_clients
from federeco.train import sample_clients
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
    parser.add_argument('-e', '--epochs', default=400, metavar='epochs', type=int,
                        help='number of training epochs, default 400')
    return parser.parse_args()


def main():
    args = parse_arguments()
    # instantiate the dataset based on passed argument
    dataset = Dataset(args.dataset)
    # run the server to load the existing model or train & save a new one
    trained_model = run_server(dataset, num_clients=20, epochs=args.epochs, path=args.path)
    # pick random client & generate recommendations for them
    clients = initialize_clients(dataset)
    client = sample_clients(clients, 1)[0]
    recommendations = client.generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k=5)
    print('Recommendations for user id:', client.client_id)
    if args.dataset == 'movielens':
        print(dataset.get_movie_names(recommendations))
    else:
        print(recommendations)


if __name__ == '__main__':
    main()
