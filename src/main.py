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
                        choices=['movielens', 'pinterest', 'yelp'],
                        help='which dataset to use, default "movielens"')
    parser.add_argument('-p', '--path', default='pretrained/ncf.h5', metavar='path',
                        help='path where trained model is stored, default "pretrained/ncf.h5"')
    parser.add_argument('-e', '--epochs', default=400, metavar='epochs', type=int,
                        help='number of training epochs, default 400')
    parser.add_argument('-s', '--save', default=True, metavar='save', action=argparse.BooleanOptionalAction,
                        help='flag that indicates if trained model should be saved')
    parser.add_argument('-n', default=50, metavar='sample_size', type=int,
                        help='number of clients to sample per epoch')
    parser.add_argument('-l', default=3, metavar='local_epochs', type=int,
                        help='number of local training epochs')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='learning_rate',
                        help='learning rate')
    return parser.parse_args()


def main():
    args = parse_arguments()
    # instantiate the dataset based on passed argument
    dataset = Dataset(args.dataset)
    # run the server to load the existing model or train & save a new one
    trained_model = run_server(dataset, num_clients=args.n, epochs=args.epochs,
                               path=args.path, save=args.save, local_epochs=args.l, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()
