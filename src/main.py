import os
from server import run_server

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main():
    run_server(num_clients=20, num_rounds=100)


if __name__ == '__main__':
    main()
