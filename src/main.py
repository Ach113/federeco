from server import run_server


def main():
    run_server(num_clients=20, num_rounds=5, save=True)


if __name__ == '__main__':
    main()
