from server import run_server, sample_clients


def main():
    model = run_server(num_clients=20, num_rounds=400, save=True)
    client = sample_clients(1)[0]
    recommendations = client.generate_recommendation(model, k=10)
    print(recommendations)


if __name__ == '__main__':
    main()
