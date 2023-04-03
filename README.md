# federeco
Neural collaborative filtering recommendation system utilizing federated averaging for data privacy.
Implementation is done using both Tensorflow/Keras and PyTorch frameworks. 
Main branch includes Pytorch implementation, with Tensorflow implementation on `tensorflow` branch.
# Requirements
numpy 1.22.1 \
torch 1.10.1 \
pandas 1.1.3 \
tqdm 4.64.0 \
scipy 1.7.3 
# How to run
Clone the repository:\
`git clone https://github.com/Ach113/federeco` 

Install the dependencies: \
`pip install -r requirements.txt`

And run `main.py` to execute the code and see the results.
    
# Dataset
The program uses [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset. Processed version of it is provided in `data` folder.

# Parameters
Training parameters, such as learning rate, number of global & local epochs, number of negative samples are all taken from 
[Federated Neural Collaborative Filtering paper](https://arxiv.org/abs/2106.04405). \
Parameters, along with file paths and other configurations are provided in `config.py` file.

# Planned features
1. way to add new users/items into the system
2. add other types of models beside NCF

# References
[1] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural Collaborative
Filtering,” Proceedings of the 26th International Conference on World Wide Web -
WWW ’17, 2017, doi: 10.1145/3038912.3052569. Available: https://arxiv.org/pdf/1708.05031.pdf \
[2] V. Perifanis and P. S. Efraimidis, “Federated Neural Collaborative Filtering,” Knowledge-Based Systems, vol. 242, p. 108441, Apr. 2022