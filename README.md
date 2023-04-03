# federeco
Federated recommendation system based on neural collaborative filtering [1].
Implementation is done using both Tensorflow/Keras and PyTorch frameworks. 
Main branch includes Pytorch implementation, with Tensorflow implementation on `tensorflow` branch.
### Requirements
torch 2.0.0 \
numpy 1.24.0 \
pandas 2.0.0 \
tqdm 4.64.0 \
scipy 1.9.3 
### Installation
Clone the repository:\
`git clone https://github.com/Ach113/federeco` 

Install the dependencies: \
`pip install -r requirements.txt` 

__Note:__ `requirements.txt` does not include `torch`, you must install it separately. \
You can find the appropriate installation command [here](https://pytorch.org/get-started/locally/).

### Usage
Package itself is located in the `src` folder as `src/federeco`. \
All the other files are drivers to instantiate clients and run the training process from a server.
```python
from federeco.models import NeuralCollaborativeFiltering as NCF
from federeco.train import training_process
from federeco.eval import evaluate_model

# define client class
class Client:
    def __init__(self, client_id):
        """  client constructor  """
    
    def train(self):
        """ method that trains client locally """

ncf = NCF(num_users, num_items)
clients =  # list of Client objects #

# launch training process
model = training_process(
    ncf,  # server-side model to train
    clients,  # list of all clients in the system
    num_clients,  # number of clients to sample per epoch
    epochs  # number of training epochs
) 

# evaluate the model
hr, ndcg = evaluate_model(model, users, items, negatives, k=10)
```
### Running the simulator
```
usage: federeco [-h] [-d dataset] [-p path] [-e epochs]

federated recommendation system

options:
  -h, --help             show this help message and exit
  -d dataset, --dataset  which dataset to use, default "movielens"
  -p path, --path        path where trained model is stored, default "pretrained/ncf.h5"
  -e epochs, --epochs    number of training epochs, default 400
```
```
python src/main.py -d movielens -p pretrained/ncf_movielens.h5 -e 1000
```
    
### Dataset
For model training and validation we use [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
and [Pinterest](https://paperswithcode.com/dataset/pinterest) datasets. \
Processed versions of the datasets are provided in `data` folder.

### Parameters
Training parameters, such as learning rate, number of global & local epochs, number of negative samples are all taken from 
Federated Neural Collaborative Filtering paper [2]. \
__Note:__ this repository is not an implementation of the paper linked above, we simply use it \
as a baseline for setting hyperparameters of our model.

Parameters, along with file paths and other configurations are provided in `src/federeco/config.py`.

### Planned features
1. way to add new users/items into the system
2. add other types of models beside NCF

### References
[[1]](https://arxiv.org/pdf/1708.05031.pdf) X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural Collaborative
Filtering,” Proceedings of the 26th International Conference on World Wide Web -
WWW ’17, 2017, doi: 10.1145/3038912.3052569 \
[[2]](https://arxiv.org/pdf/2106.04405.pdf) V. Perifanis and P. S. Efraimidis, “Federated Neural Collaborative Filtering,” Knowledge-Based Systems, vol. 242, p. 108441, Apr. 2022