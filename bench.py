import subprocess
from subprocess import PIPE
import pickle
import matplotlib.pyplot as plt


def plot(ax, y, label):
    ax.bar(range(len(n)), y, label=f'FedNCF {label}')
    ax.axhline(y=1, color='tab:orange', linestyle='--', label=f'NCF {label}')
    ax.set_xlabel('Number of sampled users')
    ax.set_ylabel(f'Normalized {label}')
    ax.set_xticks(range(len(n)), [str(x) for x in n])
    ax.set_ylim(0, 1.3)
    ax.legend()


hrs = {}
ndcgs = {}

for n in [1, 2, 4, 5, 8, 10]:
    print(fr'running python src/main.py --no-save -e 400 -n 30 -l {n}')
    result = subprocess.run(fr'python src/main.py --no-save -e 400 -n 30 -l {n}', stdout=PIPE, stderr=PIPE).stdout
    result, _ = result.decode("utf-8").split('\r\n')
    hr, ndcg = result.split(' ')
    hrs[n] = float(hr)
    ndcgs[n] = float(ndcg)


with open('hr_dump_1.pkl', 'wb') as f:
    pickle.dump(hrs, f)

with open('ndcg_dump_1.pkl', 'wb') as f:
    pickle.dump(ndcgs, f)


with open('hr_dump_1.pkl', 'rb') as f:
    hrs = pickle.load(f)

with open('ndcg_dump_1.pkl', 'rb') as f:
    ndcgs = pickle.load(f)

with open('history_dump.pkl', 'rb') as f:
    history = pickle.load(f)

with open('history_dump_pin.pkl', 'rb') as f:
    history_pin = pickle.load(f)


n = list(hrs.keys())
hr = [x / 0.4515 for x in list(hrs.values())]
ndcg = [x / 0.2492 for x in list(ndcgs.values())]

print(list(hrs.values()))
print(list(ndcgs.values()))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot(axs[0], hr, 'HR@10')
plot(axs[1], ndcg, 'NCDG@10')
plt.suptitle('FedNCF performance across different number of epochs')
plt.tight_layout()
plt.show()

plt.plot(list(history.keys()), list(history.values()), label='Movielens1M')
plt.plot(list(history_pin.keys()), list(history_pin.values()), label='Pinterest')
plt.ylim(0.2, 0.8)
plt.xlabel('Epochs')
plt.ylabel('BCE Loss')
plt.title('Mean loss across 50 clients per epoch')
plt.legend()
plt.show()
