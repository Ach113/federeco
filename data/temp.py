import pandas as pd
import random
import tqdm
import pickle

# cols = ['user_id', 'item_id', 'rating']
# df_train = pd.read_csv('yelp-train.csv', header=None, names=cols)
# df_test = pd.read_csv('yelp-test.csv', header=None, names=cols)
# df = pd.concat([df_train, df_test], axis=0)
# df = df.drop(columns=['rating'])
#
# num_users = 100_000
# sampled_users = set(random.sample(df['user_id'].values.tolist(), num_users))
# df = df[df['user_id'].isin(sampled_users)]
#
# all_items = set(df['item_id'].unique())
# missing = dict()
# gr = df.groupby('user_id')
#
# for i, g in tqdm.tqdm(gr):
#     missing[i] = list(all_items - set(g['item_id']))[:99]

with open('missing_dict.pkl', 'rb') as f:
    missing = pickle.load(f)

print(missing[0])
