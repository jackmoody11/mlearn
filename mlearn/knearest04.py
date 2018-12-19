import numpy as np
import pandas as pd
from collections import Counter
import warnings
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


df = pd.read_csv('../data/breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0

for g in test_set:
    for d in test_set[g]:
        vote = k_nearest_neighbors(train_set, d, k=5)
        if g == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)

