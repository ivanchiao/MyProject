import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

import pandas as pd
topk = 10


def construct_graph(features, label, method):
    fname = File[0]
    num = len(label)

    dist = None
    # Several methods of calculating the similarity relationship between samples i and j (similarity matrix Sij)
    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))

################################# 此处需要修改 ######################################
File = ['graph/kelin_graph.txt', 'data/klein_2000.csv', 'data/klein_truelabel.csv']
################################# 此处需要修改 #####################################

method = ['heat', 'cos', 'ncos', 'p']

# x_df = pd.read_csv(File[1]).T
# x_df = x_df.iloc[1:, :]
# x = x_df.to_numpy().astype(np.float32)
x = pd.read_csv(File[1], header=None).to_numpy().astype(np.float32)

y_df = pd.read_csv(File[2])
y = y_df['x']

lab = y.unique().tolist()
ind = list(range(0, len(lab)))
mapping = {j: i for i, j in zip(ind, lab)}
y = y.map(mapping).to_numpy()

construct_graph(x, y, method[3])