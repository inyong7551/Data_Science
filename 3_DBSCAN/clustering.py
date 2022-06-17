import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class DBSCAN:
    def __init__(self, n_, eps_, min_pts_):
        self.obj = None
        self.dist = None
        self.num = -1
        self.clusters = 0
        self.n = int(n_)
        self.eps = int(eps_)
        self.min_pts = int(min_pts_)

    def load_data(self, path):
        data = pd.DataFrame(columns=('x', 'y', 'visited', 'label'))
        input_ = open(path, 'r')
        while True:
            data_ = input_.readline().rstrip()
            if not data_:
                break
            data_ = data_.split('\t')
            data.loc[data_[0]] = [data_[1], data_[2], False, -1]
        input_.close()

        n_ = len(data)
        _data = data[['x', 'y']].to_numpy(dtype=float)
        p, q = np.meshgrid(np.arange(n_), np.arange(n_))

        self.obj = data
        self.num = n
        self.dist = np.sqrt(np.sum(((_data[p] - _data[q]) ** 2), 2))

    def clustering(self):
        for idx, p in self.obj.iterrows():
            if not p['visited']:
                p['visited'] = True
                neighbor = self.group(idx)

                if len(neighbor) >= self.min_pts:
                    p['label'] = self.clusters
                    self.expand(p, neighbor, self.clusters)
                    self.clusters += 1

    def group(self, i):
        g = self.dist[int(i), :] <= self.eps
        neighbors = np.where(g)[0].tolist()

        return neighbors

    def expand(self, p, group, label):
        for idx in group:
            idx = str(idx)
            if not self.obj.loc[idx, 'visited']:
                self.obj.loc[idx, 'visited'] = True

                neighbor = self.group(idx)
                if len(neighbor) >= self.min_pts:
                    p['label'] = label
                    self.expand(self.obj.loc[idx], neighbor, label)
                else:
                    self.obj.loc[idx, 'label'] = label

    def store_data(self, num):
        result = list()

        if self.clusters > self.n:
            for i in range(self.clusters):
                mask = self.obj['label'] == i
                result.append(self.obj[mask])

            result = sorted(result, key=lambda x: -len(x))

            for i in range(self.n):
                path = "result/input" + str(num) + "_cluster_" + str(i) + ".txt"
                file = open(path, 'w')
                for idx, p in result[i].iterrows():
                    file.write(str(idx) + "\n")

                file.close()
        else:
            for i in range(self.clusters):
                path = "result/input" + str(num) + "_cluster_" + str(i) + ".txt"
                file = open(path, 'w')
                for idx, p in self.obj.iterrows():
                    if p['label'] == i:
                        file.write(str(idx) + "\n")

                file.close()

    def plot(self):
        fig, ax = plt.subplots()

        for i in range(self.clusters):
            is_group = self.obj['label'] == i
            group = self.obj[is_group]
            print(group['x'])
            ax.plot(group['x'].to_numpy(), group['y'].to_numpy(), marker='o', linestyle='', label=i)

        is_noise = self.obj['label'] == '-1'
        noise = self.obj[is_noise]
        ax.plot(noise['x'].to_numpy(), noise['y'].to_numpy(), marker='x', linestyle='', label='noise')

        ax.legend(fontsize=10, loc='upper left')
        plt.title('Scatter Plot of Clustering results', fontsize=15)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.show()


if __name__ == "__main__":
    sys.setrecursionlimit(100000)

    input_path = "data/" + sys.argv[1]
    n = sys.argv[2]
    eps = sys.argv[3]
    min_pts = sys.argv[4]

    mod = DBSCAN(n, eps, min_pts)
    mod.load_data(input_path)
    mod.clustering()
    mod.store_data(input_path[-5])
    # mod.plot()
