import numpy as np
import sys


def preprocess(path):
    transactions = []
    input_ = open(path, 'r')
    while True:
        ta = input_.readline()
        if not ta:
            break
        temp = ta.split()
        ta_ = list(map(int, temp))
        transactions.append(ta_)
    input_.close()

    return np.array(transactions, dtype=object)


def recording(path, res):
    output_ = open(path, 'w')
    res = res.split('\n')
    for line in res:
        output_.write(line + "\n")

    output_.close()
    print("Task Completed.")


class Transactions:
    def __init__(self, ta):
        self.database = ta

    def count(self, x):
        cnt = 0
        for data in self.database:
            if len(data) >= len(x):
                res = np.isin(data, x)
                if res.sum() == len(x):
                    cnt += 1

        return cnt


class Apriori:
    def __init__(self, ta, sup):
        self.db = Transactions(ta)
        self.support = sup

    def initialize(self):
        li = np.unique(sum(self.db.database, []))
        init_set = np.empty((0, 2))
        for element in li:
            cnt = 0
            for data in self.db.database:
                if np.isin(element, data):
                    cnt += 1
            if cnt >= self.support:
                init_set = np.append(init_set, np.array([[element, cnt]]), axis=0)

        return init_set

    def join(self, candidates, level):
        size = len(candidates)
        freqs = []
        temp = []
        for i in range(0, size):
            for j in range(i + 1, size):
                itemset = np.union1d(candidates[i][0], candidates[j][0])
                if np.size(itemset) == level + 1:
                    count = self.db.count(itemset)
                    temp.append([itemset, count])

        for tmp in temp:
            b = False
            for data in freqs:
                if np.array_equiv(tmp[0], data[0]):
                    b = True
            if not b:
                freqs.append(tmp)

        return freqs

    def pruning(self, freqs):
        cand = freqs.copy()

        for i in range(0, len(freqs)):
            if freqs[i][1] < self.support:
                cand.pop(i - len(freqs) + len(cand))

        return cand

    def task(self, freqs, level):
        cand = self.join(freqs, level)
        return self.pruning(cand)

    def mining(self):
        cnt = 1
        init = self.initialize()
        cand = [list(init)]
        cand_ = self.task(init, cnt)
        cand.append(cand_)
        cnt += 1

        while not len(cand_) == 0:
            cand_ = self.task(cand_, cnt)
            if not len(cand_) == 0:
                cand.append(cand_)
            cnt += 1

        return cand


class Analyzer:
    def __init__(self, freq, size):
        self.freq = freq
        self.size = size

    def analyze(self, level):
        res = ""
        for x in range(0, len(self.freq[level])):
            for y in range(x + 1, len(self.freq[level])):
                itemset = np.union1d(self.freq[level][x][0], self.freq[level][y][0])
                if not len(itemset) == level + 2:
                    continue
                sup = 0

                if level < len(self.freq) - 1:
                    for data in self.freq[level + 1]:
                        if np.array_equiv(itemset, data[0]):
                            sup = data[1]

                    if sup == 0:
                        continue

                    conf_xy = format(sup / self.freq[level][x][1] * 100, ".2f")
                    conf_yx = format(sup / self.freq[level][y][1] * 100, ".2f")
                    sup = format(sup / self.size * 100, ".2f")
                    array_x = np.asarray(self.freq[level][x][0], dtype=int)
                    array_y = np.asarray(self.freq[level][y][0], dtype=int)

                    first = "{" + str(array_x) + "}\t" + "{" + str(array_y) + "}\t" + str(
                        sup) + "\t" + str(conf_xy) + "\n"
                    second = "{" + str(array_y) + "}\t" + "{" + str(
                        array_x) + "}\t" + str(sup) + "\t" + str(conf_yx) + "\n"
                    res += first
                    res += second

        return res


support = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

db = preprocess(input_path)
app = Apriori(db, int(support) * 0.01 * len(db))
freqs = app.mining()

analyzer = Analyzer(freqs, len(app.db.database))
res = ""
for i in range(0, len(freqs)):
    res += analyzer.analyze(i)

recording(output_path, res)