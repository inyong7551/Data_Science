import sys

import numpy as np


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
    for set_ in res:
        for line in set_:
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
        init_set = []
        for element in li:
            cnt = 0
            for data in self.db.database:
                if np.isin(element, data):
                    cnt += 1
            if cnt >= self.support:
                init_set.append(np.array([[element], cnt], dtype=object))

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
        res = []

        n = int((level + 1) / 2) + 1
        for i in range(1, n):
            set_x = self.freq[i - 1]
            set_y = self.freq[level - i]

            for x in set_x:
                for y in set_y:
                    items = np.union1d(x[0], y[0])
                    if not len(items) == level + 1:
                        continue

                    for data in self.freq[level]:
                        sup = 0
                        if np.array_equiv(items, data[0]):
                            sup = data[1]

                        if sup == 0:
                            continue

                        conf_xy = format(sup / x[1] * 100, ".2f")
                        conf_yx = format(sup / y[1] * 100, ".2f")
                        sup = format(sup / self.size * 100, ".2f")
                        array_x = np.asarray(x[0], dtype=int)
                        array_y = np.asarray(y[0], dtype=int)

                        element_x = "{"
                        element_y = "{"
                        for i in array_x:
                            element_x += str(i)
                            if not i == array_x[-1]:
                                element_x += ", "

                        for i in array_y:
                            element_y += str(i)
                            if not i == array_y[-1]:
                                element_y += ", "

                        element_x += "}\t"
                        element_y += "}\t"

                        first = element_x + element_y + str(sup) + "\t" + str(conf_xy)
                        second = element_y + element_x + str(sup) + "\t" + str(conf_yx)
                        res.append(first)
                        res.append(second)

        return set(res)


support = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

db = preprocess(input_path)
app = Apriori(db, int(support) * 0.01 * len(db))
freqs = app.mining()

for i in range(0, len(freqs)):
    print("Length " + str(i+1) + ": " + str(len(freqs[i])))

analyzer = Analyzer(freqs, len(app.db.database))
res = []
for i in range(1, len(freqs)):
   res.append(analyzer.analyze(i))

recording(output_path, res)