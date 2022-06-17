import sys

import pandas as pd


class RecSys:
    def __init__(self):
        self.rating_matrix = None

    def load_data(self, path):
        data = []
        users = {}
        input_ = open(path, 'r')
        while True:
            data_ = input_.readline()
            if not data_:
                break
            data_ = data_.split('\t')

            if data_[0] not in users:
                pair = dict()
                pair[data_[1]] = data_[2]
                users[data_[0]] = pair
            else:
                users[data_[0]][data_[1]] = data_[2]

        input_.close()

        for user_id, data_ in users.items():
            user = pd.DataFrame.from_dict(data_, orient='index')
            user.rename(columns={user.columns[0]: user_id}, inplace=True)
            data.append(user.transpose())

        self.rating_matrix = pd.concat(data).fillna(0)

    @staticmethod
    def encode_units(x):
        if int(x) <= 0:
            return 0
        if int(x) >= 1:
            return 1

    def infer(self):
        self.rating_matrix = self.rating_matrix.applymap(self.encode_units)
        print(self.rating_matrix)

    class MatFact:
        def __init__(self, data, learning_rate, epochs):
            self.data = data
            self.learning_rate = learning_rate
            self.epochs = epochs


if __name__ == "__main__":
    base_path = sys.argv[1]
    test_path = sys.argv[2]

    mod = RecSys()
    mod.load_data("data/" + base_path)
    mod.infer()
