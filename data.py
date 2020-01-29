import yaml
import torch
from torch.utils.data import Dataset, DataLoader


def get_xy(fname, sep=" ||| "):
    """
    Expects file containing <label><sep><sentence>
    :param fname: name of file
    :param sep: separator to split lines
    :return: list of labels Y and sentences X
    """
    X, Y = [], []
    with open(fname, 'r') as f:
        all_lines = f.read().split('\n')
        for line in all_lines:
            line = line.strip()
            if len(line) == 0:
                continue
            y, x = line.split(sep)
            # correct for validation mistake in labels
            if y == "Media and darama":
                y = "Media and drama"
            X.append(x)
            Y.append(y)
    assert len(X) == len(Y)
    return X, Y


class ClassificationDataset(Dataset):
    def __init__(self, data):
        """
        :param data: Tuple of X, Y
        """
        self.X, self.Y = data
        assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        element = {'sent': self.X[idx],
                   'class': self.Y[idx]}

        return element


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml"))
    DATAPATH = config['datapath']
    X_train, Y_train = get_xy(DATAPATH + 'topicclass_train.txt')
    X_valid, Y_valid = get_xy(DATAPATH + 'topicclass_valid.txt')
    print("train data len: {}".format(len(X_train)))
    print("valid data len: {}".format(len(X_valid)))