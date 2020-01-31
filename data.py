import yaml
import torch
from torch.utils.data import Dataset, DataLoader


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