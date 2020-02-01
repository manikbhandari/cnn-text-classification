import yaml
import torch
from torch.utils.data import Dataset, DataLoader


class ClassificationDataset(Dataset):
    def __init__(self, data):
        """
        :param data: dictionary of sentences, lengths and labels
        """
        self.sentences = data['sentences']
        self.labels = data['labels']
        self.lengths = data['lengths']
        assert len(self.sentences) == len(self.labels) == len(self.lengths)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        element = {'sentences': self.sentences[idx],
                   'labels': self.labels[idx],
                   'lengths': self.lengths[idx]
                   }

        return element


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml"))