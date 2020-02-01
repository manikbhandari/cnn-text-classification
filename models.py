import torch
import pickle
import pdb
import numpy as np
import torch.nn as nn


class KimCNN(nn.Module):
    """
    The CNN model used in the paper Convolutional Neural Networks for Sentence Classification
    """

    def __init__(self, args):
        super(KimCNN, self).__init__()
        embeddings = pickle.load(open(args.vocab_embed_path, 'rb'))
        self.embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(embeddings.astype(np.float32)))
        filter_heights = args.filter_heights.split(',')
        filter_heights = [int(filter_height) for filter_height in filter_heights]
        self.convs = nn.ModuleList([])
        for filter_height in filter_heights:
            self.convs.append(nn.Conv2d(in_channels=1,
                                        out_channels=args.num_filters,
                                        kernel_size=(filter_height, embeddings.shape[1])))
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=args.drop_prob)
        self.output_layer = nn.Linear(in_features=len(filter_heights) * args.num_filters,
                                      out_features=args.num_labels)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, input_batch):
        input_batch_embedded = self.embedding_layer(input_batch)
        input_batch_embedded = input_batch_embedded.unsqueeze(1)
        outs = []
        for conv in self.convs:
            out = conv(input_batch_embedded)  # B X num_filters X time X 1
            out = out.squeeze(-1)  # B X num_filters X time
            out = self.relu_layer(out)
            # max pool across time
            out = torch.max(out, dim=-1)[0]  # B X num_filters
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self.dropout_layer(out)
        out = self.output_layer(out)
        out = self.softmax_layer(out)

        return out

    def print_params(self):
        for name, param in self.named_parameters():
            print(name, param.data.shape)
