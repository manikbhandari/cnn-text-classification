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

        self.args = args
        embeddings = pickle.load(open(args.vocab_embed_path, 'rb'))

        self.embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(embeddings.astype(np.float32)))

        self.fixed_embedding_layer = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.fixed_embedding_layer.weight = nn.Parameter(torch.from_numpy(embeddings.astype(np.float32)))
        self.fixed_embedding_layer.weight.requires_grad = False

        filter_heights = args.filter_heights.split(',')
        filter_heights = [int(filter_height) for filter_height in filter_heights]
        self.convs = nn.ModuleList([])
        in_channels = 2 if self.args.use_fixed_embed and self.args.use_trainable_embed else 1
        for filter_height in filter_heights:
            self.convs.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=args.num_filters,
                                        kernel_size=(filter_height, embeddings.shape[1])))
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=args.drop_prob)
        self.output_layer = nn.Linear(in_features=len(filter_heights) * args.num_filters,
                                      out_features=args.num_labels)

    def forward(self, input_batch):
        if self.args.use_trainable_embed:
            input_batch_embedded = self.embedding_layer(input_batch).unsqueeze(1)
        if self.args.use_fixed_embed:
            input_batch_fixed_embedded = self.fixed_embedding_layer(input_batch).unsqueeze(1)

        if self.args.use_fixed_embed and self.args.use_trainable_embed:
            input = torch.cat([input_batch_embedded, input_batch_fixed_embedded], dim=1)
        else:
            input = input_batch_embedded if self.args.use_trainable_embed else input_batch_fixed_embedded

        outs = []
        for conv in self.convs:
            out = conv(input)  # B X num_filters X time X 1
            out = out.squeeze(-1)  # B X num_filters X time
            out = self.relu_layer(out)
            # max pool across time
            out = torch.max(out, dim=-1)[0]  # B X num_filters
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self.dropout_layer(out)
        out = self.output_layer(out)

        return out

    def print_params(self):
        for name, param in self.named_parameters():
            print(name, param.data.shape)
