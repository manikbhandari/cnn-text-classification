import pickle
import argparse
import yaml
import torch
import os
import torch.nn as nn
import pdb
import json
from torch.utils.data import DataLoader
from data import ClassificationDataset
from models import KimCNN


def get_dataloader(args):
    train_data_path = config['train_data_path']
    valid_data_path = config['valid_data_path']
    test_data_path = config['test_data_path']
    train_data = pickle.load(open(train_data_path, 'rb'))
    valid_data = pickle.load(open(valid_data_path, 'rb'))
    test_data = pickle.load(open(test_data_path, 'rb'))

    train_dataset = ClassificationDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = ClassificationDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = ClassificationDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


def make_model(args):
    if args.model == 'kim_cnn':
        return KimCNN(args)
    else:
        raise NotImplementedError("unkown model type {}".format(args.model))


def get_opt(args, model):
    if args.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        return NotImplementedError("unknown optimizer {}".format(args.optim))


def normalize_weight(weight_tensor):
    norm = torch.norm(weight_tensor)
    if norm > args.max_l2_norm:
        weight_tensor.div_(norm/args.max_l2_norm)


def get_loss(criterion, out, labels):
    loss = criterion(out, labels)
    return loss


def get_accuracy(out, labels):
    _, preds = torch.max(out, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.shape[0], correct


def train_epoch(train_dataloader, model, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    for i_batch, sample_batch in enumerate(train_dataloader):
        sentences = sample_batch['sentences']  # bs X sl
        lengths = sample_batch['lengths']
        labels = sample_batch['labels']
        if args.cuda:
            sentences = sentences.cuda()
            labels = labels.cuda()

        out = model.forward(sentences)
        loss = get_loss(criterion, out, labels)
        loss.backward()
        acc, ncorrect = get_accuracy(out, labels)
        total_loss += loss.item()
        total_correct += ncorrect

        optimizer.step()
        normalize_weight(model.output_layer.weight.data)
        optimizer.zero_grad()
        if i_batch > 0 and i_batch % args.batch_print == 0:
            print("batch ", i_batch, "loss ", loss.item(), "batch acc: ", acc)

    acc = total_correct / len(train_dataloader.dataset)
    print(f"train epoch: {epoch} \
            train loss: {total_loss:.4f} \
            train accuracy: {acc:.4f}")
    return total_loss, acc


def valid_epoch(valid_dataloader, model, criterion, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    for i_batch, sample_batch in enumerate(valid_dataloader):
        sentences = sample_batch['sentences']  # bs X sl
        lengths = sample_batch['lengths']
        labels = sample_batch['labels']
        if args.cuda:
            sentences = sentences.cuda()
            labels = labels.cuda()

        out = model.forward(sentences)
        loss = get_loss(criterion, out, labels)
        acc, ncorrect = get_accuracy(out, labels)
        total_loss += loss.item()
        total_correct += ncorrect

    acc = total_correct / len(valid_dataloader.dataset)
    print(f"valid epoch: {epoch} \
            valid loss: {total_loss:.4f} \
            valid accuracy: {acc:.4f}")
    return total_loss, acc


def write_results(dataloader, trained_model, out_file):
    trained_model.eval()
    all_preds = []
    lbl2id = json.load(open(config['vocab_labels_path']))
    id2lbl = {id: lbl.lower() for lbl, id in lbl2id.items()}
    for i_batch, sample_batch in enumerate(valid_dataloader):
        sentences = sample_batch['sentences']  # bs X sl
        lengths = sample_batch['lengths']
        labels = sample_batch['labels']
        if args.cuda:
            sentences = sentences.cuda()
            labels = labels.cuda()

        out = trained_model.forward(sentences)
        _, preds = torch.max(out, 1)
        preds = preds.tolist()

        all_preds += preds

    results = [id2lbl[id] for id in all_preds]
    with open(out_file, 'w') as f:
        f.write("\n".join(results))


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", action='store_true', help="Run trained model on test data")
    parser.add_argument("-batch_size", type=int, default=50, help="")
    parser.add_argument("-model", default="kim_cnn", help="model to use: kim_cnn")
    parser.add_argument("-filter_heights", default="3,4,5", help="heights of filters")
    parser.add_argument("-num_filters", type=int, default=100, help="number of filters of each height")
    parser.add_argument("-num_labels", type=int, default=16, help="number of labels in the dataset")
    parser.add_argument("-optim", default="adam", help="optimizer to use: adam")
    parser.add_argument("-lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("-batch_print", type=int, default=20000, help="print status of batch after these many batches")
    parser.add_argument("-drop_prob", type=float, default=0.5, help="probability to drop")
    parser.add_argument("-epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("-max_l2_norm", type=float, default=3.0, help="normalize output layer weights to this value")
    parser.add_argument("-cuda", action='store_true', help="train model on gpu")
    parser.add_argument("-use_fixed_embed", action='store_true', help="use both fixed embedding")
    parser.add_argument("-use_trainable_embed", action='store_true', help="use both trainable embedding")
    parser.add_argument("-gpu", default="0", help="which gpu to use")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = yaml.safe_load(open("config.yml"))
    args.vocab_embed_path = config['vocab_embed_path']

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args)
    criterion = nn.CrossEntropyLoss()
    model = make_model(args)
    model_path = "best_model"
    if args.cuda:
        model.cuda()
    optimizer = get_opt(args, model)
    best_valid_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(train_dataloader, model, criterion, optimizer, epoch)
        valid_loss, valid_acc = valid_epoch(valid_dataloader, model, criterion, epoch)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if args.test:
                save_model(model, model_path)
    print(f"best valid acc: {best_valid_acc:.4f}")
    if args.test:
        load_model(model_path, model)
        write_results(test_dataloader, model, "tests_results.txt")
        write_results(valid_dataloader, model, "valid_results.txt")