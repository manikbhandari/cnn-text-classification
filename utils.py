import os
import json
import yaml
import pickle
import random
import numpy as np
from collections import Counter


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


def make_vocab(text, voc_fname, split, add_pad, add_unk, num_words=None):
    """
    Creates and stores vocabulary for text and labels. Needs to be run only once.
    Also adds padding token <PAD>
    :param num_words: number of vocab words. Default is to use all
    :param text: list of sentences
    :param voc_fname: location of vocab file to save
    :return: None. Stores vocab files to voc_fname
    """
    vocab_path = voc_fname
    if os.path.exists(vocab_path):
        print("vocab file exists at {}".format(vocab_path))
        return
    # assume text is space separated. Might be better to use a proper word tokenizer
    if split:
        words = [wrd for line in text for wrd in line.lower().split(" ")]
    else:
        words = text
    counter = Counter(words)
    word_counts = counter.most_common(num_words)
    vocab = {}
    idx = 0
    for wrd, cnt in word_counts:
        vocab[wrd] = idx
        idx += 1
    if add_unk:
        vocab[UNK_TOKEN] = idx
        idx += 1
    if add_pad:
        vocab[PAD_TOKEN] = idx
    print("Creating vocab for {} words".format(len(vocab)))
    with open(voc_fname, 'w') as f:
        json.dump(vocab, f)


def save_dataset(text, labels, output_path):
    """
    Saves dataset padded to max length of the data. Lower cases everything. This might be crucial.
    :param output_path: where to save the pickled data
    :param text: list of sentences
    :param labels: list of labels
    :return: None. Saves the pickled data to output path.
    """
    if os.path.exists(output_path):
        print("output path {} exists".format(output_path))
        return
    label2id = json.load(open(config['vocab_labels_path']))
    word2id = json.load(open(config['vocab_text_path']))
    max_len = max([len(sent.split(" ")) for sent in text])
    sentences_ids = []
    lengths = []
    for i, sent in enumerate(text):
        sent = sent.lower()
        words = sent.split(" ")
        len_words = len(words)
        lengths.append(len_words)  # save the true length of sentence and pad to max_len
        if len_words < max_len:
            words += [PAD_TOKEN] * (max_len - len_words)
        sentence_ids = []
        for word in words:
            if word in word2id:
                sentence_ids.append(word2id[word])
            else:
                sentence_ids.append(word2id[config['unk_token']])
        sentences_ids.append(sentence_ids)
        if i % 10000 == 0:
            print("processed {} sentences".format(i))
    sentences = np.array(sentences_ids)
    lengths_arr = np.array(lengths)
    print("saving sentences of shape {}".format(sentences.shape))
    print("saving lengths_arr of shape {}".format(lengths_arr.shape))

    labels_ids = []
    for i, label in enumerate(labels):
        if label in label2id:
            labels_ids.append(label2id[label])
        else:
            # For test set we don't have labels
            labels_ids.append(-1)
    labels_arr = np.array(labels_ids)
    print("saving labels_arr of shape {}".format(labels_arr.shape))
    data = {
        'sentences': sentences,
        'lengths': lengths_arr,
        'labels': labels_arr
    }
    pickle.dump(data, open(output_path, 'wb'))


def save_embeddings(embed_file, vocab_file, out_file):
    """
    Saves the embeddings as an np array mapped according to the vocabulary
    :param out_file: location of the output pickle
    :param embed_file: embedding file with format <word><space><space separated vector>
    :param vocab_file: word to id mapping file - json format
    :return: None
    """
    if os.path.exists(out_file):
        print("output file {} exists".format(out_file))
        return
    word2id = json.load(open(vocab_file, 'r'))
    embeddings = [[] for _ in range(len(word2id))]  # initialize empty embeddings matrix
    word2vec = {}
    with open(embed_file, 'r') as f:
        for line in f:
            line = line.strip()
            line_split = line.split(" ")
            word, vec = line_split[0], line_split[1:]
            vec = [float(element) for element in vec]
            word2vec[word] = vec
    emb_dim = len(word2vec['the'])
    print("loaded word-vector mapping in memory")
    for word in word2id:
        if word in word2vec:
            embeddings[word2id[word]] = word2vec[word]
        else:
            # initialize random embeddings for unknown words
            embeddings[word2id[word]] = [random.random() for _ in range(emb_dim)]

    embeddings = np.array(embeddings)
    print("saving embeddings of shape {}".format(embeddings.shape))
    pickle.dump(embeddings, open(out_file, 'wb'))


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", 'r'))
    DATAPATH = config['datapath']
    PAD_TOKEN = config['pad_token']
    UNK_TOKEN = config['unk_token']

    valid_text, valid_labels = get_xy(DATAPATH + "topicclass_valid.txt")
    train_text, train_labels = get_xy(DATAPATH + "topicclass_train.txt")
    test_text, test_labels = get_xy(DATAPATH + "topicclass_test.txt")

    make_vocab(train_text + valid_text, config['vocab_text_path'], split=True, add_pad=True, add_unk=True)
    make_vocab(valid_labels + train_labels, config['vocab_labels_path'], split=False, add_pad=False, add_unk=True)

    save_dataset(train_text, train_labels, config['train_data_path'])
    save_dataset(valid_text, valid_labels, config['valid_data_path'])
    save_dataset(test_text, test_labels, config['test_data_path'])

    save_embeddings(config['embed_path'], config['vocab_text_path'], config['vocab_embed_path'])