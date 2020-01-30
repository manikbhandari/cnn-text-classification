import os
import json
import yaml
import pickle
import numpy as np
from collections import Counter
from data import get_xy


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
            sentence_ids.append(word2id[word])
        sentences_ids.append(sentence_ids)
        if i % 10000 == 0:
            print("processed {} sentences".format(i))
    sentences = np.array(sentences_ids)
    lengths_arr = np.array(lengths)
    print("saving sentences of shape {}".format(sentences.shape))
    print("saving lengths_arr of shape {}".format(lengths_arr.shape))

    labels_ids = []
    for i, label in enumerate(labels):
        labels_ids.append(label2id[label])
    labels_arr = np.array(labels_ids)
    print("saving labels_arr of shape {}".format(labels_arr.shape))
    data = {
        'sentences': sentences,
        'lengths': lengths_arr,
        'labels': labels_arr
    }
    pickle.dump(data, open(output_path, 'wb'))


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", 'r'))
    DATAPATH = config['datapath']
    PAD_TOKEN = config['pad_token']
    UNK_TOKEN = config['unk_token']
    valid_text, valid_labels = get_xy(DATAPATH + "topicclass_valid.txt")
    train_text, train_labels = get_xy(DATAPATH + "topicclass_train.txt")
    make_vocab(train_text + valid_text, config['vocab_text_path'], split=True, add_pad=True, add_unk=True)
    make_vocab(valid_labels + train_labels, config['vocab_labels_path'], split=False, add_pad=False, add_unk=False)
    save_dataset(train_text, train_labels, config['train_data_path'])
    save_dataset(valid_text, valid_labels, config['valid_data_path'])