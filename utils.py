import os
import json
import yaml
from collections import Counter
from data import get_xy


def make_vocab(text, voc_fname, split, add_pad, num_words=None):
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
        exit(0)
    # assume text is space separated. Might be better to use a proper word tokenizer
    if split:
        words = [wrd for line in text for wrd in line.split(" ")]
    else:
        words = text
    counter = Counter(words)
    word_counts = counter.most_common(num_words)
    vocab = {}
    idx = 0
    for wrd, cnt in word_counts:
        vocab[wrd] = idx
        idx += 1
    if add_pad:
        vocab['<PAD>'] = idx
    print("Creating vocab for {} words".format(len(vocab)))
    with open(voc_fname, 'w') as f:
        json.dump(vocab, f)


def save_dataset_embedded(text, labels, text_embed_file, labels_voc_file):
    wrd2vec = {}
    for line in open(text_embed_file, 'r'):
        words = line.split(" ")
        wrd2vec[words[0]] = words[1:]
    dim = len(wrd2vec['the'])
    wrd2vec['<UNK>'] = [0] * dim



if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", 'r'))
    DATAPATH = config['datapath']
    valid_text, valid_labels = get_xy(DATAPATH + "topicclass_valid.txt")
    make_vocab(valid_text, DATAPATH + "test_vocab.json", split=True, add_pad=True)
    make_vocab(valid_labels, DATAPATH + "test_vocab_labels.json", split=False, add_pad=False)