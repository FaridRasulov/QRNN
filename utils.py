import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from torch.utils.data import Dataset, DataLoader

def check_restore_parameters(sess, saver, checkpoint_path):
    try:
        os.mkdir(checkpoint_path)
    except OSError:
        pass

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print ("Loading parameters")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print ("Initializing fresh parameters")

def get_embeddings(vocab, path, dim = 300):
    vocab_dict = {word: int(_id) for _id, word in vocab.items()}
    embed_id = path.split('.')[-2]
    if embed_id+'_imdb.json' not in os.listdir('.'):
        embeds = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                split = line.split()
                word = split[0].encode('utf-8').lower()
                vec = split[1:]
                if word in vocab_dict.keys():
                    embeds[vocab_dict[word]] = list(map(float,vec))
        with open(embed_id+'_imdb.json', 'w') as f:
            f.write(json.dumps(embeds))
    else:
        with open(embed_id+'_imdb.json') as f:
            embeds = {int(_id): word for _id, word in json.loads(f.read()).items()}
    embed_list = []
    for i in range(3):
        var = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[dim]), dtype=tf.float32)
        embed_list.append(var)
    for _id, word in vocab.items():
        if int(_id) in embeds.keys(): embed_list.append(tf.constant(embeds[_id], dtype=tf.float32))
        else: embed_list.append(embed_list[2])
    return tf.stack(embed_list, axis=0)

class imdbDataset(Dataset):
    def __init__(self, dataset, seq_len=100):
        self.x = dataset[0]
        self.pad_inputs(seq_len)
        self.get_masks(seq_len)
        self.y = dataset[1]

    def pad_inputs(self, seq_len):
        new_xs = []
        for x in self.x:
            if len(x) > seq_len: x = x[:seq_len]
            elif len(x) < seq_len: x += [0] * (seq_len - len(x))
            assert len(x) == seq_len
            new_xs.append(x)
        self.x = new_xs

    def get_masks(self, seq_len):
        self.masks = []
        for x in self.x:
            mask = [1.0 * (x_i != 0) for x_i in x]
            assert len(mask) == seq_len
            self.masks.append(mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.masks[i], self.y[i]


def get_datasets(batch_size=100, num_words=1000, seq_len=100):
    train, test = imdb.load_data(num_words=num_words)
    vocab = imdb.get_word_index()
    vocab = {int(_id): word.encode('utf-8').lower() for word, _id in vocab.items() if _id <= num_words}
    train = imdbDataset(train, seq_len=seq_len)
    test = imdbDataset((test[0], test[1]), seq_len=seq_len)
    return (DataLoader(train, batch_size), DataLoader(test, batch_size),vocab)