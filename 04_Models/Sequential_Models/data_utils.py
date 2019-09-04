#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import copy
import json

import numpy as np

from torch.utils.data import Dataset
import tensorflow as tf

from pytorch_transformers import *

from gensim.models.keyedvectors import KeyedVectors


# DATA_DIR = 'gs://ddreview-bucket/Sequential_Models/data/data_processed'
DATA_DIR = 'data/data_processed'

MAPPINGS = {
    'SemEval2015': {
        'aspect_mapping': {'功能': 0, '品质': 1, '设计': 2, '使用': 3, '服务': 4, '价格': 5},
        'sentiment_mapping': {'negative': 0, 'positive': 1}
    }
}


def load_word_vec(path, word2idx=None):


    word_vec = KeyedVectors.load_word2vec_format(path, binary=False)

    '''
    with tf.gfile.Open(path, 'r') as f:
        glove_file = datapath('test_glove.txt')
        word_vec = {}
        for line in f.readlines():
            tokens = line.strip().split(' ')
            if word2idx is None or tokens[0] in word2idx.keys():
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
                # word_vec[tokens[0]] = np.array(list(map(float, tokens[1:])))
    '''

    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dataset):

    embedding_matrix_file_name = './embeddings/In_Use/{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), dataset)

    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        with tf.gfile.Open(embedding_matrix_file_name, 'rb') as f:
            embedding_matrix = pickle.load(f)

    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        #fname = 'gs://ddreview-bucket/Sequential_Models/embeddings/Word2Vec/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
        fname = 'embeddings/Word2Vec/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'

        word_vec = load_word_vec(fname, word2idx=word2idx)

        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
            else:
                embedding_matrix[i] = np.random.uniform(low=-0.01, high=0.01, size=embed_dim)  # unseen words will be all-zeros.

        with tf.gfile.Open(embedding_matrix_file_name, 'w') as f:
            pickle.dump(embedding_matrix, f)

    return embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, len_seq_max=None):
        self.len_seq_max = len_seq_max
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text_seg(self, words):
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):

        x = (np.ones(maxlen) * value).astype(dtype)

        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc

        return x

    def words_to_sequence(self, words, reverse=False, len_seq_max=-1):

        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]

        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence

        if reverse:
            sequence = sequence[::-1]

        if len_seq_max == -1:
            len_seq_max = self.len_seq_max

        return Tokenizer.pad_sequence(sequence, len_seq_max, dtype='int64', padding=pad_and_trunc, truncating=pad_and_trunc)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDatasetReader:
    @staticmethod
    def __read_text_seg__(file_dir):

        with tf.gfile.Open(file_dir, 'r') as f:
        # with open(ff, 'r', encoding='utf-8') as f:
            data_cleaned = json.load(f)

        words = []
        len_sentence_max = 0
        for sentence in data_cleaned:
            text_cleaned = sentence['text']
            opinion = sentence['opinion']
            aspect = opinion['aspect']
            words += text_cleaned + [aspect]
            len_sentence_max = max(len_sentence_max, len(text_cleaned))

        return words, len_sentence_max

    @staticmethod
    def __read_data__(file_dir, dataset, tokenizer):

        with tf.gfile.Open(file_dir, 'r') as f:
        # with open(ff, 'r', encoding='utf-8') as f:
            data_cleaned = json.load(f)

        dataset = []
        for sentence in data_cleaned:
            text_cleaned = sentence['text']
            opinion = sentence['opinion']
            aspect = opinion['aspect']
            polarity = opinion['polarity']

            text_indices = tokenizer.words_to_sequence(text_cleaned)
            aspect_indices = tokenizer.words_to_sequence([aspect])

            aspect = MAPPINGS[dataset]['aspect_mapping'][aspect]
            polarity = MAPPINGS[dataset]['sentiment_mapping'][polarity]
            data = {
                    'text_indices': text_indices,
                    'aspect_indices': aspect_indices,
                    'aspect': aspect,
                    'polarity': polarity,
                    }

            dataset.append(data)

        return dataset

    def __init__(self, dataset="SemEval2015", embed_dim=300, len_seq_max=-1):

        train_dev_ratio = 0.9

        print("preparing {0} dataset...".format(dataset))
        fname = {
            'SemEval2015': {
                'train': DATA_DIR + '/' + dataset + '/dataset_train_segment.json',
                'test': DATA_DIR + '/' + dataset + '/dataset_test_gold_segment.json'
            }
        }

        text_seg_train, len_sentence_max_train = MyDatasetReader.__read_text_seg__(fname[dataset]['train'])
        text_seg_test, len_sentence_max_test = MyDatasetReader.__read_text_seg__(fname[dataset]['test'])
        text_seg = text_seg_train + text_seg_test

        if len_seq_max < 0:
            len_seq_max = len_sentence_max_train

        my_tokenizer = Tokenizer(len_seq_max=len_seq_max)
        my_tokenizer.fit_on_text_seg(text_seg)

        self.embedding_matrix = build_embedding_matrix(my_tokenizer.word2idx, embed_dim, dataset)
        self.aspect_embedding_matrix = copy.deepcopy(self.embedding_matrix)

        '''
        data_train = MyDatasetReader.__read_data__(fname[dataset]['train'], dataset, my_tokenizer)
        data_test = MyDatasetReader.__read_data__(fname[dataset]['test'], dataset, my_tokenizer)
        data_train = data_train[:int(len(data_train) * train_dev_ratio)]
        data_val = data_train[int(len(data_train) * train_dev_ratio):]

        self.data_train = MyDataset(data_train)
        self.data_test = MyDataset(data_test)
        self.data_dev = MyDataset(data_val)
        '''

if __name__ == '__main__':

    # EMBED_DIM only 300 is available
    MyDatasetReader("SemEval2015", embed_dim=300, len_seq_max=80)