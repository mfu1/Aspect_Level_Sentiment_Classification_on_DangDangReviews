#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import json
import string
import codecs
import jieba

from zhon import hanzi

import xml.etree.ElementTree as ET


DATA_DIR_FROM = '../data_orign/SemEval2015'
DATA_DIR_TO = '../data_processed/SemEval2015'

STOPWORDS = codecs.open('Frequent_Stopwords_ZH.txt', encoding='utf-8').read().replace(' ', '').split(',')
PUNCTUATIONS = set(hanzi.punctuation + string.punctuation)


def remove_punctuation(s, punc_to_space=False):

    if punc_to_space:
        s = ''.join(c if c not in PUNCTUATIONS else ' ' for c in s )
        s = re.sub(r'\s+', ' ', s)
    else:
        s = ''.join(c for c in s if c not in PUNCTUATIONS)

    return s.strip()


def remove_stopwords(s):

    s = list(jieba.cut(s))
    s = ''.join(c for c in s if c not in STOPWORDS)

    return s.strip()


def clean_string(s, segment=False,
                 len_seq_max=-1, remove_pt=False, punc_to_space=False, remove_sw=True):

    s = s.lower()
    s = s.replace('&quot;', '')
    s = s.replace(' ', '')
    s = s.strip(''.join(PUNCTUATIONS))
    if remove_pt:
        s = remove_punctuation(s, punc_to_space)
    if remove_sw:
        s = remove_stopwords(s)
    if segment:
        s = list(jieba.cut(s))
    
    if len_seq_max < 0:
        return s
    return s[:len_seq_max]


def clean_data(data_tree, segment=False, have_label=True, test_gold=False, test_data_ids=set()):
    
    root = data_tree.getroot()
    review_clean = []

    aspect_mapping = {'FUNCTIONALITY': '功能',
                      'QUALITY': '品质',
                      'DESIGN': '设计',
                      'USABILITY': '使用',
                      'SERVICE': '服务',
                      'PRICE': '价格'}

    mapping_entity_tmp = {'HARDWARE': set(['CPU', 'MEMORY', 'HARD_DISK', 'HARD_DISC']),
                          'BATTERY': set(['POWER_SUPPLY']),
                          'SERVICE': set(['SUPPORT', 'WARRANTY', 'PORTS'])}
    mapping_entity = {}
    for k, v in mapping_entity_tmp.items():
        for i in v:
            mapping_entity[i] = k
        
    mapping_attribute_tmp = {'FUNCTIONALITY': set(['OPERATION_PERFORMANCE', 'CONNECTIVITY', 'MISCELLANEOUS']),
                             'QUALITY': set(['GENERAL']),
                             'DESIGN': set(['PORTABILITY', 'DESIGN_FEATURES'])}
    mapping_attribute = {}
    for k, v in mapping_attribute_tmp.items():
        for i in v:
            mapping_attribute[i] = k
    
    for rid in root:
        for sentences in rid:
            for sentence in sentences:
                
                _id = sentence.attrib['id'].strip()
                
                if have_label:
                    if not sentence.find('Opinions'):
                        continue
                else:
                    if _id not in test_data_ids:
                        continue
        
                text_clean = clean_string(sentence[0].text, segment=segment)
                
                if have_label:
                    item = sentence[1][0]
                    entity, attribute = item.attrib['category'].split('#')
                    if entity in mapping_entity:
                        entity = mapping_entity[entity]
                    if attribute in mapping_attribute:
                        attribute = mapping_attribute[attribute]

                    if entity == 'SERVICE':
                        aspect = aspect_mapping[entity]
                    else:
                        aspect = aspect_mapping[attribute]
                    polarity = item.attrib['polarity']
                    opinion = {'entity': entity, 'attribute': attribute, 'aspect': aspect, 'polarity': polarity}
                else:
                    opinion = {}
                    
                review_clean.append({'id': _id, 'text': text_clean, 'opinion': opinion})
                if test_gold:
                    test_data_ids.add(_id)
                    
    return review_clean, test_data_ids


def process_data(file_name_train, file_name_test_gold, file_name_test, segment=False):
    
    data_path_train = DATA_DIR_FROM + '/' + file_name_train + '.xml'
    data_path_test_gold = DATA_DIR_FROM + '/' + file_name_test_gold + '.xml'
    data_path_test = DATA_DIR_FROM + '/' + file_name_test + '.xml'

    data_tree_train = ET.parse(data_path_train)
    data_tree_test_gold = ET.parse(data_path_test_gold)
    data_tree_test = ET.parse(data_path_test)

    data_train_clean, _ = clean_data(data_tree_train, segment=segment)
    data_test_gold_clean, test_data_ids = clean_data(data_tree_test_gold, segment=segment, test_gold=True)
    data_test_clean, _ = clean_data(data_tree_test, segment=segment, have_label=False, test_data_ids=test_data_ids)

    status = '_segment' if segment else ''
    
    data_path_output_train = DATA_DIR_TO + '/' + file_name_train + status + '.json'
    data_path_output_test_gold = DATA_DIR_TO + '/' + file_name_test_gold + status + '.json'
    data_path_output_test = DATA_DIR_TO + '/' + file_name_test + status + '.json'

    json.dump(data_train_clean, open(data_path_output_train, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(data_test_gold_clean, open(data_path_output_test_gold, 'w', encoding='utf-8'), ensure_ascii=False)
    json.dump(data_test_clean, open(data_path_output_test, 'w', encoding='utf-8'), ensure_ascii=False)


def process_data_for_bert(file_name_train, file_name_dev, file_name_test):

    data_path_train = DATA_DIR_FROM + '/' + file_name_train + '.xml'
    data_tree_train = ET.parse(data_path_train)
    data_train_clean, _ = clean_data(data_tree_train)

    data_path_test = DATA_DIR_FROM + '/' + file_name_test + '_gold.xml'
    data_tree_test = ET.parse(data_path_test)
    data_test_clean, _ = clean_data(data_tree_test)

    data_path_output_train = DATA_DIR_TO + '/' + file_name_train + '.csv'
    data_path_output_dev = DATA_DIR_TO + '/' + file_name_dev + '.csv'
    data_path_output_test = DATA_DIR_TO + '/' + file_name_test + '.csv'

    ratio_train = 0.85
    n_samples = len(data_train_clean)

    with open(data_path_output_train, 'w', encoding='utf-8') as f:
        for data in data_train_clean[:int(n_samples * ratio_train)]:
            f.write("{}\t{}\t{}\n".format(data['opinion']['polarity'], data['opinion']['aspect'], data['text']))

    with open(data_path_output_dev, 'w', encoding='utf-8') as f:
        for data in data_train_clean[int(n_samples * ratio_train):]:
            f.write("{}\t{}\t{}\n".format(data['opinion']['polarity'], data['opinion']['aspect'], data['text']))

    with open(data_path_output_test, 'w', encoding='utf-8') as f:
        for data in data_test_clean:
            f.write("{}\t{}\t{}\n".format(data['opinion']['polarity'], data['opinion']['aspect'], data['text']))


def calc_stat(file_name, data_type):

    f = open(DATA_DIR_TO + '/' + file_name + '.json', 'r', encoding='utf-8')
    data_cleaned = json.load(f)
    f.close()

    len_sentences = 0
    cnt_sentences = len(data_cleaned)
    len_sentences_max = 0
    len_sentences_min = sys.maxsize
    len_sentences_max_str = ""
    len_sentences_min_str = ""
    cnt_entity = {}
    cnt_attribute = {}
    cnt_aspect = {}
    cnt_polarity = {'neutral': 0, 'positive': 0, 'negative': 0}

    for sentence in data_cleaned:
        _id = sentence['id']
        text = sentence['text']
        len_sentences += len(text)
        if len_sentences_max < len(text):
            len_sentences_max = len(text)
            len_sentences_max_str = text
        if len_sentences_min > len(text):
            len_sentences_min = len(text)
            len_sentences_min_str = text

        if data_type == 'train':
            opinion = sentence['opinion']
            aspect = opinion['aspect']
            entity = opinion['entity']
            attribute = opinion['attribute']
            polarity = opinion['polarity']

            cnt_aspect[aspect] = cnt_aspect.get(aspect, 0) + 1
            cnt_entity[entity] = cnt_entity.get(entity, 0) + 1
            cnt_attribute[attribute] = cnt_attribute.get(attribute, 0) + 1
            cnt_polarity[polarity] = cnt_polarity.get(polarity, 0) + 1

    len_sentences_ave = len_sentences / cnt_sentences

    f_output = open(DATA_DIR_TO + '/' + file_name + '_stat' + '.txt', 'w')

    f_output.write('{}: sample numbers: {}\n'.format(data_type, cnt_sentences))
    f_output.write('{}: average sample length: {:.2f}\n'.format(data_type, len_sentences_ave))
    f_output.write('{}: max sample length: {:.2f}\n'.format(data_type, len_sentences_max))
    f_output.write('{}: max sample: {}\n'.format(data_type, len_sentences_max_str))
    f_output.write('{}: min sample length: {:.2f}\n'.format(data_type, len_sentences_min))
    f_output.write('{}: min sample: {}\n'.format(data_type, len_sentences_min_str))

    if data_type == 'train':

        aspects = sorted(cnt_aspect.items(), key=lambda x: (x[1], x[0]), reverse=True)
        entities = sorted(cnt_entity.items(), key=lambda x: (x[1], x[0]), reverse=True)
        attributes = sorted(cnt_attribute.items(), key=lambda x: (x[1], x[0]), reverse=True)

        f_output.write('{}: negative number: {} ({:.2f}%)\n'.format(data_type,
                                                                    cnt_polarity['negative'],
                                                                    cnt_polarity['negative'] / cnt_sentences * 100))
        f_output.write('{}: neutral number: {} ({:.2f}%)\n'.format(data_type,
                                                                   cnt_polarity['neutral'],
                                                                   cnt_polarity['neutral'] / cnt_sentences * 100))
        f_output.write('{}: positive number: {} ({:.2f}%)\n'.format(data_type,
                                                                    cnt_polarity['positive'],
                                                                    cnt_polarity['positive'] / cnt_sentences * 100))

        f_output.write('\n{}: total unique aspects: {}\n'.format(data_type, len(aspects)))
        f_output.write('{}: top 10 aspects: \n'.format(data_type))
        for aspect in aspects[:10]:
            f_output.write('- {}: {}\n'.format(*aspect))

        f_output.write('\n{}: total unique entities: {}\n'.format(data_type, len(entities)))
        f_output.write('{}: all entities: \n'.format(data_type))
        for entity in entities:
            f_output.write('- {}: {}\n'.format(*entity))

        f_output.write('\n{}: total unique attributes: {}\n'.format(data_type, len(attributes)))
        f_output.write('{}: all attributes: \n'.format(data_type))
        for attribute in attributes:
            f_output.write('- {}: {}\n'.format(*attribute))

    f_output.write('\n----------------------------------------\n')

    f_output.close()


if __name__ == '__main__':

    process_data('dataset_train', 'dataset_test_gold', 'dataset_test', segment=False)
    process_data('dataset_train', 'dataset_test_gold', 'dataset_test', segment=True)
    process_data_for_bert('dataset_train', 'dataset_dev', 'dataset_test')

    calc_stat('dataset_train', 'train')
    calc_stat('dataset_test', 'test')
