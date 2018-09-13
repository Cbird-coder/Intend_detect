# coding=utf-8
# @author: cyl
import codecs
import numpy as np
import sys
import os
import re
from collections import OrderedDict
import hparamter as _hp
import csv

def check_then_mkdir(dirpath):
    i=0
    for path in dirpath:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            i=i+1
            continue
    if i==0:
        return 0
    else:
        return 1

def FullToHalf(s):
        n = []
        for char in s:
                num = ord(char)
                if num == 0x3000:
                        num = 32
                elif 0xFF01 <= num <= 0xFF5E:
                        num -= 0xfee0
                num = unichr(num)
                n.append(num)
        return ''.join(n)

def clean_seq(sentence):
    sentence = sentence.strip().decode('utf-8')
    sentence = FullToHalf(sentence)
    sentence = re.sub(ur"[^\u4e00-\u9fffA-Za-z0-9%\._\/\-+]","",sentence)
    sentence = sentence.lower()
    return sentence

def write_vocabulary(path,word_dict,word_freq):
    with open(path,'wb') as f_w:
        f_w.write('<unk>\n<padding>\n'.encode('utf-8'))
        word_dict_new = OrderedDict(sorted(word_dict.items(),key=lambda t: t[1],reverse=True))
        dict_len = 2
        for word,cnt in word_dict_new.items():
            if cnt<word_freq or word == '<unk>' or word =='<padding>':
                continue
            else:
                dict_len = dict_len + 1
                f_w.write(word.encode('utf-8') + '\n')
    return dict_len

def gen_dict(data_list,data_dict):
    for it in data_list:
        if it!='<unk>' and it not in data_dict:
            data_dict[it]= 1
        else:
            data_dict[it]+=1
    return data_dict

def open_file(filename):
    data_dict = {}
    label0_dict = {}
    label1_dict = {}
    label2_dict = {}
    csv_reader = csv.reader(open(filename))
    for inx,items in enumerate(csv_reader):
        if inx == 0:
            continue
        seq = clean_seq(items[1].strip())
        seq = [item for item in seq]
        label0 = clean_seq(items[2].strip())
        label1 = items[3].strip()#-1,0,1
        label2 = clean_seq(items[4].strip())
        if len(seq) ==0:
            print items
            continue
        data_dict = gen_dict(seq,data_dict)
        if label0!='<unk>' and label0 not in label0_dict:
            label0_dict[label0]= 1
        else:
            label0_dict[label0]+=1
        if label1!='<unk>' and label1 not in label1_dict:
            label1_dict[label1]= 1
        else:
            label1_dict[label1]+=1
        label2_dict = gen_dict(label2,label2_dict)

    return data_dict,label0_dict,label1_dict,label2_dict

def build_vocabulary(path_label,path_word,word_freq,voc_path):
    if not os.path.exists(_hp.data_train_path):
        print("No corpus exist!!!,please create now....")
    if os.path.isfile(_hp.data_train_path):
        data_dict,l0_dict,l1_dict,l2_dict = open_file(_hp.data_train_path)
    label0_dict_len = write_vocabulary(voc_path + path_label[0],l0_dict,word_freq)
    label1_dict_len = write_vocabulary(voc_path + path_label[1],l1_dict,word_freq)
    label2_dict_len = write_vocabulary(voc_path + path_label[2],l2_dict,word_freq)
    data_dict_len = write_vocabulary(voc_path + path_word,data_dict,word_freq)
    return [label0_dict_len,label1_dict_len,label2_dict_len],data_dict_len
def char_label_voc(word_freq,voc_path):
    path_word = _hp.voc_word
    path_label = _hp.voc_label
    label_size,voc_size = build_vocabulary(path_label,path_word,word_freq,voc_path)
    with open('hparamter.py', 'r') as file_r:
        code_lines = file_r.read().strip().decode('utf-8').split('\n')
        new_lines = []
        for line in code_lines:
            if 'word_nums' in line:
                line = line.split('=')
                line[1] = str(voc_size)
                line = '='.join(line)
            if 'class_0' in line:
                line = line.split('=')
                line[1] = str(label_size[0])
                line = '='.join(line)
            if 'class_1' in line:
                line = line.split('=')
                line[1] = str(label_size[1])
                line = '='.join(line)
            if 'class_2' in line:
                line = line.split('=')
                line[1] = str(label_size[2])
                line = '='.join(line)
            new_lines.append(line)
        with open('hparamter.py', 'w') as file_w:
            file_w.write('\n'.join(new_lines).encode('utf-8'))
if __name__ == '__main__':
    train_dir=['./models/','./vocs/']
    check_then_mkdir(train_dir)
    word_freq=1
    char_label_voc(word_freq,train_dir[1])
