import codecs
import random
import hparamter as _hp
import csv
from data_voc import clean_seq
import numpy as np

def load_vocs(vocab_file):
    if (src_vocab_file != ''):
        with codecs.open(vocab_file, 'r', encoding='utf-8') as src_f:
            src_vocab_lines = src_f.readlines()
            src_temp_vocab = {}
            for line in src_vocab_lines:
                line = line.strip()
                if (line.endswith(u'\n')):
                    line = line[:-1]
                src_temp_vocab[line] = len(src_temp_vocab)
            src_datas_vocab = src_temp_vocab
            del src_temp_vocab

            temp_rev_vocab = {}
            for (i, j) in zip(src_datas_vocab.keys(), src_datas_vocab.values()):
                temp_rev_vocab[j] = i
            src_rev_vocab = temp_rev_vocab
            del temp_rev_vocab

    return src_datas_vocab,src_rev_vocab
def load_data_labels_voc():
    data_voc,_ = load_vocs(_hp.voc_path + _hp.voc_word)
    label0_voc,label0voc_out = load_vocs(_hp.voc_path + _hp.voc_label[0])
    label1_voc,label1voc_out = load_vocs(_hp.voc_path + _hp.voc_label[1])
    label2_voc,label2voc_out = load_vocs(_hp.voc_path + _hp.voc_label[2])
    return data_voc,label0_voc,label1_voc,label2_voc,label0voc_out,label1voc_out,label2voc_out
def map_item2id(items, voc, max_len):
    '''
        Look up word or label dict,change sequence to dict number style 
    '''
    PADDING = 1
    arr = []
    for i in range(len(items)):
        if voc.has_key(items[i]):
            data_id = voc[items[i]]
        else:
            data_id = voc['<unk>']
        arr.append(data_id)
    if len(arr) < max_len:
        arr = arr + [PADDING]*(max_len-len(arr)) + [PADDING]
    else:
        arr = arr + [PADDING]
    return arr

def get_data_list(mode='train'):
    data_path = ''
    if mode == 'train':
        data_path = _hp.data_train_path
    elif mode == 'test':
        data_path = _hp.data_test_path
    else:
        raise 'unknow mode...'
    datas = []
    csv_reader = csv.reader(open(data_path))
    for inx,items in enumerate(csv_reader):
        if inx == 0:
            continue
        else:
            datas.append(items)
    return datas

def gen_data(datas,mode='train'):
    
    random.shuffle(datas)
    
    data_seq = []
    data_label0 = []
    data_label1 = []
    for items in datas:
        if mode == 'train':
            seq = clean_seq(items[1].strip())
            seq = [item for item in seq]
            label0 = clean_seq(items[2].strip())#class label
            label1 = clean_seq(items[3].strip())#-1,0,1
            label2 = clean_seq(items[4].strip())#key words
            data_seq.append(seq)
            data_label0.append(label0)
            data_label1.append(label1)
        else:
            seq = clean_seq(items[1].strip())
            seq = [item for item in seq]
            data_seq.append(seq)

    if mode == 'train':
        return data_seq,data_label0,data_label1
    elif mode == 'test':
        return data_seq
    else:
        raise ValueError

def gen_train_data(data_train,batch_size,step):
    data = data_train[0]
    label0 = data_train[1]
    label1 = data_train[1]

    if batch_size * (step + 1) >= len(data):
        return data[batch_size * step:],label0[batch_size * step:],label1[batch_size * step:]
    else:
        return data[batch_size * step:batch_size * (step + 1)],\
            label0[batch_size * step:batch_size * (step + 1)],\
            label1[batch_size * step:batch_size * (step + 1)]
def gen_inference_data(data_inference,batch_size,step):
    if batch_size * (step + 1) >= len(data):
        return data_inference[batch_size * step:]
    else:
        return data_inference[batch_size * step:batch_size * (step + 1)]

def data2ids(batch_data,mode='train'):
    data_voc,label0_voc,label1_voc,label2_voc,label0voc_out,label1voc_out,label2voc_out = load_data_labels_voc()
    if mode == 'train':
        data = batch_data[0]
        label0 = batch_data[1]
        label1 = batch_data[2]
        max_len = 0
        for data_it in data:
            if len(data_it) > max_len:
                max_len = len(data_it)
        data2ids = []
        label02ids = []
        label12ids = []
        for inx,data_it in enumerate(data):
            dataid=[]
            for item in data_it:
                id_data = 0
                if data_voc.has_key(item):
                    id_data = data_voc[item]
                dataid.append(id_data)
            if len(dataid) < max_len:
                dataid = dataid + [0] * (max_len - len(dataid))
            data2ids.append(dataid)

            label0id = 0
            if label0_voc.has_key(label0[inx]):
                label0id = label0_voc[label0[inx]]
            label02ids.append(label0id)
            label1id = 0
            if label1_voc.has_key(label1[inx]):
                label1id = label1_voc[label1[inx]]
            label12ids.append(label1id)
        return np.array(data2ids),np.array(label02ids),np.array(label12ids)
    else:
        max_len = 0
        for data in batch_data:
            if len*(data) > max_len:
                max_len = len*(data)

        data2ids = []
        for item in batch_data:
            data2id = []
            for it in item:
                id_data = 0
                if data_voc.has_key(it):
                    id_data = data_voc[it]
                data2id.append(id_data)
            if len(data2id) < max_len:
                data2id = data2id + [0] * (max_len - len(data2id))
            data2ids.append(data2id)
        return np.array(data2ids)
def id2label(*label):
    label0 =  label[0]
    label1 =  label[1]
    _,_,_,_,label0voc_out,label1voc_out,_ = load_data_labels_voc()
    label0s = []
    label1s = []
    for inx,label0_item in enumerate(label0):
        label0s.append(label0voc_out[label0_item])
        label1s.append(label0voc_out[label1[inx]])
    return label0s,label1s
def csv_write(datas,label0,label1):
    content_ids = []
    contents = []
    for data in datas:
        content_ids.append(data[0])
        contents.append(data[1])
    csvfile = file('result.csv','wb')
    writer = csv.writer(csvfile)
    data = [('content_id','content','subject','sentiment_value')]
    writer.writerow(data)
    for inx,data_id in enumerate(content_ids):
        data = [(data_id,contents[inx],label0[inx],label1[inx])]
        writer.writerow(data)
    csvfile.close()
