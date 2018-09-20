import codecs
import random
import hparamter as _hp
import csv
from data_voc import clean_seq
import numpy as np

def load_vocs(vocab_file):
    if (vocab_file != ''):
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
    random.shuffle(datas)
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
    label1 = data_train[2]

    if batch_size * (step + 1) >= len(data):
        return data[batch_size * step : ],label0[batch_size * step : ],label1[batch_size * step : ]
    else:
        return data[batch_size * step : batch_size * (step + 1)],\
            label0[batch_size * step : batch_size * (step + 1)],\
            label1[batch_size * step : batch_size * (step + 1)]
def gen_inference_data(data_inference,batch_size,step):
    if batch_size * (step + 1) >= len(data_inference):
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
                dataid = dataid + [1] * (max_len - len(dataid))
            else:
                dataid = dataid
            data2ids.append(np.array(dataid))

            label0id = 0
            if label0_voc.has_key(label0[inx]):
                label0id = label0_voc[label0[inx]]
            label02ids.append(np.array(label0id))
            label1id = 0
            if label1_voc.has_key(label1[inx]):
                label1id = label1_voc[label1[inx]]
            label12ids.append(np.array(label1id))
            #print data2ids
        return np.array(data2ids),np.array(label02ids),np.array(label12ids)
    else:
        max_len = 0
        for data in batch_data:
            if len(data) > max_len:
                max_len = len(data)

        data2ids = []
        for item in batch_data:
            data2id = []
            for it in item:
                id_data = 0
                if data_voc.has_key(it):
                    id_data = data_voc[it]
                data2id.append(id_data)
            if len(data2id) < max_len:
                data2id = data2id + [1] * (max_len - len(data2id))
            else:
                data2id = data2id
            data2ids.append(data2id)
        return np.array(data2ids)

def eval_process(label,infer_label):
    acc = 0
    for inx,label_it in enumerate(label):
        if label_it == infer_label[inx]:
            acc = acc + 1
    acc_rate = float(acc)/float(len(label))
    return acc_rate

def id2label(labels,task,type='test'):
    acc,label =  labels
    _,_,_,_,label0voc_out,label1voc_out,_ = load_data_labels_voc()
    voc = None
    if task == 'subject':
        voc = label0voc_out
    else:
        voc = label1voc_out
    label_res = []
    if type == 'test_save':
        for inx,label_topK in enumerate(label):
            label_res.append(voc[label_topK[0]])
    else:
        for inx,label_topK in enumerate(label):
            acc_items = acc[inx]
            labels_inner = []
            for inx,label_item in enumerate(label_topK):
                labels_inner.append(voc[label_item]+':'+str(acc_items[inx]))
            label_res.append(';'.join(labels_inner))
    return label_res

def csv_write(datas,label,task):
    content_ids = []
    contents = []
    for data in datas:
        content_ids.append(data[0])
        contents.append(data[1])
    column_name = task
    csvfile = file('result.csv','wb')
    writer = csv.writer(csvfile)
    data = ['content_id','content',column_name]
    writer.writerow(data)
    for inx,data_id in enumerate(content_ids):
        contents[inx] = contents[inx].decode('utf-8')
        data = [data_id.encode('utf-8'),
                contents[inx].encode('utf-8'),
                label[inx].encode('utf-8')]
        writer.writerow(data)
    csvfile.close()
