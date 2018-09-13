import codecs
import hparamter as _hp
import csv
from data_voc import clean_seq

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
    return [data_voc,label0_voc,label1_voc,label2_voc],[label0voc_out,label1voc_out,label2voc_out]
def map_item2id(items, voc, max_len):
    '''
        Look up word or label dict,change sequence to dict number style 
    '''
    PADDING = 0
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

def data2id(mode='train'):
    data_path = ''
    if mode == 'train':
        data_path = _hp.data_train_path
    elif mode == 'test':
        data_path = _hp.data_test_path
    else:
        raise 'unknow mode...'
    data_seq = []
    data_label0 = []
    data_label1 = []
    csv_reader = csv.reader(open(data_path))
    for inx,items in enumerate(csv_reader):
        if inx == 0:
            continue
        if mode == 'train':
            seq = clean_seq(items[1].strip())
            label0 = clean_seq(items[2].strip())#class label
            label1 = items[3].strip()#-1,0,1
            label2 = clean_seq(items[4].strip())#key words
        else:
            seq = clean_seq(items[1].strip())
        if len(seq) ==0:
            print items
            continue
