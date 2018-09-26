import tensorflow as tf
import csv
import codecs
import re
import numpy as np
import heapq

subject_pb_path = "./models/subject.pb"
sentiment_pb_path = "./models/sentiment.pb"
data_voc_path = "./vocs/vocab.data"
subject_voc_path = "./vocs/vocab.lb0"
sentiment_voc_path = "./vocs/vocab.lb1"
#node name
input_node='encoder_inputs:0'
output_node='output_node:0'

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

def get_tensor_graph(model_pb_file):
    with tf.gfile.GFile(model_pb_file, "rb") as f:
        graph_o = tf.GraphDef()
        graph_o.ParseFromString(f.read())
    with tf.Graph().as_default() as G:
        tf.import_graph_def(graph_o,
                            input_map=None,
                            return_elements=None,
                            name='',
                            op_dict=None,
                            producer_op_list=None)
    return G
#data preprocess
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
    sentence = re.sub(ur"[^\u4e00-\u9fffA-Za-z0-9\-]","",sentence)
    sentence = sentence.lower()
    return sentence

def seq2ids(data_in,data_voc):
	data2id = []
	items = [item for item in data_in]
	for it in items:
		id_data = 0
		if data_voc.has_key(it):
			id_data = data_voc[it]
			data2id.append(id_data)
        return np.array([data2id])

def id2label(voc_out,label):
	prob,prob_indx = label[0],label[1]
	label_out = []
	for item in prob_indx:
		label_out.append(voc_out(item))
	return label_out,prob

class Model(object):
    """docstring for Model"""
    def __init__(self, task):
        super(Model, self).__init__()
        self.task = task
        model_path = None
        voc_path = None
        if self.task == 'subject':
            model_path = subject_pb_path
            voc_path = subject_voc_path
        elif self.task == 'sentiment':
            model_path = sentiment_pb_path
            voc_path = sentiment_voc_path
        else:
            raise ValueError
        graph = get_tensor_graph(model_path)
        self.sess = tf.Session(graph=graph)
        self.data_voc,_ = load_vocs(data_voc_path)
        self.voc,self.voc_out = load_vocs(voc_path)
        self.in_graph = self.sess.graph.get_tensor_by_name(input_node)
        self.out_graph = self.sess.graph.get_tensor_by_name(output_node)
    def run(self,data_in):
        to_tf = seq2ids(data_in,self.data_voc)
        output = self.sess.run(self.out_graph,feed_dict={self.in_graph:to_tf})
        return output[0]

global subject
global sentiment

subject = Model('subject')
sentiment = Model('sentiment')

test_path = 'test_public.csv'
csv_reader = csv.reader(open(test_path))

csvfile = file('result.csv','wb')
writer = csv.writer(csvfile)
data = ['content_id','content','subject','sentiment_value']
writer.writerow(data)

for inx,items in enumerate(csv_reader):
    if inx == 0:
        continue
    seq = clean_seq(items[1].strip())
    content_id = items[0]
    prob_sub = subject.run(seq)
    prob_senti = sentiment.run(seq)
    min_prob = 0.35
    subject_n = []
    sentimen_n = []
    for index,prob_sub_it in enumerate(prob_sub):
        if prob_sub_it >= min_prob:
            subject_n.append(subject.voc_out[index])
    for index,prob_senti_it in enumerate(prob_senti):
        if prob_senti_it >= min_prob:
            sentimen_n.append(sentiment.voc_out[index])

    for subject_it in subject_n:
        for sentiment_it in sentimen_n:
            data = [content_id.encode('utf-8'),\
                    items[1],\
                    subject_it.encode('utf-8'),\
                    sentiment_it.encode('utf-8')]
            writer.writerow(data)
csvfile.close()