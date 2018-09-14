#coding=utf-8
#rnn相关参数
rnn_layer = 2
num_hidden = 128
output_keep_prob = 0.9
embedding_size = 64
time_major = False
#cnn相关参数
strides = 1
out_channels = 128
filter_sizes = [1,2,3]
#训练参数
batch_size = 10
epoch_num = 50 #迭代次数
#data set relate
val_size=0.2
#vocab relate
word_nums=2634
class_0=12
class_1=5
class_2=830
model_path='./models/'
voc_path='./vocs/'
voc_word='vocab.data'
voc_label=['vocab.lb0','vocab.lb1','vocab.lb2']
data_train_path='./dataset/train.csv'
data_test_path='./dataset/test_public.csv'