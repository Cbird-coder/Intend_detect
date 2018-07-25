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
#训练数据相关参数，请不要修改，此处修改会出错
time_step = 50
vocab_size = 871
label_size = 22