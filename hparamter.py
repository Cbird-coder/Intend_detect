#coding=utf-8
#rnn相关参数
rnn_layer = 1
num_hidden = 64
output_keep_prob = 0.9
embedding_size = 64
time_major = False
#cnn相关参数
strides = 1
out_size = num_hidden
kernel_sizes = [1,3,5]
#训练参数
batch_size = 8
epoch_num = 1000 #迭代次数
#data set relate
val_size=0.1
test_rate=0.1
#vocab relate
max_len=186
word_nums=2629
class_0=10
class_1=3
class_2=827
model_path='./models/'
voc_path='./vocs/'
voc_word='vocab.data'
voc_label=['vocab.lb0','vocab.lb1','vocab.lb2']
data_train_path='./dataset/train.csv'
data_test_path='./dataset/test_public.csv'
#train param
l2rate = 0.00001
use_l2 = True
learning_rate = 0.005
save_step=100
sava_time=10
optimizer='adam'
max_gradient_norm=5
colocate_gradients_with_ops=True
decay_steps=500
LEARNING_RATE_DECAY_FACTOR=0.985
MOVING_AVERAGE_DECAY=0.999
train_log_dir = 'models/log/train/'
val_log_dir = 'models/log/val/'
test_log_dir = 'models/log/test/'
