# coding=utf-8
import tensorflow as tf
from data import *
from intend_model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np
import hparamter as h_set
import math

def train(is_debug=False):
    model = Model('train')
    model.build()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, intent2index, index2intent = get_info_from_training_data(train_data_ed)

    data_train,data_len = to_index(train_data_ed, word2index, intent2index)
    data_test,test_data_len = to_index(test_data_ed, word2index, intent2index)

    batch_step = int(math.ceil(float(data_len)/h_set.batch_size))
    batch_test_step = int(math.ceil(float(test_data_len)/h_set.batch_size))

    for epoch in range(h_set.epoch_num):
        mean_loss = 0.0
        train_loss = 0.0
        step = 0
        random.shuffle(data_train)
        while step < batch_step:
            batch = getBatch(h_set.batch_size, data_train,step)
            if len(batch) == 0:
                break
            _, loss, intent = model.step(sess, "train", batch)
            train_loss = train_loss + loss
            step = step + 1
        train_loss /= step
        print("[Epoch {}] Average train loss: {}".format(epoch, train_loss))
        # 每个epoch，测试一次
        intent_accs = []
        test_step = 0
        while test_step < batch_test_step:
            batch = getBatch(h_set.batch_size, data_test,test_step)
            if len(batch) == 0:
                break            
            intent = model.step(sess, "infer", batch)
            test_step = test_step + 1
            index = random.choice(range(len(batch)))

            intent_acc = accuracy_score(list(zip(*batch))[1], intent)
            intent_accs.append(intent_acc)
        print("Intent accuracy for epoch {}: {}".format(epoch, np.average(intent_accs)))

if __name__ == '__main__':
    train()
