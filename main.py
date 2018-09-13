# coding=utf-8
import tensorflow as tf
from data import *
from intend_model import Model
from my_metrics import *
import numpy as np
import hparamter as h_set
import math

def train():
    model = Model('train')
    model.build()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    data_ids,label_ids = data2id(mode='train')
    test_ids,test_label_ids = data2id(mode='test')
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
