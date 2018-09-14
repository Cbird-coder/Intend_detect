# coding=utf-8
import tensorflow as tf
from data2ids import *
from intend_model import Model
from my_metrics import *
import numpy as np
import hparamter as _hp
import math
from tqdm import tqdm

def train():
    model = Model('train')
    model.build()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    datas = get_data_list()
    for epoch in range(_hp.epoch_num):
        
        data_seq,data_label0,data_label1 = gen_data(datas,mode='train')
        data_seq_val = []
        data_label0_val = []
        data_label1_val = []
        if _hp.val_size is not None:
            val_size = _hp.val_size * len(data_label0)
            data_seq_val,data_label0_val,data_label1_val = \
            data_seq[:val_size],data_label0[:val_size],data_label1[:val_size]

            data_seq,data_label0,data_label1 = \
            data_seq[val_size:],data_label0[val_size:],data_label1[val_size:]

        data_len = len(data_seq)
        val_len = len(data_seq_val)
        batch_step = int(math.ceil(float(data_len)/_hp.batch_size))
        batch_val_step = int(math.ceil(float(val_len)/_hp.batch_size))
        data_train = [data_seq,data_label0,data_label1]
        data_val = [data_seq_val,data_label0_val,data_label1_val]
        
        mean_loss = 0.0
        train_loss = 0.0
        step = 0
        train_tq = tqdm(batch_step)
        for step in train_tq:
            batch = gen_train_data(data_train,_hp.batch_size,step)
            if len(batch) == 0:
                break
            batch_data = data2ids(batch)
            _, loss, intent = model.step(sess, "train", batch_data)
            train_loss = train_loss + loss
        train_loss /= batch_step
        print("[Epoch {}] train set loss: {}".format(epoch, train_loss))
        #after each epoch,validate set
        if batch_val_step == 0:
            continue
        else:
            val_step = 0
            val_loss = 0.0
            val_tq = tqdm(batch_val_step)
            for val_step in val_tq:
                batch_val = gen_train_data(data_train,_hp.batch_size,val_step)
                if len(batch) == 0:
                    break
                batch_val_data =  data2ids(batch_val)
                val_loss = val_loss + model.step(sess, "val", batch_val_data)
            val_loss /= batch_val_step
            print("[Epoch {}] validation set loss: {}".format(epoch, val_loss))
def inference():
    model = Model('test')
    model.build()
    datas = get_data_list(mode='test')
    test_data = gen_data(datas,mode='test')
    test_len =  len(test_data)
    test_step = int(math.ceil(float(test_len)/_hp.batch_size))
    label0_out = []
    label1_out = []
    for step in tqdm(test_step):
        data_test = gen_inference_data(test_data,_hp.batch_size,step)
        data_test_ids = data2ids(data_test,mode='test')
        label0,label1 = model.step(sess, "test", data_test_ids)
        l0,l1 = id2label(label0,labe1)
        label0_out = label0_out + l0
        label1_out = label1_out + l1
    csv_write(datas,label0_out,label1_out)
if __name__ == '__main__':
    train()
