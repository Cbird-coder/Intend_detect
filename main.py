# coding=utf-8
import tensorflow as tf
from data2ids import *
from model_structure import Model
import numpy as np
import hparamter as _hp
import model_helper as _mh
import math
from tqdm import tqdm
import argparse as parse_pm
def train(task):
    datas = get_data_list()
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            model = Model(task,'train')
            model,curent_global_step = _mh.create_or_load_model(model,_hp.model_path,sess)
            writer_train = tf.summary.FileWriter(_hp.train_log_dir ,sess.graph)
            writer_val = tf.summary.FileWriter(_hp.val_log_dir,sess.graph)

            print('current global_step:',curent_global_step)
            for epoch in range(_hp.epoch_num):
                data_seq,data_label0,data_label1 = gen_data(datas,mode='train')
                data_seq_val = []
                data_label0_val = []
                data_label1_val = []
                if _hp.val_size is not None:
                    val_size = int(_hp.val_size * len(data_label0))
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
                train_tq = tqdm(range(batch_step))
                for step in train_tq:
                    batch = gen_train_data(data_train,_hp.batch_size,step)
                    if len(batch) == 0:
                        break
                    batch_data = data2ids(batch)
                    _, loss , lr, global_step,train_summary = model.model_exe(sess, "train", batch_data)
                    train_tq.set_description("batch loss: %f,learning rate: %f"%(loss,lr))
                    if step > 0 and step % _hp.save_step == 0:
                        model.saver.save(sess,_hp.model_path+'model.ckpt',global_step=global_step)
                    writer_train.add_summary(train_summary,global_step)
                    train_loss = train_loss + loss
                train_loss /= batch_step
                print("[Epoch {}] train set average loss: {}".format(epoch, train_loss))
                #after each epoch,validate set
                if batch_val_step == 0:
                    continue
                else:
                    val_step = 0
                    _loss = 0.0
                    val_tq = tqdm(range(batch_val_step))
                    for val_step in val_tq:
                        batch_val = gen_train_data(data_train,_hp.batch_size,val_step)
                        if len(batch) == 0:
                            break
                        batch_val_data =  data2ids(batch_val)
                        val_loss,val_summary = model.model_exe(sess, "val", batch_val_data)
                        val_tq.set_description("val loss: %f"%(val_loss))
                        writer_val.add_summary(val_summary,global_step)
                        _loss = _loss + val_loss
                    _loss /= batch_val_step
                    print("[Epoch {}] validation set average loss: {}".format(epoch, val_loss))
                model.saver.save(sess,_hp.model_path+'model.ckpt',global_step=global_step)
                eval(task,datas)

def eval(task,test_data):
    data_seq,data_label0,data_label1 = gen_data(test_data,mode='train')
    data_len = len(data_seq)
    eval_step = int(math.ceil(float(data_len)/_hp.batch_size))
    data_eval = [data_seq,data_label0,data_label1]
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            model = Model(task,'test')
            model,global_step = _mh.create_or_load_model(model,_hp.model_path,sess)
            eval_tq = tqdm(range(eval_step))
            l_total = 0.0
            for eval_step_it in eval_tq:
                    batch = gen_train_data(data_eval,_hp.batch_size,eval_step_it)
                    if len(batch) == 0:
                        break
                    batch_data = data2ids(batch)
                    data_sess = batch_data[0]
                    subject_label = batch_data[1]
                    sentiment_label = batch_data[2]
                    label = model.model_exe(sess, "eval", data_sess)
                    label_result = None
                    if task == 'subject':
                        label_result = subject_label
                    else:
                        label_result = sentiment_label
                    acc_rate = eval_process(label_result,label)
                    eval_tq.set_description("%s acc: %f"%(task,acc_rate))
                    l_total = l_total + acc_rate
            print("%s train set average acc: %f"%(task,l_total/float(eval_step)))

def inference(task):
    datas = get_data_list(mode='test')
    test_data = gen_data(datas,mode='test')
    test_len =  len(test_data)
    test_step = int(math.ceil(float(test_len)/_hp.batch_size))
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            model = Model(task,'test')
            model,global_step = _mh.create_or_load_model(model,_hp.model_path,sess)
            label_out = []
            for step in tqdm(range(test_step)):
                data_test = gen_inference_data(test_data,_hp.batch_size,step)
                data_test_ids = data2ids(data_test,mode='test')
                label = model.model_exe(sess, "test", data_test_ids)
                label = [label.values,label.indices] # TopKV2 to list
                l = id2label(label,task,type='test')
                label_out = label_out + l
            csv_write(datas,label_out,task)
            tf.train.write_graph(sess.graph_def,_hp.model_path,'sentiment.pbtxt')
if __name__ == '__main__':
    parser = parse_pm.ArgumentParser(description='arg parse')
    parser.add_argument('--mode',type=str,default=None)
    parser.add_argument('--task',type=str,default='subject')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.task)
    else:
        inference(args.task)