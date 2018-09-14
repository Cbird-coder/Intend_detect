import tensorflow as tf
import numpy as np
import model_helper as _mh
import hparamter as _hp
import sys

class Model:
    def __init__(self,mode='train'):
        self.mode = mode
        self.encoder_inputs = tf.placeholder(tf.int32, [None,None],name='encoder_inputs')
        self.subject = tf.placeholder(tf.int32, [None],name='subject')
        self.sentiment_value = tf.placeholder(tf.int32, [None],name='sentiment_value')
        self.l2_reg = tf.contrib.layers.l2_regularizer(_hp.l2rate)
    def build(self):   
        with tf.variable_scope('lstm_layer'):
            self.embeddings = tf.Variable(tf.random_uniform([_hp.word_nums, _hp.embedding_size],
                                                        -0.01, 0.01), dtype=tf.float32, name="embedding")
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            #compute input data len
            len_temp = tf.sign(tf.add(tf.abs(tf.sign(self.encoder_inputs_embedded)),1))
            seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(len_temp,2)),1),tf.int32)
            self.batch_size = tf.size(seq_len)
            #rnn layer
            self.encoder_outputs,rnn_states = _mh.bi_rnns(self.mode,self.encoder_inputs_embedded,seq_len)
        with tf.variable_scope('cnn_layer'):
            self.encoder_outputs = tf.expand_dims(self.encoder_outputs,-1)
            print('input shape: {}'.format(encoder_outputs.get_shape().as_list()))
            print('rnn fw shape: {}'.format(rnn_states[0].get_shape().as_list()))
            print('rnn bw shape: {}'.format(rnn_states[1].get_shape().as_list()))
            pools_out = []
            for i in _hp.filter_sizes:
                conv = _mh.conv2d('cnn_filter'+str(i),self.encoder_outputs,i,2*_hp.num_hidden)
                bn_out = _mh.batch_norm(self.mode,'cnn_filter'+str(i),conv)
                relu_out = _mh.leaky_relu(bn_out)
                pool_out = _mh.max_pool(relu_out,i)
                print('cnn shape: {}'.format(pool_out.get_shape().as_list()))
                pools_out.append(pool_out)
        
        with tf.variable_scope('full_connect_layer'):
            #contact cnn output
            cnn_out_len = len(_hp.filter_sizes) * _hp.out_channels
            self.all_pool = tf.reshape(tf.concat(pools_out,3),[-1,cnn_out_len])
            print self.all_pool
            #get rnn last word output state
            output_data = tf.concat((tf.concat((rnn_states[0],self.all_pool),-1),rnn_states[-1]),-1)

            all_len = cnn_out_len + 2 * _hp.num_hidden

            suboutput_layer = tf.layers.Dense(_hp.class_0, use_bias=True,
                kernel_regularizer=self.l2_reg,bias_regularizer=self.l2_reg)
            subject_logits = suboutput_layer(output_data)
            self.subject_out = tf.argmax(subject_logits, axis=1,name='subject_out')

        cross_entropy0 = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.subject, depth=_hp.class_0, dtype=tf.float32),
            logits=subject_logits)
        self.subject_loss = tf.reduce_mean(cross_entropy0)
        #sentiment_value
        sensoutput_layer = tf.layers.Dense(_hp.class_1, use_bias=True,
                kernel_regularizer=self.l2_reg,bias_regularizer=self.l2_reg)
        sentiment_value_logits = sensoutput_layer(self.encoder_outputs)
        cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.sentiment_value, depth=_hp.class_1, dtype=tf.float32),
            logits=sentiment_value_logits)
        self.svalue_loss = tf.reduce_mean(cross_entropy1)
        tf.add_to_collection("l2loss",tf.add_n(tf.losses.get_regularization_loss())/tf.cast(self.batch_size,dtype=tf.float32))
        self.loss = self.subject_loss + self.svalue_loss
        if _hp.use_l2:
            self.loss = self.loss + tf.add_n(tf.get_collection('l2loss'))

        optimizer = tf.train.AdamOptimizer(_hp.learning_rate)
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

    def step(self, sess, mode, trarin_batch):
        data_seq,sub,svalue = trarin_batch
        if mode == 'train':
            output_feeds = [self.train_op, self.loss]
            feed_dict = {self.encoder_inputs: data_seq,self.subject: sub,self.sentiment_value:svalue}
        if mode == 'val':
            output_feeds = self.loss
            feed_dict = {self.encoder_inputs: data_seq,self.subject: sub,self.sentiment_value:svalue}
        if mode == 'test':
            output_feeds = [self.subject_out,self.sentiment_value_out]
            feed_dict = {self.encoder_inputs: data_seq}
        return sess.run(output_feeds, feed_dict=feed_dict)