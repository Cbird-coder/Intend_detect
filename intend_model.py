import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import model_helper as _mh
import hparamter as h_set
import sys

class Model:
    def __init__(self,mode='train'):
        self.mode = mode
        self.encoder_inputs = tf.placeholder(tf.int32, [None,h_set.time_step],name='encoder_inputs')
        self.intent_targets = tf.placeholder(tf.int32, [None],name='intent_targets')

    def build(self):   
        with tf.variable_scope('lstm_layer'):
            self.embeddings = tf.Variable(tf.random_uniform([h_set.vocab_size, h_set.embedding_size],
                                                        -0.1, 0.1), dtype=tf.float32, name="embedding")
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            #compute input data len
            len_temp = tf.sign(tf.add(tf.abs(tf.sign(self.encoder_inputs_embedded)),1))
            seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(len_temp,2)),1),tf.int32)
            #rnn layer
            encoder_outputs,rnn_states = _mh.bi_rnns(self.mode,self.encoder_inputs_embedded,seq_len)

        with tf.variable_scope('cnn_layer'):
            encoder_outputs = tf.expand_dims(encoder_outputs,-1)
            print('input shape: {}'.format(encoder_outputs.get_shape().as_list()))
            print('rnn fw shape: {}'.format(rnn_states[0].get_shape().as_list()))
            print('rnn bw shape: {}'.format(rnn_states[1].get_shape().as_list()))
            pools_out = []
            for i in h_set.filter_sizes:
                conv = _mh.conv2d('cnn_filter'+str(i),encoder_outputs,i,2*h_set.num_hidden)
                bn_out = _mh.batch_norm(self.mode,'cnn_filter'+str(i),conv)
                relu_out = _mh.leaky_relu(bn_out)
                pool_out = _mh.max_pool(relu_out,i)
                print('cnn shape: {}'.format(pool_out.get_shape().as_list()))
                pools_out.append(pool_out)
        
        with tf.variable_scope('full_connect_layer'):
            #contact cnn output
            cnn_out_len = len(h_set.filter_sizes) * h_set.out_channels
            self.all_pool = tf.reshape(tf.concat(pools_out,3),[-1,cnn_out_len])
            print self.all_pool
            #get rnn last word output state
            output_data = tf.concat((tf.concat((rnn_states[0],self.all_pool),-1),rnn_states[-1]),-1)

            all_len = cnn_out_len + 2 * h_set.num_hidden
            intent_W = tf.Variable(tf.random_uniform([all_len, h_set.label_size], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W")
            intent_b = tf.Variable(tf.zeros([h_set.label_size]), dtype=tf.float32, name="intent_b")

            intent_logits = tf.add(tf.matmul(output_data, intent_W), intent_b)
            self.intent = tf.argmax(intent_logits, axis=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.intent_targets, depth=h_set.label_size, dtype=tf.float32),
            logits=intent_logits)
        self.loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer()
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

    def step(self, sess, mode, trarin_batch):
        unziped = list(zip(*trarin_batch))
        data = np.array(unziped[0])
        label= np.array(unziped[1])
        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.intent]
            feed_dict = {self.encoder_inputs: data,self.intent_targets: label}
        if mode == 'infer':
            output_feeds = self.intent
            feed_dict = {self.encoder_inputs: data}

        return sess.run(output_feeds, feed_dict=feed_dict)