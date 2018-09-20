import tensorflow as tf
import numpy as np
import model_helper as _mh
import hparamter as _hp
import sys

class Model:
    def __init__(self,task='subject',mode='train'):
        self.mode = mode
        self.task = task
        self.encoder_inputs = tf.placeholder(tf.int32, [None,None],name='encoder_inputs')
        if self.task == 'subject':
            self.label = tf.placeholder(tf.int32, [None])
        else:
            self.label = tf.placeholder(tf.int32, [None])
        self.global_step = tf.Variable(0, trainable=False)
        self.l2_reg = tf.contrib.layers.l2_regularizer(_hp.l2rate)

        encoder_outputs,rnn_states = self.encode_layer()
        logits = self.decoder_layer(encoder_outputs,rnn_states)
        softmax = tf.nn.softmax(logits,name='output_node')
        self.topKout = tf.nn.top_k(softmax,k=3,sorted=True)
        self.output = tf.argmax(softmax,1)
        if self.mode == 'train':
            self.loss = self.compute_loss(logits)
            self.train_op = self.op_train(self.loss)
        self.train_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=_hp.sava_time)

    def encode_layer(self):
        with tf.variable_scope('lstm_layer'):
            self.embeddings = tf.Variable(tf.random_uniform([_hp.word_nums, _hp.embedding_size],
                                                        -0.01, 0.01), dtype=tf.float32, name="embedding")
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            #compute input data len
            len_temp = tf.sign(tf.add(tf.abs(tf.sign(self.encoder_inputs_embedded)),1))
            seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(len_temp,2)),1),tf.int32)
            self.batch_size = tf.size(seq_len)
            #rnn layer
            encoder_outputs,rnn_states = _mh.bi_rnns(self.mode,self.encoder_inputs_embedded,seq_len)
            if _hp.time_major:
                encoder_outputs = tf.transpose(self.encoder_outputs,[1,0,2])
            return encoder_outputs,rnn_states

    def decoder_layer(self,encoder_outputs,rnn_states):
        with tf.variable_scope('cnn_layer'):
            print('cnn input shape: {}'.format(encoder_outputs.get_shape().as_list()))
            pools_out = []
            for kernel_size in _hp.kernel_sizes:
                conv = tf.layers.conv1d(encoder_outputs,_hp.out_size,kernel_size) 
                pool_out = tf.reduce_max(conv, reduction_indices=[1])
                print('cnn shape: {}'.format(pool_out.get_shape().as_list()))
                pools_out.append(pool_out)
            all_pool = tf.concat(pools_out,1)
        with tf.variable_scope('FC_layer'):
            print ('all pool shape: {}'.format(all_pool.get_shape().as_list()))
            #get rnn last word output state
            print('rnn fw state shape: {}'.format(rnn_states[0].get_shape().as_list()))
            print('rnn bw state shape: {}'.format(rnn_states[1].get_shape().as_list()))
            output_data = tf.concat((tf.concat((rnn_states[0],all_pool),-1),rnn_states[-1]),-1)
            print ('logits input shape: {}'.format(output_data.get_shape().as_list()))
            #fc1
            fc1 = tf.layers.dense(output_data,1024)
            if self.mode == 'train':
                fc1 = tf.contrib.layers.dropout(fc1, _hp.output_keep_prob)
            fc1 = _mh.leaky_relu(fc1)
            print ('subject_logits fc1 shape: {}'.format(fc1.get_shape().as_list()))
            #fc2
            if self.task == 'subject':
                output_label_num = _hp.class_0
            else:
                output_label_num = _hp.class_1
            output_layer = tf.layers.Dense(output_label_num, use_bias=True,
                kernel_regularizer=self.l2_reg,bias_regularizer=self.l2_reg)
            logits = output_layer(fc1)
            print('subject_logits shape: {}'.format(logits.get_shape().as_list()))
        return logits

    def compute_loss(self,logits):
        if self.task == 'subject':
            output_label_num = _hp.class_0
        else:
            output_label_num = _hp.class_1
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.label, depth=output_label_num, dtype=tf.float32),
            logits=logits)
        self.c_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy_loss', self.c_loss)

        #get l2 loss
        l2loss = tf.losses.get_regularization_losses()
        batch_l2 = tf.cast(self.batch_size,dtype=tf.float32)
        tf.add_to_collection("l2loss",tf.add_n(l2loss)/batch_l2)

        loss = self.c_loss
        if _hp.use_l2:
            loss = loss + tf.add_n(tf.get_collection('l2loss'))
        tf.summary.scalar('loss', loss)
        return loss

    def op_train(self,loss):
        #learning rate update
        self.lr = tf.train.exponential_decay(_hp.learning_rate,
                                  self.global_step,
                                  _hp.decay_steps,
                                  _hp.LEARNING_RATE_DECAY_FACTOR,
                                  staircase=False)
        tf.summary.scalar('learning_rate', self.lr)
        # Optimizer
        if _hp.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif _hp.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        else:
            raise ValueError("Unknown optimizer type %s" % _hp.optimizer)
        params = tf.trainable_variables()
        gradients = tf.gradients(loss,params,
          colocate_gradients_with_ops=_hp.colocate_gradients_with_ops)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, _hp.max_gradient_norm)

        opt_op = optimizer.apply_gradients(zip(clipped_grads, params),global_step=self.global_step)
        variable_averages = tf.train.ExponentialMovingAverage(_hp.MOVING_AVERAGE_DECAY, self.global_step)
        variables_averages_op = variable_averages.apply(params)
        with tf.control_dependencies([opt_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

    def model_exe(self, sess, mode, train_batch):
        if mode == 'test' or mode == 'eval':
            data_seq = train_batch
        else:
            data_seq,sub,svalue = train_batch
        feed_dict = {}
        if mode == 'train' or mode == 'val':
            if self.task == 'subject':
                feed_dict = {self.encoder_inputs: data_seq,self.label: sub}
            else:
                feed_dict = {self.encoder_inputs: data_seq,self.label: svalue}
        else:
            feed_dict = {self.encoder_inputs: data_seq}

        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.lr, self.global_step,self.train_summary]
        if mode == 'val':
            output_feeds = [self.loss,self.train_summary]
        if mode == 'eval':
            output_feeds = self.output
        if mode == 'test':
            output_feeds = self.topKout
        return sess.run(output_feeds, feed_dict=feed_dict)