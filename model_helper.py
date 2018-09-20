import tensorflow as tf
import hparamter as _hp
import time
import os

########rnn relate function######
def create_single_layer_rnn():
	return tf.nn.rnn_cell.LSTMCell(_hp.num_hidden, state_is_tuple=True)

def create_rnn_layers(mode):
	cells = []
	for i in range(_hp.rnn_layer):
		cell = create_single_layer_rnn()
		if mode=='train':
			cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=_hp.output_keep_prob)
		if i >= 1:
			cell = tf.contrib.rnn.ResidualWrapper(cell)
		cells.append(cell)

	rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
	return rnn_cells

def typical_rnns(mode,x,seq_len):

	rnn_cells = create_rnn_layers(mode)

	outputs, _ = tf.nn.dynamic_rnn(
                cell=rnn_cells,
                inputs=x,
                sequence_length=seq_len,
                dtype=tf.float32,
                time_major=_hp.time_major
            )
	return outputs

def bi_rnns(mode,x,seq_len):
	fw_cell = create_rnn_layers(mode)
	bw_cell = create_rnn_layers(mode)
	rnn_output,rnn_states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
													bw_cell,
													x,
													scope = 'bi_lstm',
													dtype = tf.float32,
													sequence_length = seq_len,
													time_major=_hp.time_major)
	rnn_output = tf.concat(rnn_output,axis=2)
	fw_c,fw_h = rnn_states[0][-1]
	bw_c,bw_h = rnn_states[1][-1]

	return rnn_output,(fw_h,bw_h)

#cnn layer relate code
def leaky_relu(x, leakiness=0.0):
	return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

#load model
def del_logdir(*logdir):
	for dir_log in logdir:
		os.unlink(dir_log)
def load_model(model, ckpt_path, session):
	start_time = time.time()
	try:
		model.saver.restore(session, ckpt_path)
	except tf.errors.NotFoundError as e:
		print("Can't load checkpoint")
		print("%s" % str(e))

	session.run(tf.tables_initializer())
	print("loaded model parameters from %s, time %.2fs" %(ckpt_path, time.time() - start_time))
	return model

def create_or_load_model(model, model_dir, session):
	latest_ckpt = tf.train.latest_checkpoint(model_dir)
	if latest_ckpt:
		model = load_model(model, latest_ckpt, session)
	else:
		start_time = time.time()
		session.run(tf.global_variables_initializer())
		session.run(tf.tables_initializer())
		print("created model with fresh parameters, time %.2fs" %(time.time() - start_time))

	global_step = model.global_step.eval(session=session)
	return model, global_step