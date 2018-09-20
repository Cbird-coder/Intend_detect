import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import meta_graph
import os
import sys

model_name = sys.argv[1]

saved_graph_name = './models/sentiment.pbtxt'
saved_ckpt_name = './models/model.ckpt-34300'
out_node_name = 'output_node'

output_frozen_graph_name = model_name+'.pb'

freeze_graph.freeze_graph(input_graph=saved_graph_name,\
						input_saver='', \
						input_binary=False, \
						input_checkpoint=saved_ckpt_name, \
						output_node_names=out_node_name, \
						restore_op_name='', filename_tensor_name='', \
						output_graph=output_frozen_graph_name, \
						clear_devices=True, initializer_nodes='')

