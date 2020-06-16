import os

import numpy as np
import tensorflow as tf

import tvm
import tvm.relay.testing
import tvm.relay.testing.tf as tf_testing
from tvm import relay
from tvm.contrib import graph_runtime

# target = tvm.target.cuda()
target = 'llvm'
network = 'mobilenet-v3-small-minimalistic'
log_file = "%s.log" % network
dtype = 'float32'

with tf.gfile.GFile('./tmp/mobilenet-v3-small-minimalistic.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(
            sess, 'sequential/Predictions/Softmax/Softmax')
shape_dict = {'input_2': (1, 224, 224, 3)}
layout = "NCHW"
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, params=params)

dtype = 'float32'
ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('input_2',
            tvm.nd.array(np.random.rand(1, 224, 224, 3).astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 6)), 'float32'))
print(tvm_output)
