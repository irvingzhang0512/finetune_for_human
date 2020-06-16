import os

import tensorflow as tf

import tvm
import tvm.relay.testing
import tvm.relay.testing.tf as tf_testing
from tvm import autotvm, relay
from tvm.autotvm.tuner import GATuner, GridSearchTuner, RandomTuner, XGBTuner

target = tvm.target.cuda()
# target = 'llvm'
layout = "NCHW"

input_tensor_shape = (1, 224, 224, 3)
output_tensor_name = 'sequential/Predictions/Softmax/Softmax'
input_tensor_name = 'input_2'
network = 'mobilenet-v3-small-minimalistic'
log_file = "%s.log" % network
pb_path = './data/pb/mobilenet-v3-small-minimalistic.pb'

tuning_option = {
    'log_filename':
    log_file,
    'tuner':
    'xgb',
    'n_trial':
    2000,
    'early_stopping':
    600,
    'measure_option':
    autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20,
                                   repeat=3,
                                   timeout=4,
                                   min_repeat_ms=150),
        # runner=autotvm.RPCRunner(
        #     '1080ti',  # change the device key to your key
        #     '0.0.0.0', 9190,
        #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial,
                                                         prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


with tf.gfile.GFile(pb_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(
            sess, output_tensor_name)
shape_dict = {input_tensor_name: input_tensor_shape}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod, target=target, params=params)

tasks = autotvm.task.extract_from_program(mod["main"],
                                          target=target,
                                          params=params,
                                          ops=(relay.op.get("nn.conv2d"), ))

tune_tasks(tasks, **tuning_option)
