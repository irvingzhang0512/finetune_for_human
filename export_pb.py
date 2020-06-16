import os

import tensorflow as tf
from tensorflow.python.framework import graph_io

from model_factory import build_model

tf.keras.backend.set_learning_phase(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = True
tf.keras.backend.set_session(tf.Session(config=config))

pb_path = "./data/pb/mobilenet-v3-small-minimalistic.pb"
ckpt_path = "./logs/logs-mobilenet-v3-small-minimalistic/weights_012-0.0475.h5"

model_type = "mobilenet_v3_small"
minimalistic = True

model, _ = build_model(model_type, minimalistic=minimalistic)
model.load_weights(ckpt_path)

session = tf.keras.backend.get_session()
output_node_names = [out.op.name for out in model.outputs]
with session.graph.as_default():
    graphdef_inf = tf.graph_util.remove_training_nodes(
        session.graph.as_graph_def())
    graphdef_frozen = tf.graph_util.convert_variables_to_constants(
        session, graphdef_inf, output_node_names)
    graph_io.write_graph(graphdef_frozen,
                         os.path.dirname(pb_path),
                         os.path.basename(pb_path),
                         as_text=False)
