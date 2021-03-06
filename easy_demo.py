import os
import time
from functools import partial

import numpy as np
import tensorflow as tf
from keras_applications import mobilenet_v3
from tensorflow.keras import layers

import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class_names = ["nothing", "other", "close", "left", "right", "ok"]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = True
tf.keras.backend.set_session(tf.Session(config=config))
history_length = 5
prob_threshold = 0.8

ckpt_path = "/hdd02/zhangyiyang/finetune_for_human/logs/logs-mobilenet-v3-small-minimalistic/weights_012-0.0475.h5"
in_video_path = "./data/arimgs/videos/input/ar.mp4"


def build_mobilenet_v3_small(
    num_classes=6,
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
    weights=None,
    minimalistic=True,
):
    backbone = mobilenet_v3.MobileNetV3Small(
        include_top=False,
        dropout_rate=0.5,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
        weights=weights,
        minimalistic=minimalistic,
    )
    classifier = tf.keras.models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Reshape((1, 1, 576)),
        layers.Dropout(dropout_rate),
        layers.Conv2D(num_classes,
                      kernel_size=1,
                      padding='same',
                      name='Logits'),
        layers.Flatten(),
        layers.Softmax(name='Predictions/Softmax')
    ])
    inputs = layers.Input(input_shape)
    x = backbone(inputs)
    x = classifier(x)
    model = tf.keras.Model(inputs, x, name="ar_mobilenet_v3_small")
    model.build(input_shape)

    preprocess_fn = partial(
        mobilenet_v3.preprocess_input,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
    )
    return model, preprocess_fn


def main(args):
    model, preprocess_fn = build_mobilenet_v3_small()
    model.load_weights(ckpt_path)

    cap = cv2.VideoCapture(in_video_path)
    history = []
    t1 = time.time()
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        img = np.array(cv2.resize(frame, (224, 224)))

        # 得到shape为 [1, 6] 的向量，表示每一类的概率
        probs = model.predict(
            preprocess_fn(img[:, :, ::-1].astype(np.float32).reshape(
                1, 224, 224, 3)))

        # 对history_length个历史记录累加求平均
        history.append(probs)
        history = history[-history_length:]
        cur_probs = np.sum(np.concatenate(history, axis=0),
                           axis=0) / len(history)
        # 设置阈值为 prob_threshold
        if np.max(cur_probs) < prob_threshold:
            cur_id = 0
            cur_prob = 0
        else:
            cur_id = np.argmax(cur_probs)
            cur_prob = np.max(cur_probs)

        # 结果可视化
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        t2 = time.time()
        cur_fps = 1. / (t2 - t1)
        t1 = t2
        height, width, _ = frame.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
        cv2.putText(
            label, class_names[cur_id] + ", "
            '{:.1f} %'.format(cur_prob * 100.), (0, int(height / 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.2f} fps'.format(cur_fps),
                    (width - 170, int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        frame = np.concatenate((frame, label), axis=0)
        cv2.imshow('', frame)


if __name__ == '__main__':
    main(None)
