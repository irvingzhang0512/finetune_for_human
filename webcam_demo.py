import cv2
import time
import numpy as np
import tensorflow as tf

from model_factory import build_model

class_names = ["nothing", "other", "close", "left", "right", "ok"]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = True
tf.keras.backend.set_session(tf.Session(config=config))
model_type = "mobilenet_v3_small"
ckpt_path = "./logs/weights_009-0.0587.h5"
history_length = 5
prob_threshold = 0.8


def main(args):
    model, preprocess_fn = build_model(
        model_type=model_type,
        minimalistic=True,
    )
    model.load_weights(ckpt_path)

    cap = cv2.VideoCapture(0)
    history = []
    t1 = time.time()
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        img = np.array(cv2.resize(frame, (224, 224)))
        probs = model.predict(
            preprocess_fn(img[:, :, ::-1].astype(np.float32).reshape(
                1, 224, 224, 3)))
        history.append(probs)
        history = history[-history_length:]
        cur_probs = np.sum(np.stack(history), axis=0) / len(history)
        if np.max(cur_probs) < prob_threshold:
            cur_id = 0
            cur_prob = 0
        else:
            cur_id = np.argmax(cur_probs)
            cur_prob = np.max(cur_probs)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        t2 = time.time()
        cur_fps = 1. / (t2 - t1)
        t1 = t2
        height, width, _ = frame.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
        cv2.putText(
            label, 'Prediction: ' + class_names[cur_id] + ", "
            '{:.1f} %'.format(cur_prob * 100.), (0, int(height / 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.2f} fps'.format(cur_fps),
                    (width - 170, int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        frame = np.concatenate((frame, label), axis=0)
        cv2.imshow('', frame)


if __name__ == '__main__':
    main(None)
