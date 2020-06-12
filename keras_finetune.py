import argparse
import keras
import os
from functools import partial
from datasets import get_flow_from_directory
from model_factory import build_model

# TODO: use argparse instead of global vars
parser = argparse.ArgumentParser()

parser.add_argument("--logs_dir", type=str, default="./logs")

parser.add_argument("--string", type=str, default="")
parser.add_argument("--int", type=int, default=0)
parser.add_argument("--float", type=float, default=.1)
parser.add_argument("--action", action="store_true", default=False)



logs_dir = "./logs"

# TRAINING PARAMS
epochs = 20
validation_freq = 1
early_stopping_patience = 3
# lr & optimizer & loss & metrics
learning_rate_start = 0.01
lr_factor = 0.1
lr_patience = 5
min_lr = 1e-4
# optimizer
optimizer_type = 'sgd'
optimizer_momentum = 0.9
optimizer_nesterov = False
# loss
label_smoothing = 0
# metrics
metrics_monitor = 'val_loss'

# DATASET PARAMS
TRAIN_DIR = 'data/arimgs/train'
VAL_DIR = 'data/arimgs/val'
CLASS_NAMES = ['nothing', 'other', 'close', 'left', 'right', 'ok']
TRAIN_BATCH_SIZE = 32
VAL_bATCH_SIZE = 32

# AUGMENTATION
train_rotation_range = 3
train_width_shift_range = 0.05
train_height_shift_range = 0.05
train_brightness_range = [0.7, 1.3]
train_shear_range = 3
train_zoom_range = [0.8, 1.0]

# MODEL PARAMS
MODEL_TYPE = 'mobilenet_v3_small'
NUM_CLASSES = 6
DROPOUT_RATE = 0.5
INPUT_IMG_SIZE = 224
# mobilenet v3
last_conv_ch = 1024


def build_optimizer(args):
    if optimizer_type == 'sgd':
        return keras.optimizers.SGD(learning_rate=learning_rate_start,
                                    momentum=optimizer_momentum,
                                    nesterov=optimizer_nesterov)
    raise ValueError("unknown optimizer {}".format(optimizer_type))


def build_model_and_preprocess_fn(args):
    model, preprocess_fn = build_model(
        model_type=MODEL_TYPE,
        num_classes=NUM_CLASSES,
        input_shape=(INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3),
        dropout_rate=DROPOUT_RATE,
        last_conv_ch=last_conv_ch,
        weights=None,
    )

    model.compile(
        optimizer=build_optimizer(),
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=label_smoothing,
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
        ],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
    )

    return model, preprocess_fn


def build_data_generator(preprocess_fn, args):
    preprocess_fn = partial(
        preprocess_fn,
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
    )
    train_generator, val_generator = get_flow_from_directory(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        classes=CLASS_NAMES,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_bATCH_SIZE,
        train_rotation_range=train_rotation_range,
        train_width_shift_range=train_width_shift_range,
        train_height_shift_range=train_height_shift_range,
        train_brightness_range=train_brightness_range,
        train_shear_range=train_shear_range,
        train_zoom_range=train_zoom_range,
        class_mode='categorical',
        preprocess_fn=preprocess_fn,
    )


def build_callbacks(args):
    callbacks = []
    # early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=metrics_monitor,  # 检测的变量
            min_delta=0,  # 变化多少才会触发
            patience=early_stopping_patience,  # 最多等待几轮
            verbose=0,  # 显示相关
            mode='auto',  # 是取最大值还是最小值，默认自动选择，取值范围[auto, min, max]
            baseline=None,  # 当检测的变量达到多少停止
            restore_best_weights=False  # 如果这次的效果没有最佳模型好，是否要restore最佳模型参数数值
        ))

    # model checkpoint
    filepath = os.path.join(logs_dir, "weights_{epoch:03d}-{val_loss:.4f}.h5")
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=metrics_monitor,  # 判断模型 best 的依据
            verbose=0,  # 显示相关
            save_best_only=False,  # 只保存最佳模型
            save_weights_only=False,  # 只保存模型参数，不保存结构
            mode='auto',  # 是取最大值还是最小值，默认自动选择，取值范围[auto, min, max]
            period=1  # 每过多少 epochs 保存一次模型
        ))

    # learning rate

    # # 自定义学习率方法一，通过函数返回学习率
    # def lr_schedule(epoch):  # epoch为整数，从0开始编号
    #     if epoch < 10:
    #         return 0.01
    #     elif epoch < 20:
    #         return 0.001
    #     else:
    #         return 0.0001
    # callbacks.append(keras.callbacks.LearningRateScheduler(
    #     schedule=lr_schedule
    # ))

    # 另外一种学习率衰减的方法：ReduceLROnPlateau
    # 某参数不再提升后降低学习率
    # 本方法还会用到 optimizers 中定义的 lr，将其作为初始lr
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=metrics_monitor,  # 监控的性能指标
            factor=lr_factor,  # 学习率衰减印字
            patience=lr_patience,  # 如果多久性能指标不提升就进行学习率衰减
            verbose=0,  # 展示相关
            mode='auto',  # 性能指标相关
            min_delta=1e-4,  # 性能指标变化的最小值（小于这个值就认为没有变化）
            cooldown=0,  # 相当于是warmup，先执行 cooldown epochs 后，在执行本callback的学习率优化策略
            min_lr=min_lr,  # 学习率最小值
        ))

    # tensorboard
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=0,
            batch_size=None,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None,
            update_freq='epoch',  # batch/epoch/int，int为任意整数，表示样本数量
        ))


def main(args):
    model, preprocess_fn = build_model_and_preprocess_fn(args)
    train_generator, val_generator = build_data_generator(preprocess_fn, args)
    callbacks = build_callbacks(args)
    model.fit_generator(train_generator,
                        steps_per_epoch=None,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=None,
                        validation_freq=validation_freq,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0)


if __name__ == '__main__':
    main(parser.parse_args())
