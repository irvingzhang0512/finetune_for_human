import argparse
import os

import tensorflow as tf

from datasets import get_flow_from_directory
from model_factory import build_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = True
tf.keras.backend.set_session(tf.Session(config=config))
parser = argparse.ArgumentParser()

# BASE PARAMS
parser.add_argument("--logs_dir", type=str, default="./logs")

# TRAINING PARAMS
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--validation_freq", type=int, default=1)
parser.add_argument("--early_stopping_patience", type=int, default=3)
# lr
parser.add_argument("--learning_rate_start", type=float, default=.01)
parser.add_argument("--lr_factor", type=float, default=.1)
parser.add_argument("--lr_patience", type=int, default=5)
parser.add_argument("--min_lr", type=float, default=1e-4)
# optimizer
parser.add_argument("--optimizer_type", type=str, default="sgd")
parser.add_argument("--optimizer_momentum", type=float, default=.9)
parser.add_argument("--use_optimizer_nesterov",
                    action="store_true",
                    default=False)
# loss
parser.add_argument("--label_smoothing", type=int, default=0)
# metrics
parser.add_argument("--metrics_monitor", type=str, default="val_loss")

# DATASET PARAMS
parser.add_argument("--train_dir", type=str, default="data/arimgs/train")
parser.add_argument("--val_dir", type=str, default="data/arimgs/val")
parser.add_argument(
    "--class_names",
    type=list,
    default=['nothing', 'other', 'close', 'left', 'right', 'ok'])
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--val_batch_size", type=int, default=32)

# AUGMENTATION
parser.add_argument("--train_rotation_range", type=int, default=5)
parser.add_argument("--train_width_shift_range", type=float, default=.1)
parser.add_argument("--train_height_shift_range", type=float, default=.1)
parser.add_argument("--train_brightness_range", type=list, default=[0.5, 1.5])
parser.add_argument("--train_shear_range", type=int, default=5)
parser.add_argument("--train_zoom_range", type=list, default=[0.6, 1.4])

# MODEL PARAMS
parser.add_argument("--model_type", type=str, default="mobilenet_v3_small")
parser.add_argument("--num_classes", type=int, default=6)
parser.add_argument("--dropout_rate", type=float, default=.5)
parser.add_argument("--input_img_size", type=int, default=224)
parser.add_argument(
    "--weights",
    type=str,
    default=
    "F:\\data\\keras\\weights_mobilenet_v3_small_minimalistic_224_1.0_float_no_top.h5"
)
# mobilenet v3
parser.add_argument("--minimalistic", action="store_true")


def build_optimizer(args):
    if args.optimizer_type == 'sgd':
        return tf.keras.optimizers.SGD(args.learning_rate_start,
                                       momentum=args.optimizer_momentum,
                                       nesterov=args.use_optimizer_nesterov)
    raise ValueError("unknown optimizer {}".format(args.optimizer_type))


def build_model_and_preprocess_fn(args):
    model, preprocess_fn = build_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        input_shape=(args.input_img_size, args.input_img_size, 3),
        dropout_rate=args.dropout_rate,
        weights=args.weights,
        minimalistic=args.minimalistic,
    )

    model.compile(
        optimizer=build_optimizer(args),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=args.label_smoothing,
        ),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
        ],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
    )

    return model, preprocess_fn


def build_data_generator(preprocess_fn, args):
    train_generator, val_generator = get_flow_from_directory(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        classes=args.class_names,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        train_rotation_range=args.train_rotation_range,
        train_width_shift_range=args.train_width_shift_range,
        train_height_shift_range=args.train_height_shift_range,
        train_brightness_range=args.train_brightness_range,
        train_shear_range=args.train_shear_range,
        train_zoom_range=args.train_zoom_range,
        class_mode='categorical',
        preprocess_fn=preprocess_fn,
    )
    return train_generator, val_generator


def build_callbacks(args):
    callbacks = []
    # early stopping
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=args.metrics_monitor,  # 检测的变量
            min_delta=0,  # 变化多少才会触发
            patience=args.early_stopping_patience,  # 最多等待几轮
            verbose=0,  # 显示相关
            mode='auto',  # 是取最大值还是最小值，默认自动选择，取值范围[auto, min, max]
            baseline=None,  # 当检测的变量达到多少停止
            restore_best_weights=False  # 如果这次的效果没有最佳模型好，是否要restore最佳模型参数数值
        ))

    # model checkpoint
    filepath = os.path.join(args.logs_dir,
                            "weights_{epoch:03d}-{val_loss:.4f}.h5")
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor=args.metrics_monitor,  # 判断模型 best 的依据
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
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=args.metrics_monitor,  # 监控的性能指标
            factor=args.lr_factor,  # 学习率衰减印字
            patience=args.lr_patience,  # 如果多久性能指标不提升就进行学习率衰减
            verbose=0,  # 展示相关
            mode='auto',  # 性能指标相关
            min_delta=1e-4,  # 性能指标变化的最小值（小于这个值就认为没有变化）
            cooldown=0,  # 相当于是warmup，先执行 cooldown epochs 后，在执行本callback的学习率优化策略
            min_lr=args.min_lr,  # 学习率最小值
        ))

    # tensorboard
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=args.logs_dir,
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
    return callbacks


def main(args):
    model, preprocess_fn = build_model_and_preprocess_fn(args)
    train_generator, val_generator = build_data_generator(preprocess_fn, args)
    callbacks = build_callbacks(args)
    model.fit_generator(train_generator,
                        steps_per_epoch=None,
                        epochs=args.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=None,
                        validation_freq=args.validation_freq,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0)


if __name__ == '__main__':
    main(parser.parse_args())
