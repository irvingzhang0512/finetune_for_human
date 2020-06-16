# Keras image classification fine-tune for Human

## 0. Overview
+ Target: Fine tune image classification models by keras.
  + Creating dataset by `ImageDataGenerator` and `flow_from_directory`
  + finetune models from `keras_applications`
  + deploy models to android and jetbot.
+ Feature
  + [x] img path dataset
  + [x] simple training scripts
  + [x] webcam demo
  + [x] keras to tvm
  + [x] keras to pb
+ TODO
  + [ ] finetune codes
  + [ ] eval codes


## 1. easy demo
+ Install:
  + `tensorflow-gpu==1.15.0`
  + newest version of keras-applications:
    + `pip uninstall keras_applications`
    + `git clone https://github.com/keras-team/keras-applications.git`
    + `cd keras-applications && python setup.py install`
+ 