# 手势识别图片数据集构建

## 1. 数据来源
+ 之前录制的视频。
+ 开源数据集中选择：
  + EgoDexter
  + FreiHAND

## 2. 自己录制的视频抽帧
+ 通过opense对所有非`nothing`类别的视频帧进行过滤
  + 记录手部关键点数量大于15的图片，保存结果。
  + 使用的脚本是 `dataset_tools/arimgs/openpose_filter/action_to_images.py`。
    + 输入数据是 categoried/annotated 形式的 to_label.txt。
    + 输出也是一个txt文件，每行包括一个图片帧路径，以及对应的类别编号。
  + 通过 `dataset_tools/arimgs/copy_openpose_filter_results.ipynb` 将所有图片根据类别保存到 `./data/arimgs/openpose_results_raw` 中。
+ 人工工作：
  + 手工筛选 `./data/arimgs/openpose_results/raw` 中的图片，筛选结果保存到 `./data/arimgs/openpose_results_manual_filter` 中。
  + 手工筛选 EgoDexter 中的数据作为 `ohter` 类别。
  + 手工筛选 FreiHAND 中的数据。
+ 根据各类数据特点，选择图片保存到一个文件夹中，作为分类模型的输入。
  + 通过 `images-concat.ipynb` 实现。
  + 最后也对数据集进行拆分，得到 `train/val` 文件夹。
  + 需要注意的是：
    + 视频抽帧图片类似度太高，所以需要隔5帧获取。
    + **背景视频抽帧**也是在这一步中实现。


## 3. 数据集拆分结果
+ 训练集
nothing 5939
other 3754
close 2129
left 2088
right 2088
ok 2798


+ 验证集
nothing 659
other 417
close 236
left 231
right 231
ok 310