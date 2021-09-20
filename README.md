## Introduction
原神自动钓鱼AI由[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), DQN两部分模型组成。使用迁移学习，半监督学习进行训练。
模型也包含一些使用opencv等传统数字图像处理方法实现的不可学习部分。

其中YOLOX用于鱼的定位和类型的识别以及鱼竿落点的定位。DQN用于自适应控制钓鱼过程的点击，让力度落在最佳区域内。

## 准备
安装yolox

```shell
python setup.py develop
```

下载预训练[权重](https://github.com/7eu7d7/genshin_auto_fish/releases/tag/weights)


## YOLOX训练工作流程
YOLOX部分因为打标签太累所以用半监督学习。标注少量样本后训练模型生成其余样本伪标签再人工修正，不断迭代提高精度。
样本量较少所以使用迁移学习，在COCO预训练的模型上进行fine-tuning.

训练代码:
```shell
python yolox_tools/train.py -f yolox/exp/yolox_tiny_fish.py -d 1 -b 8 --fp16 -o -c weights/yolox_tiny.pth
```

## DQN训练工作流程
控制力度使用强化学习模型DQN进行训练。两次进度的差值作为reward为模型提供学习方向。模型与环境间交互式学习。

直接在原神内训练耗时较长，太累了。首先制作一个仿真环境，大概模拟钓鱼力度控制操作。在仿真环境内预训练一个模型。
随后将这一模型迁移至原神内，实现域间迁移。

仿真环境预训练代码:
```shell
python train_sim.py
```
原神游戏内训练:
```shell
python train.py
```

## 运行钓鱼AI
```shell
python fishing.py image -f yolox/exp/yolox_tiny_fish.py -c weights/best_tiny3.pth --conf 0.25 --nms 0.45 --tsize 640 --device gpu
```
