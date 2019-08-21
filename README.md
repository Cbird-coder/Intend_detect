# RCNN_Intend_detect
本代码是对阿里巴巴的论文RCNN的实现与测试
其论文所在地址如下：
http://www.kdd.org/kdd2017/papers/view/a-hybrid-framework-for-text-modeling-with-convolutional-rnn


#usage:

#train model

python data_voc.py

python main.py --mode=train --task=subject

python main.py --mode=train --task=sentiment

#infer

python main.py --mode=test --task=subject

python main.py --mode=test --task=sentiment

#训练与测试语料说明

‘’语料使用汽车评测相关语料。
测试模型以及代码再demo文件夹下面
