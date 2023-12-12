## A Novel Cascade Binary Tagging Framework for Relational Triple Extraction

### 实验环境

- tqdm
- codecs
- keras-bert = 0.80.0
- tensorflow-gpu = 1.13.1

### 实验运行

实验数据在data/MYDATA文件夹里，分为train_triples.json、dev_triples.json、test_triples.json三个文件。文件夹中trainsfrom.py可以根据label-studio导出的json文件得到以上三个文件。

1. 训练模型

```shell
python run.py --train=True --dataset=MYDATA
```

2. 测试模型

```shell
python run.py --dataset=MYDATA
```

模型运行结果在results/MYDATA/test_results.json中。
