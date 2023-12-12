

# Chinese Named Entity Recognition using IDCNN

## Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

* jieba (>=0.37)


## Usage


### * Training:

1. 准备数据，存放在 data/ 目录下，包括训练数据 (example.train)，验证数据 (example.dev)，测试数据 (example.test)，以及中文字符嵌入数据 (vec.txt)。

2. 训练，具有在验证数据上取得最佳 F1 分数的模型将保存在 ckpt/ 目录下。


要使用 IDCNN+CRF 进行训练（默认），运行：

```bash
python3 main.py --train=True --clean=True --model_type=idcnn
```


### * Inference with command line input:

以下命令将在 /ckpt 目录下运行训练好的模型：

```bash
python3 main.py
```

