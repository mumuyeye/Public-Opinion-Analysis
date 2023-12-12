# Chinese Relation Extraction by biGRU with Character and Sentence Attentions



![](http://www.crownpku.com/images/201708/1.jpg)



![](http://www.crownpku.com/images/201708/2.jpg)





## Requrements

- Python (>=3.5)
- TensorFlow (>=r1.0)
- scikit-learn (>=0.18)

## Usage

###  * Training:

1. 准备数据，存放在 data/ 目录下，包括训练数据 (example.train)，验证数据 (example.dev)，测试数据 (example.test)，以及中文字符嵌入数据 (vec.txt)。

```
国家间 0
机构间 1
国家-机构 2
机构-人物 3
国家-人物 4
+ 5
- 6
? 7
```

2. 将数据整理成.npy文件，文件将保存在data/目录下。

```python
python3 initial.py
```

3. 使用以下命令进行模型训练，训练后的模型将保存在model/目录下。

```python
python3 train_GRU.py
```

### * Inference

如果你训练了一个新模型，请记得在test_GRU.py文件的`main_for_evaluation()`和`main()`函数中修改路径名，使用你自己的模型名称。

```python
python3 test_GRU.py
```







