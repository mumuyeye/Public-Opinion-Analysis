# A Novel Cascade Binary Tagging Framework for Relational Triple Extraction

## Requirements

* tqdm
* codecs
* keras-bert = 0.80.0
* tensorflow-gpu = 1.13.1

在终端运行下面命令进行环境配置
```bash
pip install -r requirements.txt
```

## Usage

实验数据在data/MYDATA文件夹里，分为train_triples.json、dev_triples.json、test_triples.json三个文件。文件夹中trainsfrom.py可以实现将label-studio导出的json文件转换为以上三个文件的数据格式。

1. 训练模型

```shell
python run.py --train=True --dataset=MYDATA
```

2. 测试模型

```shell
python run.py --dataset=MYDATA
```
输出*Precision*、*recall*、*F1-score*等模型metric<br>
模型运行结果在results/MYDATA/test_results.json中。

```bash
python test.py
```
输入文本，得到<entity， entity， relation>三元组结果

3. 网页demo展示

```bash
python app.py
```


