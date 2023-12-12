import json
import jieba
from string import punctuation
from flask import Flask
from flask import request
from flask import render_template
from model import E2EModel
import argparse
from tqdm import tqdm
from utils import extract_items, get_tokenizer

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Model Controller')
args = parser.parse_args()

def transform(input_text):
    text_without_punctuation = ''.join([char for char in input_text if char not in punctuation])
    word_list = jieba.lcut(text_without_punctuation)
    result_dict = {"text": ' '.join(word_list), "triple_list": []}
    return [result_dict]

def metric(subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False):
    orders = ['subject', 'relation', 'object'] 

    for line in tqdm(iter(eval_data)):
        Pred_triples = set(extract_items(subject_model, object_model, tokenizer, line['text'], id2rel))
        Gold_triples = set(line['triple_list'])    

        result = json.dumps({
            'text': line['text'],
            'triple_list_gold': [
                dict(zip(orders, triple)) for triple in Gold_triples
            ],
            'triple_list_pred': [
                dict(zip(orders, triple)) for triple in Pred_triples
            ],
            'new': [
                dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
            ],
            'lack': [
                dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
            ]
        }, ensure_ascii=False, indent=4) + '\n'

    return result

def extract_relations(data):
    result = json.loads(data)
    triple_list_pred = result.get("triple_list_pred", [])

    extract_relation = ""
    for triple in triple_list_pred:
        subject = triple.get("subject", "")
        relation = triple.get("relation", "")
        object_ = triple.get("object", "")

        if relation:
            relation_str = "<" + subject + ', ' + object_ + ', ' + relation + ">"
            if triple == 0:
                extract_relation = relation_str
                continue
            extract_relation = extract_relation + '|' + relation_str 

    return extract_relation

@app.route('/index')

def index():
    return render_template('index.html')

@app.route('/api/getInfo')

def getInfo():
    test_data = request.values.get("testdata")
    print(test_data)
    data = transform(test_data)

    id2rel = {
    0: "国家间",
    1: "机构间",
    2: "国家-机构",
    3: "国家-人物",
    4: "机构-人物",
    5: "+",
    6: "-",
    7: "?"
    }

    LR = 1e-5
    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)
    subject_model, object_model, hbt_model = E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels)
    hbt_model.load_weights(save_weights_path)

    id2rel = {int(i): j for i, j in id2rel.items()}

    tokenizer = get_tokenizer(bert_vocab_path)

    isExactMatch = False

    result = metric(subject_model, object_model, data, id2rel, tokenizer, isExactMatch)

    result = extract_relations(result)

    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    bert_model = 'chinese_L-12_H-768_A-12'
    bert_vocab_path = 'pretrained_bert_models/' + bert_model + '/vocab.txt'
    save_weights_path = 'saved_weights/' + 'MYDATA' + '/best_model.weights'
    bert_config_path = 'pretrained_bert_models/' + bert_model + '/bert_config.json'
    bert_checkpoint_path = 'pretrained_bert_models/' + bert_model + '/bert_model.ckpt'
    app.run(host='127.0.0.1', port=5000, debug=True)