import json
import jieba
from string import punctuation
from flask import Flask
from flask import request
from flask import render_template
from model import E2EModel
from data_loader import to_tuple
import argparse
from tqdm import tqdm
from utils import extract_items, get_tokenizer

# app = Flask(__name__)

parser = argparse.ArgumentParser(description='Model Controller')
# parser.add_argument('--dataset', default='WebNLG', type=str, help='specify the dataset from ["NYT","WebNLG","ACE04","NYT10-HRL","NYT11-HRL","Wiki-KBP"]')
args = parser.parse_args()

def transform(input_text):
    # result_list = []
    text_without_punctuation = ''.join([char for char in input_text if char not in punctuation])
    word_list = jieba.lcut(text_without_punctuation)
    result_dict = {"text": ' '.join(word_list), "triple_list": []}
    # result_list.append(result_dict)
    # result = json.dumps(result_list, ensure_ascii=False, indent=1, separators=(',', ': '))
    return [result_dict]

def metric(subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False):
    orders = ['subject', 'relation', 'object'] 
    # results = []  # 保存所有结果的列表

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

        # results.append(result)

    return result

def extract_relations(data):
    result = json.loads(data)

    triple_list_pred = result.get("triple_list_pred", [])
    # extracted_relations = []
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
            # extracted_relations.append(relation_str)

    return extract_relation


# @app.route('/index')

# def index():
#     return render_template('index.html')

# @app.route('/api/getInfo')

def getInfo(test_data):
    # test_data = request.values.get("testdata")
    # print(test_data)
    data = transform(test_data)

    # for sent in data:
    #     to_tuple(sent)

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
    # subject_model = load_model('ckpts/subject_model.pkl')
    # object_model = load_model('ckpts/object_model.pkl')
    # hbt_model = load_model('hbt_model.pkl')
    num_rels = len(id2rel)
    subject_model, object_model, hbt_model = E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels)
    hbt_model.load_weights(save_weights_path)

    tokenizer = get_tokenizer(bert_vocab_path)

    isExactMatch = False

    result = metric(subject_model, object_model, data, id2rel, tokenizer, isExactMatch)

    result = extract_relations(result)

    return result

if __name__ == '__main__':
    # dataset = args.dataset
    bert_model = 'chinese_L-12_H-768_A-12'
    save_weights_path = 'saved_weights/' + 'MYDATA' + '/best_model.weights'
    bert_vocab_path = 'pretrained_bert_models/' + bert_model + '/vocab.txt'
    bert_config_path = 'pretrained_bert_models/' + bert_model + '/bert_config.json'
    bert_checkpoint_path = 'pretrained_bert_models/' + bert_model + '/bert_model.ckpt'
    # app.run(host='0.0.0.0', port=8848, debug=True)
    input = "习近平将出席并主持中国东盟建立对话关系30周年纪念峰会,外交部发言人华春莹19日宣布国家主席习近平将于11月22日在北京出席并主持中国东盟建立对话关系30周年纪念峰会,峰会将以视频方式举行。"
    output = getInfo(input)
    print(output)
    # with open('output.txt', 'w', encoding='utf-8') as f:
    #     f.write(output)