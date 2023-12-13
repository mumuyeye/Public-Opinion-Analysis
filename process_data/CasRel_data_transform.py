import json
import jieba


def convert_format(data_list):
    converted_data_list = []

    for data in data_list:
        sentences = [' '.join(jieba.cut(sentence, cut_all=False)) for sentence in data["data"]["text"].split("\n") if
                     sentence.strip()]

        ner = []
        relations = []

        entity_map = {}

        rows = 1000
        cols = 1000
        array_2d = [['' for j in range(cols)] for i in range(rows)]

        for annotation in data["annotations"]:
            ner_annotation = []
            for entity in annotation["result"]:
                if "value" in entity and "start" in entity["value"]:
                    entity_id = entity["id"]
                    start = entity["value"]["start"]
                    end = entity["value"]["end"]
                    label = entity["value"]["labels"][0]
                    entity_map[entity_id] = {"start": start, "end": end}
                    array_2d[start][end] = entity["value"]["text"]
                    ner_annotation.append([start, end, label])
            ner.append(ner_annotation)

            # relation_annotation = []
            for relation in annotation["result"]:
                if "relation" in relation["type"]:
                    relation_type = relation["labels"]
                    from_id = relation["from_id"]
                    to_id = relation["to_id"]

                    if from_id in entity_map and to_id in entity_map:
                        from_start = entity_map[from_id]["start"]
                        from_end = entity_map[from_id]["end"]
                        to_start = entity_map[to_id]["start"]
                        to_end = entity_map[to_id]["end"]
                    if len(relation_type) == 1:
                        if relation_type[0] == '？':
                            relations.append([array_2d[from_start][from_end], "?", array_2d[to_start][to_end]])
                        else:
                            relations.append([array_2d[from_start][from_end], relation_type[0], array_2d[to_start][to_end]])
                    elif len(relation_type) == 2:
                        if relation_type[0] == '？':
                            relations.append([array_2d[from_start][from_end], "?", array_2d[to_start][to_end]])
                        else:
                            relations.append([array_2d[from_start][from_end], relation_type[0], array_2d[to_start][to_end]])
                        if relation_type[1] == '？':
                            relations.append([array_2d[from_start][from_end], "?", array_2d[to_start][to_end]])
                        else:
                            relations.append([array_2d[from_start][from_end], relation_type[1], array_2d[to_start][to_end]])

        # indexed_sentences = [{"index": int(data['id']), "sentence": sentence} for sentence in sentences]


        # 将分词后的句子连接成一个字符串
        sentences = ' '.join(sentences)

        converted_data = {
            # "clusters": [],
            "text": sentences,
            # "ner": ner,
            "triple_list": relations
            # "doc_key": f"your_doc_key_{data['id']}"  # Using data ID as part of the doc_key
        }

        converted_data_list.append(converted_data)

    return converted_data_list


def convert_to_json(obj):
    if isinstance(obj, int):
        return int(obj)
    raise TypeError


# 读取原始数据文件
input_file_path = "./input_data.json"  # 替换为实际的文件路径
with open(input_file_path, "r", encoding="utf-8") as input_file:
    input_data_list = json.load(input_file)

# 转换数据格式
converted_data_list = convert_format(input_data_list)

# 将字典转换为字符串，并删除逗号
# json_string = json.dumps(converted_data, ensure_ascii=False, indent=None, default=convert_to_json, separators=(',', ': '))for converted_data in converted_data_list
# json_string = json_string.replace(',\n', '\n')
# json_strings = [json_string]
# json_string = json.dumps(converted_data_list, ensure_ascii=False, indent=1, default=convert_to_json, separators=(',', ': '))
# json_string = json_string.replace(',\n', '\n')
# json_string = json_string.replace('"id":', '')

# output_file_path = "./output_data.json"  # 替换为实际的输出文件路径
# with open(output_file_path, "w", encoding="utf-8") as output_file:
    # json.dump(converted_data_list, output_file, ensure_ascii=False, indent=1, default=convert_to_json, separators=(',', ': '))
# 分割数据
train_data = converted_data_list
test_data = converted_data_list[:1000]
dev_data = converted_data_list[1000:1500]

# 写入 train_triples.json
with open("train_triples.json", "w", encoding="utf-8") as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=1, default=convert_to_json, separators=(',', ': '))

# 写入 test_triples.json
with open("test_triples.json", "w", encoding="utf-8") as test_file:
    json.dump(test_data, test_file, ensure_ascii=False, indent=1, default=convert_to_json, separators=(',', ': '))

# 写入 dev_triples.json
with open("dev_triples.json", "w", encoding="utf-8") as dev_file:
    json.dump(dev_data, dev_file, ensure_ascii=False, indent=1, default=convert_to_json, separators=(',', ': '))

print("Data split and saved successfully.")


# print(f"Data conversion completed. Results saved to {output_file_path}")
