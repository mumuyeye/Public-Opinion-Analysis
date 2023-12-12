import json
import pandas as pd

def gen_train_data(file_path, save_path):
    """
    file_path: 通过Label Studio导出的csv文件
    save_path: 保存的路径
    """
    data = pd.read_csv(file_path,encoding='utf-8')
    for idx, item in data.iterrows():
        text = item['text']
        if pd.isna(text):
            text = ''
        text_list = list(text)
        label_list = ['O' for _ in range(len(text_list))]
        
        labels = item['label']
        if not pd.isna(labels):
            labels = json.loads(labels)
            for label_item in labels:
                start = label_item['start']
                end = label_item['end']
                label = label_item['labels'][0]
                
                # Update label_list based on BIO format
                label_list[start] = f'B-{label}'
                label_list[start + 1:end] = [f'I-{label}' for _ in range(start + 1, end)]
        
        assert len(label_list) == len(text_list)
        
        with open(save_path, 'a',encoding='utf-8') as f:
            for idx_, (char, label) in enumerate(zip(text_list, label_list)):
                if char == '\t' or char == ' ':
                    char = ','
                line = f"{char} {label}\n"
                f.write(line)
            f.write('\n')
def convert_to_entity_relation_format(data):
    result = []

    for item in data:
        text = item['data']['text']
        annotations = item['annotations'][0]['result']
        print(annotations)
        # Create a dictionary to map entity IDs to entity text
        annotations_dict = {}
        for annotation in annotations:
            if 'from_id' in annotation and 'value' in annotation:
                annotations_dict[annotation['from_id']] = annotation['value']['text']

        entity_relations = []

        for annotation in annotations:
            if 'from_id' in annotation and 'to_id' in annotation and 'type' in annotation:
                relation = {
                    'from_entity': annotations_dict.get(annotation['from_id'], ''),
                    'to_entity': annotations_dict.get(annotation['to_id'], ''),
                    'relation_type': annotation['type'][0]  # 'type' instead of 'labels'
                }
                entity_relations.append(relation)

        result.append({
            'text': text,
            'entity_relations': entity_relations
        })

    return result

# Load data from a JSON file
with open('project-5-at-2023-12-01-17-41-3d2e6b49.json', 'r', encoding='utf-8') as file:
    your_data = json.load(file)

# Convert data to the desired format
converted_data = convert_to_entity_relation_format(your_data)

# Save the result to a TXT file
with open('output2.txt', 'w', encoding='utf-8') as file:
    for item in converted_data:
        file.write(f"Text: {item['text']}\n")
        file.write("Entity Relations:\n")
        for relation in item['entity_relations']:
            file.write(f"  From Entity: {relation['from_entity']}, To Entity: {relation['to_entity']}, Relation Type: {relation['relation_type']}\n")
        file.write("\n")


# Ner Example usage:
# gen_train_data('project-5-at-2023-12-08-22-50-447a8882.csv', 'output.txt')
