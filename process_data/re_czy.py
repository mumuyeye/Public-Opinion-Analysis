import json

def extract_relationships(data):
    relationships = []

    for annotation_task in data:
        text = annotation_task['data']['text']
        annotations = annotation_task['annotations'][0]['result']

        for annotation in annotations:
            if annotation['type'] == 'relation':
                entity1_id = annotation.get('from_id')
                entity2_id = annotation.get('to_id')

                entity1 = next((v['value']['text'] for v in annotations if v['id'] == entity1_id), None)
                entity2 = next((v['value']['text'] for v in annotations if v['id'] == entity2_id), None)

                relationship_type = annotation.get('labels', [])[0]  # Assuming each relation has a single label

                # Use the original text provided in the data
                sentence = text.strip()

                relationships.append({
                    'entity1': entity1,
                    'entity2': entity2,
                    'relationship_type': relationship_type,
                    'sentence': sentence
                })

    return relationships

# Read data from a JSON file
with open('project-5-at-2023-12-01-18-56-3d2e6b49.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

relationships = extract_relationships(data)

output_file_path = 'output2.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    max_entity1_width = max(len(relationship['entity1']) for relationship in relationships)
    max_entity2_width = max(len(relationship['entity2']) for relationship in relationships)
    max_relationship_type_width = max(len(relationship['relationship_type']) for relationship in relationships)
    max_sentence_width = max(len(relationship['sentence']) for relationship in relationships)
    print(max_sentence_width)

    for relationship in relationships:
        output_file.write(f"{relationship['entity1']:<{max_entity1_width}} {relationship['entity2']:<{max_entity2_width}} {relationship['relationship_type']:<{max_relationship_type_width}} {relationship['sentence']:<{max_sentence_width}}\n")

print(f"Results written to {output_file_path}")
