import json


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as filep:
        json.dump(data, filep, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as filep:
        output = json.load(filep)
    return output
