import json
import tqdm
import numpy as np


def read_json_by_line(path, keys):
    data = {}
    for k in keys:
        data[k] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == '[' or line == ']':
                # print(line)
                continue
            if line.endswith(','):
                line = line[:-1]
            input_dict = json.loads(line)
            for k in keys:
                data[k].append(input_dict[k])

    return data


def read_json_by_line_selective(f, keys, ids):
    data = {}
    for k in keys:
        data[k] = []

    for line in tqdm.tqdm(f):
        line = line.strip()
        if line == '[' or line == ']':
            print(line)
            continue
        if line.endswith(','):
            line = line[:-1]
        input_dict = json.loads(line)
        if input_dict['id'] in ids:
            for k in keys:
                data[k].append(input_dict[k])