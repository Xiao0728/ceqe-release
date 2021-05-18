import tqdm
import json


def write_queries_list_tsv(queries, path):
    """
    Writes queries in a tsv format file.

    :param path: The path for the output file.
    :param queries: A list of query tuple (query_id, query_text)
    :return: --
    """
    with open(path, 'w') as output:
        for i in tqdm.tqdm(queries):
            output.write('%s\t%s\n' % (i[0], i[1]))


def write_queries_dict_tsv(queries, path):
    """
    Writes queries in a tsv format file.

    :param path: The path for the output file.
    :param queries: A list of query tuple (query_id, query_text)
    :return: --
    """
    with open(path, 'w') as output:
        for q in tqdm.tqdm(queries):
            output.write('%s\t%s\n' % (q, queries[q]))


def write_query_galago_format(queries, output):
    with open(output, 'w') as f:
        f.write('{\n\"queries\" : [\n')
        for q in queries:
            f.write('{\"number\": \"%s\", \"text\": \"%s\"},\n' % (q, queries[q]))
        f.write(']\n}')


def write_expansion_terms(queries, output):
    # sorted_keys = sorted([int(q) for q in queries.keys()])
    sorted_keys = queries.keys()
    new_structure = []
    for q in sorted_keys:
        new_dict = {}
        new_dict["topicNumber"] = str(q)
        new_dict["terms"] = []
        for tuple_i in queries[q]:
            new_dict["terms"].append({"word": tuple_i[0], "weight": tuple_i[1]})
        new_structure.append(new_dict)

    with open(output,'w') as o:
        o.write('[\n')
        for i in range(len(new_structure) - 1):
            o.write(json.dumps(new_structure[i]) + ',\n')
        o.write(json.dumps(new_structure[len(new_structure) - 1]) + '\n')
        o.write(']')
