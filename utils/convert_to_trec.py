import json
import tqdm


def convert_exp_terms_to_trec_format(exp_terms_json, method, output):
    exp_terms = json.load(open(exp_terms_json))
    with open(output, 'w') as o:
        # iterate over queries
        for i in range(len(exp_terms)):
            q_id = exp_terms[i]['topicNumber']
            terms = exp_terms[i]['terms']
            for j in range(len(terms)):
                word = terms[j]['word']
                score = terms[j]['weight']
                rank = j + 1
                o.write('%s Q0 %s %s %s %s\n' % (q_id, word, rank, score, method))


def convert_labels_to_qrels(labels, output):
    with open(output, 'w') as o:
        with open(labels) as f:
            for line in tqdm.tqdm(f):
                qrel = json.loads(line)
                q_id = qrel['expansion']['queryId']
                terms = qrel['expansion']['expansionTerms']
                term = terms[0]['word']
                label = qrel['label']
                o.write('%s 0 %s %s\n' % (q_id, term, label))
