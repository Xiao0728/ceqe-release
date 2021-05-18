import json


def weight_to_pretty_string(weights):
    weights_string = ""
    for w in range(len(weights)):
        weights_string += "%s=%s:" % (w, weights[w])
    weights_string = weights_string[:-1]  # remove the additional :
    return weights_string


def get_weighted_exp_query_galago_format(expansion_terms, orig_queries, orig_weight, num_exp_term, output):
    transformed_queries = {'queries': []}
    for i in range(len(expansion_terms)):
        transformed_query = {'number': '', 'text': ''}

        q_id = expansion_terms[i]['topicNumber']
        transformed_query['number'] = q_id

        q_exp_terms_weights = []
        q_exp_terms_words = []
        for t in range(num_exp_term):
            q_exp_terms_weights.append(expansion_terms[i]['terms'][t]['weight'])
            q_exp_terms_words.append(expansion_terms[i]['terms'][t]['word'])

        orig_query_galago_qlang = '#combine(%s)' % (orig_queries[int(q_id)])
        exp_query_galago_q_lang = '#combine:%s(%s)' % (
            weight_to_pretty_string(q_exp_terms_weights), ' '.join(q_exp_terms_words))

        query = orig_query_galago_qlang + ' ' + exp_query_galago_q_lang
        transformed_query_text = "#combine:%s( %s )" % (weight_to_pretty_string([orig_weight, 1 - orig_weight]), query)
        transformed_query['text'] = transformed_query_text
        #         print(transformed_query_text)
        transformed_queries['queries'].append(transformed_query)
    with open(output, 'w') as o:
        o.write(json.dumps(transformed_queries, indent=4))
