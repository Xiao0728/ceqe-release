import numpy as np


def get_terms(query_info):
    # Groups query's tokens in a way to build unit of terms in the query
    # and groups the embedding vector of the aforementioned tokens

    queries_terms = [[] for i in range(len(query_info['id']))]
    queries_terms_embeds = [[] for i in range(len(query_info['id']))]

    # loop over all of the queries
    for i in range(len(query_info['id'])):
        # get different feature of each query
        q_id = query_info['id'][i]
        q_tokens = query_info['tokens'][i]
        q_embeds = query_info['embedding'][i]

        #     print(q_id, q_tokens)
        for j in range(len(q_tokens)):
            token = q_tokens[j]
            token_embeds = np.array(q_embeds[j]).reshape(-1, len(q_embeds[j]))
            if token.startswith('##'):
                prev_token = queries_terms[i].pop()
                # if the first token is remaining of the word in the previous passage add only the subtoken
                if prev_token == "[CLS]":
                    # print(token)
                    queries_terms[i].append("[CLS]")  # add [CLS] back
                    queries_terms[i].append(token)
                    queries_terms_embeds[i].append(token_embeds)
                    continue
                new_token = ''.join(token[2:])  # to delet the '##' substring in the token
                queries_terms[i].append(prev_token + new_token)
                prev_embeds = queries_terms_embeds[i].pop()
                term_embeds = np.append(prev_embeds, token_embeds, axis=0)
                queries_terms_embeds[i].append(term_embeds)

            else:
                queries_terms[i].append(token)
                queries_terms_embeds[i].append(token_embeds)
        #     print(len(queries_terms[i]))
        #     print(len(queries_terms_embeds[i]))
        assert len(queries_terms[i]) == len(queries_terms_embeds[i])
    #     print('-----------------')
    return queries_terms, queries_terms_embeds


def get_terms_embeds_pooled(queries_terms, queries_terms_embeds):
    # Pools the tokens in the embedding
    queries_terms_embeds_pooled = [[] for i in range(len(queries_terms_embeds))]

    for i in range(len(queries_terms)):
        q_terms = queries_terms[i]
        q_terms_embeds = queries_terms_embeds[i]
        for j in range(len(q_terms_embeds)):
            queries_terms_embeds_pooled[i].append(np.mean(q_terms_embeds[j], axis=0))
    return np.array(queries_terms_embeds_pooled)


def get_sip_weight(term, vocab_fq, tokenizer, collection_size, alpha):
    '''
    SIP weighting function: (alpha / (alpha + p(w)))
    :param term:
    :param vocab_fq:
    :param tokenizer:
    :param collection_size:
    :return:
    '''

    tokens = tokenizer.tokenize(term)

    tokens_sip_weight = []
    for t in tokens:
        # Todo ignore [CLS] and [SEP] for now
        if t == "[CLS]" or t == "[SEP]":
            token_prob = 0.9999
            tokens_sip_weight.append((alpha / (alpha + token_prob)))
            continue

        # remove the ##
        if t.startswith("##"):
            t = t[2:]
        if t in vocab_fq:
            token_prob = (float(vocab_fq[t]))/float(collection_size)
        else:
            token_prob = (100 / collection_size)# ToDo Change the 0.01?
            print(t)
        tokens_sip_weight.append((alpha / (alpha + token_prob)))

    return np.reshape(np.array(tokens_sip_weight), (-1,1))


def get_terms_embeds_sip_pooling(queries_terms, queries_terms_embeds, vocab_fq, collection_size, tokenizer, alpha=1e-4):
    """
    Get the term embedding representation using the SIP (alpha / alpha + p(token)) weighting function.
    :param queries_terms:
    :param queries_terms_embeds:
    :param vocab_fq:
    :param collection_size:
    :param tokenizer:
    :param alpha:
    :return:
    """

    queries_terms_embeds_pooled = [[] for i in range(len(queries_terms_embeds))]
    try:
        # iterate over the items
        for i in range(len(queries_terms)):
            q_terms = queries_terms[i]
            q_terms_embeds = queries_terms_embeds[i]

            # iterate over the terms embeddings [ [term1_token1, term1_token2, ...], [term2_token1, term2_token2, ...], ... ]
            for j in range(len(q_terms_embeds)):
                sip_weights = get_sip_weight(q_terms[j], vocab_fq, tokenizer, collection_size, alpha)
                sip_weighted_term_embeds = np.mean(q_terms_embeds[j] * sip_weights, axis=0) / np.sum(sip_weights)
                queries_terms_embeds_pooled[i].append(sip_weighted_term_embeds)
    except:
        # import IPython
        # IPython.embed()
        print("get_terms_embeds_sip_pooling ERROR!")
    return queries_terms_embeds_pooled

