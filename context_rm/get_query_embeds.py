import numpy as np
import math


def get_idf_weighted_query(q_ith, query_info, term_df, collection_len):
    idf_weighted_embeddings = []
    embeddings = query_info['embedding'][q_ith]
    query_terms = query_info['terms'][q_ith]
    normalizer = 0
    for i in range(len(query_terms)):
        q_t = query_terms[i]
        if q_t == '[SEP]' or q_t == '[CLS]':
            continue
            idf_weighted_embeddings.append(np.array(embeddings[i]))
            normalizer += 1
        elif q_t not in term_df:
            print(q_t + ' is not in the corpus!')
            continue
        else:
            idf_weight = math.log(collection_len / (term_df[q_t] + 0.5))
            idf_weighted_embeddings.append(idf_weight * np.array(embeddings[i]))
            normalizer += idf_weight

    idf_weighted_embeddings = np.array(idf_weighted_embeddings)
    idf_weighted_query_embedding = np.sum(idf_weighted_embeddings, axis=0) / float(normalizer)
    return idf_weighted_query_embedding


def get_avg_query_no_sep_cls(q_ith, query_info):
    term_embeddings_no_cls_sep = []
    embeddings = query_info['embedding'][q_ith]
    query_terms = query_info['terms'][q_ith]
    for i in range(len(query_terms)):
        q_t = query_terms[i]
        if q_t == '[SEP]' or q_t == '[CLS]':
            continue
        else:
            term_embeddings_no_cls_sep.append(np.array(embeddings[i]))

    term_embeddings_no_cls_sep = np.array(term_embeddings_no_cls_sep)
    avg_embeddings = np.sum(term_embeddings_no_cls_sep, axis=0) / float(len(term_embeddings_no_cls_sep))
    return avg_embeddings
