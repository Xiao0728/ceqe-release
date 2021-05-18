import numpy as np
from scipy import spatial


def get_similarity_queryTerms_and_docsTerms(query_ith, queries_info, top_k_ret_docs_info_q_ith,
                                            set_top_k_ret_docs_id_q_ith):
    similarity_q = {}
    q_id = queries_info['id'][query_ith]
    query_terms_vec = np.array(queries_info['embedding'][query_ith])
    for q_t in range(len(query_terms_vec)):
        term_vec = query_terms_vec[q_t]
        for j in range(len(top_k_ret_docs_info_q_ith['id'])):
            doc_id = top_k_ret_docs_info_q_ith['id'][j]
            # To look only in the topk retrieved docs
            if doc_id in set_top_k_ret_docs_id_q_ith:
                doc_terms_vec = np.array(top_k_ret_docs_info_q_ith['embedding'][j])
                for d_t in range(len(doc_terms_vec)):
                    term = top_k_ret_docs_info_q_ith['terms'][j][d_t]
                    doc_term_vec = doc_terms_vec[d_t]
                    sim = 1 - spatial.distance.cosine(term_vec, doc_term_vec)
                    q_t_mention = queries_info['terms'][query_ith][q_t]
                    similarity_q.setdefault(q_t_mention, {})
                    similarity_q[q_t_mention].setdefault(doc_id, []).append((term, sim))
    return similarity_q


def get_similarity_query_and_docTerms(query_ith, queries_info, top_k_ret_docs_info_q_ith,
                                      set_top_k_ret_docs_id_q_ith):
    similarity_q = {}
    q_id = queries_info['id'][query_ith]
    query_vec = np.array(queries_info['embedding'][query_ith])
    for doc_j in range(len(top_k_ret_docs_info_q_ith['id'])):
        doc_id = top_k_ret_docs_info_q_ith['id'][doc_j]
        # To look only in the topk retrieved docs
        if doc_id in set_top_k_ret_docs_id_q_ith:
            doc_terms_vec = np.array(top_k_ret_docs_info_q_ith['embedding'][doc_j])
            for d_t in range(len(doc_terms_vec)):
                doc_term = top_k_ret_docs_info_q_ith['terms'][doc_j][d_t]
                doc_term_vec = doc_terms_vec[d_t]
                sim = 1 - spatial.distance.cosine(query_vec, doc_term_vec)
                similarity_q.setdefault(doc_id, []).append((doc_term, sim))
    return similarity_q


def get_similarity_query_and_docTerms_CAR(query_info, top_k_ret_docs_info_q_ith):
    similarity_q = {}
    q_id = query_info['id']
    query_vec = np.array(query_info['embedding'])
    for doc_j in range(len(top_k_ret_docs_info_q_ith['id'])):
        doc_id = top_k_ret_docs_info_q_ith['id'][doc_j]
        # To look only in the topk retrieved docs
        doc_terms_vec = np.array(top_k_ret_docs_info_q_ith['embedding'][doc_j])
        for d_t in range(len(doc_terms_vec)):
            doc_term = top_k_ret_docs_info_q_ith['terms'][doc_j][d_t]
            doc_term_vec = doc_terms_vec[d_t]
            sim = 1 - spatial.distance.cosine(query_vec, doc_term_vec)
            similarity_q.setdefault(doc_id, []).append((doc_term, sim))
    return similarity_q


def get_similarity_queryTerms_and_docsTerms_CAR(query_info, top_k_ret_docs_info_q_ith):
    similarity_q = {}
    q_id = query_info['id']
    query_terms_vec = np.array(query_info['embedding'])
    for q_t in range(len(query_terms_vec)):
        term_vec = query_terms_vec[q_t]
        for j in range(len(top_k_ret_docs_info_q_ith['id'])):
            doc_id = top_k_ret_docs_info_q_ith['id'][j]
            # To look only in the topk retrieved docs
            doc_terms_vec = np.array(top_k_ret_docs_info_q_ith['embedding'][j])
            for d_t in range(len(doc_terms_vec)):
                term = top_k_ret_docs_info_q_ith['terms'][j][d_t]
                doc_term_vec = doc_terms_vec[d_t]
                sim = 1 - spatial.distance.cosine(term_vec, doc_term_vec)
                q_t_mention = query_info['terms'][q_t]
                similarity_q.setdefault(q_t_mention, {})
                similarity_q[q_t_mention].setdefault(doc_id, []).append((term, sim))
    return similarity_q
