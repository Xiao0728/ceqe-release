import sys

# get the list of terms for all topk documents
def get_terms_all_docs(similarity_queryTerms_to_DocsTerms):
    terms = set()
    sample_query_term = list(similarity_queryTerms_to_DocsTerms.keys())[0]
    for doc_id in similarity_queryTerms_to_DocsTerms[sample_query_term]:
        for mention in similarity_queryTerms_to_DocsTerms[sample_query_term][doc_id]:
            terms.add(mention[0])
    return list(terms)


# get the normalizing value for query term q_t
def get_normalizer(similarity_queryTerms_to_DocsTerms, q_t):
    normalizer = 0
    for doc_id in similarity_queryTerms_to_DocsTerms[q_t]:
        for m in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
            normalizer += m[1]
    return normalizer


# get the aggregate value of similarity of term mentions to query term q_t
def get_sum_sim_score_termMentions_to_queryTerm(similarity_queryTerms_to_DocsTerms, q_t, term):
    sum_sim_score_termMentions_to_queryTerm = 0
    for doc_id in similarity_queryTerms_to_DocsTerms[q_t]:
        for mention in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
            if mention[0] == term:
                sum_sim_score_termMentions_to_queryTerm += mention[1]
    return sum_sim_score_termMentions_to_queryTerm


def get_docsTerms_per_queryTerms_normalized_score(similarity_queryTerms_to_DocsTerms):
    terms_scores = {}
    docs_terms = get_terms_all_docs(similarity_queryTerms_to_DocsTerms)
    for q_term in similarity_queryTerms_to_DocsTerms:
        normalizer_q_term = get_normalizer(similarity_queryTerms_to_DocsTerms, q_term)
        for t in docs_terms:
            score = get_sum_sim_score_termMentions_to_queryTerm(similarity_queryTerms_to_DocsTerms, q_term, t)
            normalized_score = float(score) / float(normalizer_q_term)
            terms_scores.setdefault(t, {})
            terms_scores[t][q_term] = normalized_score
    return terms_scores


def get_docsTerms_per_query_normalized_score_mul_pool(docsTerms_per_queryTerms_normalized_score):
    final_scores = {}

    for t in docsTerms_per_queryTerms_normalized_score:
        final_score_t = 1
        for q_term_i in docsTerms_per_queryTerms_normalized_score[t]:
            if q_term_i != '[SEP]' and q_term_i != '[CLS]':
                final_score_t *= docsTerms_per_queryTerms_normalized_score[t][q_term_i]
        final_scores[t] = final_score_t
    return final_scores


def get_docsTerms_per_query_normalized_score_max_pool(docsTerms_per_queryTerms_normalized_score):
    final_scores = {}

    for t in docsTerms_per_queryTerms_normalized_score:
        final_score_t = -sys.maxsize
        for q_term_i in docsTerms_per_queryTerms_normalized_score[t]:
            if q_term_i != '[SEP]' and q_term_i != '[CLS]':
                # find the maximum value
                if docsTerms_per_queryTerms_normalized_score[t][q_term_i] > final_score_t:
                    final_score_t = docsTerms_per_queryTerms_normalized_score[t][q_term_i]
        final_scores[t] = final_score_t
    return final_scores
