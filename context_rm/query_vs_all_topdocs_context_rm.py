import sys


# get the list of terms for all topk documents
def get_terms_all_docs(similarity_queryTerms_to_DocsTerms):
    terms = set()
    for doc_id in similarity_queryTerms_to_DocsTerms:
        for mention in similarity_queryTerms_to_DocsTerms[doc_id]:
            terms.add(mention[0])
    return list(terms)


# get the normalizing value for query
def get_normalizer(similarity_queryTerms_to_DocsTerms):
    normalizer = 0
    for doc_id in similarity_queryTerms_to_DocsTerms:
        for m in similarity_queryTerms_to_DocsTerms[doc_id]:
            normalizer += m[1]
    return normalizer


# get the aggregate value of similarity of term mentions to query
def get_sum_sim_score_termMentions_to_query(similarity_queryTerms_to_DocsTerms, term):
    sum_sim_score_termMentions_to_queryTerm = 0
    for doc_id in similarity_queryTerms_to_DocsTerms:
        for mention in similarity_queryTerms_to_DocsTerms[doc_id]:
            if mention[0] == term:
                sum_sim_score_termMentions_to_queryTerm += mention[1]
    return sum_sim_score_termMentions_to_queryTerm


def get_docsTerms_per_query_normalized_score(similarity_queryTerms_to_DocsTerms):
    terms_scores = {}
    docs_terms = get_terms_all_docs(similarity_queryTerms_to_DocsTerms)
    normalizer_query = get_normalizer(similarity_queryTerms_to_DocsTerms)
    for t in docs_terms:
        score = get_sum_sim_score_termMentions_to_query(similarity_queryTerms_to_DocsTerms, t)
        normalized_score = float(score) / float(normalizer_query)
        terms_scores[t] = normalized_score
    return terms_scores