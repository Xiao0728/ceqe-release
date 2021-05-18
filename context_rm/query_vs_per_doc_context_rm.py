import math
from scipy.special import logsumexp
from scipy.special import softmax


# get the list of terms for all topk documents
def get_terms(similarity_queryTerms_to_DocsTerms, doc_id):
    terms = set()
    for mention in similarity_queryTerms_to_DocsTerms[doc_id]:
        terms.add(mention[0])
    return list(terms)


# get the normalizing value for query for document doc_id
def get_normalizer(similarity_queryTerms_to_DocsTerms, doc_id):
    normalizer = 0
    for m in similarity_queryTerms_to_DocsTerms[doc_id]:
        normalizer += m[1]
    return normalizer


# get the aggregate value of similarity of term mentions to query term q_t
def get_sum_sim_score_termMentions_to_query(similarity_queryTerms_to_DocsTerms, doc_id, term):
    sum_sim_score_termMentions_to_queryTerm = 0
    for mention in similarity_queryTerms_to_DocsTerms[doc_id]:
        if mention[0] == term:
            sum_sim_score_termMentions_to_queryTerm += mention[1]
    return sum_sim_score_termMentions_to_queryTerm


def get_docsTerms_per_query_normalized_score(similarity_queryTerms_to_DocsTerms):
    docs_terms_scores = {}
    for doc_id in similarity_queryTerms_to_DocsTerms:
        doc_terms = get_terms(similarity_queryTerms_to_DocsTerms, doc_id)
        normalizer_query = get_normalizer(similarity_queryTerms_to_DocsTerms, doc_id)
        for t in doc_terms:
            score = get_sum_sim_score_termMentions_to_query(similarity_queryTerms_to_DocsTerms, doc_id, t)
            normalized_score = float(score) / float(normalizer_query)
            docs_terms_scores.setdefault(doc_id, {})
            docs_terms_scores[doc_id][t] = normalized_score
    return docs_terms_scores


def get_exp_terms_score_mul_doc_prob(docsTerms_per_query_normalized_score, retrieval_result_query):
    exp_terms_score_mul_doc_prob = {}
    posterior_scores = logs_to_posteriors(retrieval_result_query)

    for doc_id in docsTerms_per_query_normalized_score:
        doc_prob = posterior_scores[doc_id]

        for t in docsTerms_per_query_normalized_score[doc_id]:
            exp_terms_score_mul_doc_prob.setdefault(t, 0)
            exp_terms_score_mul_doc_prob[t] += (docsTerms_per_query_normalized_score[doc_id][t] * doc_prob)
    return exp_terms_score_mul_doc_prob


def logs_to_posteriors(retrieval_result_query):
    docs_score = []
    docs_id = []

    for doc_id in retrieval_result_query:
        docs_score.append(retrieval_result_query[doc_id])
        docs_id.append(doc_id)
    assert len(docs_id) == len(docs_score)

    docs_score_normalized = softmax(docs_score)

    #log_sum_exp = logsumexp(docs_score_normalized)
    posteriors = {}

    #for i in range(len(docs_id)):
     #   log_posterior =  docs_score_normalized[i] - log_sum_exp
      #  doc_id = docs_id[i]
       # posteriors[doc_id] = (math.exp(log_posterior))
    log_sum_exp = logsumexp(docs_score)
    for doc_id in retrieval_result_query:
        log_posterior = retrieval_result_query[doc_id] - log_sum_exp
        posteriors[doc_id] = (math.exp(log_posterior))
    return posteriors
