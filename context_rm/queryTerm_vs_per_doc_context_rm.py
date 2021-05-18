import sys
import math
from scipy.special import logsumexp
from scipy.special import softmax



# get the list of terms for all topk documents
def get_terms(similarity_queryTerms_to_DocsTerms, doc_id):
    terms = set()
    sample_query_term = list(similarity_queryTerms_to_DocsTerms.keys())[0]
    for mention in similarity_queryTerms_to_DocsTerms[sample_query_term][doc_id]:
        terms.add(mention[0])
    return list(terms)


# get the normalizing value for query term q_t for document doc_id
def get_normalizer(similarity_queryTerms_to_DocsTerms, q_t, doc_id):#, topk_mentions):
    # create a bin of the mentions score
#     mentions_bin = {}
#     for m in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
#         if m[0] not in mentions_bin:
#             mentions_bin.setdefault(m[0], []).append(m[1])
#     normalizer = 0
#     for mention_word in mentions_bin:
#         a = mentions_bin[mention_word]
#         a.sort()
#         mentions_bin[mention_word] = a
#         normalizer += sum(a[:topk_mentions])
#     return normalizer
    
    normalizer = 0
    for m in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
        normalizer += m[1]
    return normalizer


# get the aggregate value of similarity of term mentions to query term q_t
def get_sum_sim_score_termMentions_to_queryTerm(similarity_queryTerms_to_DocsTerms, q_t, doc_id, term):#, topk_mentions):
#     sim_score_termMentions_to_queryTerm = []
#     for mention in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
#         if mention[0] == term:
#             sim_score_termMentions_to_queryTerm.append(mention[1])
#     sim_score_termMentions_to_queryTerm.sort()
#     return sum(sim_score_termMentions_to_queryTerm[:topk_mentions])
    
    sum_sim_score_termMentions_to_queryTerm = 0
    for mention in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
        if mention[0] == term:
            sum_sim_score_termMentions_to_queryTerm += mention[1]
    return sum_sim_score_termMentions_to_queryTerm
#     # max instead of mean over mentions
#     max_sim_score_termMentions_to_queryTerm = 0
#     for mention in similarity_queryTerms_to_DocsTerms[q_t][doc_id]:
#         if mention[0] == term:
#             max_sim_score_termMentions_to_queryTerm = max(mention[1], max_sim_score_termMentions_to_queryTerm)
#     return max_sim_score_termMentions_to_queryTerm


def get_docsTerms_per_queryTerms_normalized_score(similarity_queryTerms_to_DocsTerms):
    #topk_mentions = 10
    docs_terms_scores = {}
    for q_term in similarity_queryTerms_to_DocsTerms:
        for doc_id in similarity_queryTerms_to_DocsTerms[q_term]:
            doc_terms = get_terms(similarity_queryTerms_to_DocsTerms, doc_id)
            normalizer_q_term = get_normalizer(similarity_queryTerms_to_DocsTerms, q_term, doc_id)#, topk_mentions)
            for t in doc_terms:
                score = get_sum_sim_score_termMentions_to_queryTerm(similarity_queryTerms_to_DocsTerms, q_term, doc_id,
                                                                    t)#, topk_mentions)
                normalized_score = float(score) / float(normalizer_q_term)
                docs_terms_scores.setdefault(doc_id, {})
                docs_terms_scores[doc_id].setdefault(t, {})
                docs_terms_scores[doc_id][t][q_term] = normalized_score
    return docs_terms_scores


def get_docsTerms_per_query_normalized_score_mul_pool(docsTerms_per_queryTerms_normalized_score):
    final_scores = {}
    for doc_id in docsTerms_per_queryTerms_normalized_score:
        final_scores.setdefault(doc_id, {})
        for t in docsTerms_per_queryTerms_normalized_score[doc_id]:
            final_score_t = 1
            for q_term_i in docsTerms_per_queryTerms_normalized_score[doc_id][t]:
                if q_term_i != '[SEP]' and q_term_i != '[CLS]':
                    final_score_t *= docsTerms_per_queryTerms_normalized_score[doc_id][t][q_term_i]
            final_scores[doc_id][t] = final_score_t
    return final_scores


def get_docsTerms_per_query_normalized_score_max_pool(docsTerms_per_queryTerms_normalized_score):
    final_scores = {}
    for doc_id in docsTerms_per_queryTerms_normalized_score:
        final_scores.setdefault(doc_id, {})
        for t in docsTerms_per_queryTerms_normalized_score[doc_id]:
            final_score_t = -sys.maxsize
            for q_term_i in docsTerms_per_queryTerms_normalized_score[doc_id][t]:
                if q_term_i != '[SEP]' and q_term_i != '[CLS]':
                    # find the maximum value
                    if docsTerms_per_queryTerms_normalized_score[doc_id][t][q_term_i] > final_score_t:
                        final_score_t = docsTerms_per_queryTerms_normalized_score[doc_id][t][q_term_i]
            final_scores[doc_id][t] = final_score_t
    return final_scores


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

    #docs_socre = []
    #docs_id = []

    #for doc_id in retrieval_result_query:
     #   docs_socre.append(retrieval_result_query[doc_id])
      #  docs_id.append(doc_id)
    #assert len(docs_id) == len(docs_socre)

    #docs_socre_normalized = softmax(docs_socre)

    #log_sum_exp = logsumexp(docs_socre_normalized)
    #posteriors = {}

    #for i in range(len(docs_id)):
     #   log_posterior =  docs_socre_normalized[i] - log_sum_exp
      #  doc_id = docs_id[i]
       # posteriors[doc_id] = (math.exp(log_posterior))

    #return posteriors

    doc_socres = []
    for doc_id in retrieval_result_query:
        doc_socres.append(retrieval_result_query[doc_id])
   
    log_sum_exp = logsumexp(doc_socres)
    posteriors = {}
   
    for doc_id in retrieval_result_query:
        log_posterior = retrieval_result_query[doc_id] - log_sum_exp
        posteriors[doc_id] = (math.exp(log_posterior))
    return posteriors
