import sys
import operator
import math
import numpy as np
import argparse
from pathlib import Path
import krovetz
import pprint

import torch

import extract_features as ex_f
import main.tokens_to_terms as tokens_to_terms
from graph import *
from loader import *
from utils import writer
from context_rm import get_similarity, query_vs_per_doc_context_rm, queryTerm_vs_per_doc_context_rm


def get_query_text(q_id):
    q_text = load_queries_dict(args.query_file)[q_id]
    return [(q_id, q_text)]


def get_prf_docs(q_id, num_prf):
    doc_file = args.prf_docs_path + q_id.replace('/', '_')
    a = load_collection(doc_file)[:num_prf]
    return a


def inputFeature_to_dict(feature):
    feature_dict = {'input_ids': feature.input_ids, 'input_mask': feature.input_mask,
                    'segment_ids': feature.segment_ids, 'tokens': feature.input_tokens, 'id': feature.guid}

    return feature_dict


def get_features(input_id_text, chunk):
    if args.state_dict:
        state_dict = torch.load(open(args.state_dict, 'rb'))
        print('has state_dict')
    else:
        state_dict = None
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer, state_dict=state_dict)
    processor = ex_f.GetEmbeddingProcessor()
    examples = processor.get_input_examples(input_id_text)
    features = []
    for example in examples:
        example_for_processing = (example, args.max_seq_len, tokenizer, chunk)
        features.append(ex_f.convert_example_to_feature(example_for_processing))

    if chunk:
        features_flattened = []
        for i in range(len(features)):
            features_flattened.extend(features[i])
        features = features_flattened

    features_dict = [inputFeature_to_dict(f) for f in features]
    return features_dict, tokenizer


def get_embedding(features, graph_bert):
    # Getting the layers and write down
    batch_size = args.batch_size
    batch_count = math.ceil(len(features) / batch_size)
    instances_embedding_matrix = []
    for i in range(batch_count):
        embedding_matrix = graph_bert.get_embedding_matrix(features[i * batch_size: (i + 1) * batch_size])
        instances_embedding_matrix.append(embedding_matrix)

    return torch.cat(instances_embedding_matrix, 0)


def remove_zero_tokens(embedding_matrix):
    embedding_matrix_non_zero = []
    for i in range(embedding_matrix.shape[0]):
        # embedding matrix for one instance
        embedding_matrix_i = embedding_matrix[i]
        embedding_matrix_i_non_zero = []
        for k in embedding_matrix_i:
            if np.array_equal(k, np.zeros(embedding_matrix_i.shape[1])):
                break
            embedding_matrix_i_non_zero.append(k)
        embedding_matrix_non_zero.append(np.array(embedding_matrix_i_non_zero))
    embedding_matrix_non_zero = np.array(embedding_matrix_non_zero)
    return embedding_matrix_non_zero


def split_layers():
    return


def get_info_structure(features, embeddings, pooling):
    info_structure = []
    if pooling == PoolingStrategy.REDUCE_MEAN:
        for i in range(len(features)):
            info_structure.append(
                {'id': features[i]['id'], 'embedding': embeddings[i]})
    elif pooling == PoolingStrategy.NONE:
        for i in range(len(features)):
            info_structure.append(
                {'id': features[i]['id'], 'embedding': embeddings[i], 'tokens': features[i]['tokens']})

    return info_structure


def context_rm_query(query_info, list_prf_docs_tokens_info, ret_result, query):
    # only one query, prf_docs_tokens_info is for the only query
    q_id = query_info['id']
    dict_prf_docs_tokens_info = {'id': [], 'tokens': [], 'embedding': []}
    for d in range(len(list_prf_docs_tokens_info)):
        dict_prf_docs_tokens_info['id'].append(list_prf_docs_tokens_info[d]['id'])
        dict_prf_docs_tokens_info['tokens'].append(list_prf_docs_tokens_info[d]['tokens'])
        dict_prf_docs_tokens_info['embedding'].append(list_prf_docs_tokens_info[d]['embedding'])

    docs_terms, docs_terms_embeds = tokens_to_terms.get_terms(dict_prf_docs_tokens_info)
    docs_terms_embeds_pooled_np = tokens_to_terms.get_terms_embeds_pooled(docs_terms, docs_terms_embeds)
    # docs_terms_embeds_pooled_np = tokens_to_terms.get_terms_embeds_sip_pooling(docs_terms, docs_terms_embeds, cfg.vocab_freq, cfg.collection_size, tokenizer)
    # print(len(docs_terms), len(docs_terms_embeds), len(docs_terms_embeds_pooled_np), len(list_prf_docs_tokens_info))

    prf_docs_terms_info = {'id': [], 'terms': [], 'embedding': []}
    for d in range(len(dict_prf_docs_tokens_info['id'])):
        prf_docs_terms_info['id'].append(dict_prf_docs_tokens_info['id'][d])
        prf_docs_terms_info['terms'].append(docs_terms[d])
        prf_docs_terms_info['embedding'].append(docs_terms_embeds_pooled_np[d])

    similarity = get_similarity.get_similarity_query_and_docTerms_CAR(query_info, prf_docs_terms_info)
    # pprint.pprint(similarity)

    docsTerms_per_queryTerms_normalized_score = query_vs_per_doc_context_rm.get_docsTerms_per_query_normalized_score(
        similarity)

    # pprint.pprint(docsTerms_per_queryTerms_normalized_score)
    exp_terms_score_mul_doc_prob = query_vs_per_doc_context_rm.get_exp_terms_score_mul_doc_prob(
        docsTerms_per_queryTerms_normalized_score,
        ret_result[q_id])
    unique_expansion_terms = get_unique_expansion_terms(exp_terms_score_mul_doc_prob, query)
    writer.write_expansion_terms(unique_expansion_terms,
                                 args.output_dir + 'centroid/' + str(q_id).replace('/', '_') + '_query.json')


def context_rm_queryTerm(query_token_info, list_prf_docs_tokens_info, ret_result, query):
    # only one query, list_prf_docs_tokens_info is for the only query
    q_id = query_token_info['id']
    queries_token_info = {'id': [], 'tokens': [], 'embedding': []}
    queries_token_info['id'].append(query_token_info['id'])
    queries_token_info['tokens'].append(query_token_info['tokens'])
    queries_token_info['embedding'].append(query_token_info['embedding'])
    query_terms, query_terms_embeds = tokens_to_terms.get_terms(queries_token_info)
    query_terms_embeds_pooled_np = tokens_to_terms.get_terms_embeds_pooled(query_terms, query_terms_embeds)
    # query_terms_embeds_pooled_np = tokens_to_terms.get_terms_embeds_sip_pooling(query_terms, query_terms_embeds, cfg.vocab_freq, cfg.collection_size, tokenizer)

    query_info = {}
    query_info['id'] = query_token_info['id']
    query_info['terms'] = query_terms[0]
    query_info['embedding'] = query_terms_embeds_pooled_np[0]

    dict_prf_docs_tokens_info = {'id': [], 'tokens': [], 'embedding': []}
    for d in range(len(list_prf_docs_tokens_info)):
        dict_prf_docs_tokens_info['id'].append(list_prf_docs_tokens_info[d]['id'])
        dict_prf_docs_tokens_info['tokens'].append(list_prf_docs_tokens_info[d]['tokens'])
        dict_prf_docs_tokens_info['embedding'].append(list_prf_docs_tokens_info[d]['embedding'])

    # group tokens to terms
    docs_terms, docs_terms_embeds = tokens_to_terms.get_terms(dict_prf_docs_tokens_info)
    # docs_terms_embeds_pooled_np = tokens_to_terms.get_terms_embeds_sip_pooling(docs_terms, docs_terms_embeds, cfg.vocab_freq, cfg.collection_size, tokenizer)

    docs_terms_embeds_pooled_np = tokens_to_terms.get_terms_embeds_pooled(docs_terms, docs_terms_embeds)
    # print(len(docs_terms), len(docs_terms_embeds), len(docs_terms_embeds_pooled_np), len(list_prf_docs_tokens_info))

    prf_docs_terms_info = {'id': [], 'terms': [], 'embedding': []}
    for d in range(len(dict_prf_docs_tokens_info['id'])):
        prf_docs_terms_info['id'].append(dict_prf_docs_tokens_info['id'][d])
        prf_docs_terms_info['terms'].append(docs_terms[d])
        prf_docs_terms_info['embedding'].append(docs_terms_embeds_pooled_np[d])

    similarity = get_similarity.get_similarity_queryTerms_and_docsTerms_CAR(query_info, prf_docs_terms_info)
    # pprint.pprint(similarity)

    docsTerms_per_queryTerms_normalized_score = queryTerm_vs_per_doc_context_rm.get_docsTerms_per_queryTerms_normalized_score(
        similarity)

    final_scores_mul = queryTerm_vs_per_doc_context_rm.get_docsTerms_per_query_normalized_score_mul_pool(
        docsTerms_per_queryTerms_normalized_score)
    exp_terms_score_mul_doc_prob_mul = queryTerm_vs_per_doc_context_rm.get_exp_terms_score_mul_doc_prob(
        final_scores_mul, ret_result[q_id])

    unique_expansion_terms = get_unique_expansion_terms(exp_terms_score_mul_doc_prob_mul, query)
    writer.write_expansion_terms(unique_expansion_terms,
                                 args.output_dir + "term_based_mul/" + str(q_id).replace("/",
                                                                                       "_") + '_queryTerm_mul.json')

    final_scores_max = queryTerm_vs_per_doc_context_rm.get_docsTerms_per_query_normalized_score_max_pool(
        docsTerms_per_queryTerms_normalized_score)
    exp_terms_score_mul_doc_prob_max = queryTerm_vs_per_doc_context_rm.get_exp_terms_score_mul_doc_prob(
        final_scores_max, ret_result[q_id])
    unique_expansion_terms = get_unique_expansion_terms(exp_terms_score_mul_doc_prob_max, query)
    writer.write_expansion_terms(unique_expansion_terms,
                                 args.output_dir + "term_based_max/" + str(q_id).replace("/",
                                                                                       "_") + '_queryTerm_max.json')


def get_unique_expansion_terms(exp_terms_with_score, query):
    ks = krovetz.PyKrovetzStemmer()

    new_queries = {}
    q_id = query[0][0]
    q_text = query[0][1]
    # bow_q_text = set(q_text.split(' '))
    bow_q_text = set([ks.stem(q_t) for q_t in set(q_text.split(' '))])

    most_sim = sorted(exp_terms_with_score.items(), key=operator.itemgetter(1), reverse=True)
    unique_exp_terms = []
    for t in most_sim:
        # add topk
        if len(unique_exp_terms) < args.num_exp_terms:
            # eliminate stopwords
            if t[0] not in args.stopwords and t[0] != '[CLS]' and t[0] != '[SEP]':
                if ks.stem(t[0]) not in bow_q_text:
                    # if t[0] not in bow_q_text:
                    unique_exp_terms.append(t)
        else:
            break
    new_queries.setdefault(q_id, [])
    new_queries[q_id] = unique_exp_terms
    # print([xx[0] for xx in unique_exp_terms[:20]])
    return new_queries


def rank_expansion_terms(q_id, num_prf, ret_result):

    query = get_query_text(q_id)
    prf_docs = get_prf_docs(q_id, num_prf)
    query_features, tokenizer = get_features(query, chunk=False)
    prf_docs_features, tokenizer = get_features(prf_docs, chunk=True)

    graph_pooling_MEAN = Graph(args.model_config, args.pooling_layer, PoolingStrategy.REDUCE_MEAN, args.attention_pooling)
    graph_pooling_MEAN.initiate_model()

    query_embedding_matrix = get_embedding(query_features, graph_pooling_MEAN)
    query_embedding_matrix = query_embedding_matrix.detach().cpu().numpy()
    query_info = get_info_structure(query_features, np.array(query_embedding_matrix), PoolingStrategy.REDUCE_MEAN)[
        0]  # only one query
    # print(query_info)
    # print(query_embedding_matrix.size())
    # print(np.array(query_embedding_matrix).shape)

    graph_pooling_NONE = Graph(args.model_config, args.pooling_layer, PoolingStrategy.NONE, args.attention_pooling)
    graph_pooling_NONE.initiate_model()

    prf_docs_tokens_embedding_matrix = get_embedding(prf_docs_features, graph_pooling_NONE)
    prf_docs_tokens_embedding_matrix = prf_docs_tokens_embedding_matrix.detach().cpu().numpy()
    prf_docs_tokens_embedding_matrix_non_zero = remove_zero_tokens(np.array(prf_docs_tokens_embedding_matrix))
    prf_docs_tokens_info = get_info_structure(prf_docs_features, prf_docs_tokens_embedding_matrix_non_zero,
                                              PoolingStrategy.NONE)
    # print(prf_docs_tokens_info)
    # print(prf_docs_tokens_embedding_matrix.size())
    # print(np.array(prf_docs_tokens_embedding_matrix).shape)
    # print(prf_docs_tokens_embedding_matrix_non_zero.shape)

    query_tokens_embedding_matrix = get_embedding(query_features, graph_pooling_NONE)
    query_tokens_embedding_matrix = query_tokens_embedding_matrix.detach().cpu().numpy()
    query_tokens_embedding_matrix_non_zero = remove_zero_tokens(np.array(query_tokens_embedding_matrix))
    query_tokens_info = get_info_structure(query_features, query_tokens_embedding_matrix_non_zero,
                                           PoolingStrategy.NONE)[0]  # only one query
    # print(query_tokens_info)
    # print(query_tokens_embedding_matrix.size())
    # print(np.array(query_tokens_embedding_matrix).shape)
    # print(type(query_tokens_embedding_matrix_non_zero))
    # print(query_tokens_embedding_matrix_non_zero.shape)

    context_rm_query(query_info, prf_docs_tokens_info, ret_result, query)
    context_rm_queryTerm(query_tokens_info, prf_docs_tokens_info, ret_result, query)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--query_id', type=str, required=True,
                        help="query id of the query input for ranking expansion terms")

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False,
                        help="The Bert model name or path")

    parser.add_argument("--model_config_or_path", default="bert-base-uncased", type=str,
                        help="Pretrained config name or path")

    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="Pretrained tokenizer name or path")

    # parser.add_argument("--cache_dir", default="/mnt/scratch/shnaseri/huggingface_cache/", type=str,
    #                     help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--state_dict", default="", type=str, required=False,
                        help="In case you want to initiate the BERT model with the weights of a fine-tuned BERT model, you need to give the weight pickle file as the state_dict.")

    parser.add_argument("--max_seq_len", default=128, type=int, required=False,
                        help='The maximum sequence length of the PRF documents in tokens to be input in the BERT model.')

    parser.add_argument("--batch_size", default=100, type=int, required=False,
                        help='Batch size of the expansion terms to calculate their score.')

    parser.add_argument("--num_prf_docs", default=20, type=int, required=False,
                        help='Depth of the pseudo relevance feedback(PRF) documents.')

    parser.add_argument("--num_exp_terms", default=1000, type=int, required=False,
                        help='Number of expansion terms to rank.')

    parser.add_argument("--output_dir", type=str, required=True,
                        help="The directory where the ranked expansion terms are saved.")

    parser.add_argument("--stopwords_file", default='./data/rmstop.txt', type=str, required=False,
                        help="The stopwords file. The default stopwords list is the Galago search engine stopwords list.")

    parser.add_argument("--query_file", type=str, required=True,
                        help="The stopped query file in a tab separated format: q_id\tq_text.")

    parser.add_argument("--prf_docs_path", type=str, required=True,
                        help="The path of prf documents text for each query. The PRF documents are saved separately for each query in a tab separated file: doc_id\tdoc_text.")

    parser.add_argument("--run_file", type=str, required=True,
                        help='The first pass run file resulting in the PRF documents.')

    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    output_hidden_states = True
    output_attentions = False
    if args.state_dict:
        state_dict = torch.load(open(args.state_dict, 'rb'))
    else:
        state_dict = None

    model_config = (args.model_name_or_path, args.model_config_or_path, device, output_hidden_states, output_attentions, state_dict)
    args.model_config = model_config

    # using the second to last layer of the BERT model for representing tokens
    args.pooling_layer = [-2]
    args.attention_pooling = PoolingStrategy.NONE

    # The expansion terms are scored using the 3 approaches in one pass
    Path("{}/centroid".format(args.output_dir)).mkdir(parents=True, exist_ok=True)
    Path("{}/max_pool".format(args.output_dir)).mkdir(parents=True, exist_ok=True)
    Path("{}/mul_pool".format(args.output_dir)).mkdir(parents=True, exist_ok=True)

    temp = []
    with open(args.stopwords_file) as f:
        for line in f:
            temp.append(line.strip())
    stopwords = set(temp)
    args.stopwords = stopwords


    retrieval_result = load_run(args.run_file)
    rank_expansion_terms(args.query_id, args.num_prf_docs, retrieval_result)
