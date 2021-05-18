import json
import tqdm
import numpy as np
from graph import *


def save_features(inputs_features, save_dir):
    inputs_features_json_format = []
    for i in tqdm.tqdm(range(len(inputs_features))):
        inputs_features_json_format.append(
            get_json_format_features(inputs_features[i]))

    with open(save_dir, 'w') as f:
        f.write('[\n')
        for i in range(len(inputs_features_json_format) - 1):
            f.write("%s,\n" % json.dumps(inputs_features_json_format[i]))
        f.write("%s\n" % json.dumps(inputs_features_json_format[len(inputs_features_json_format) - 1]))
        f.write(']')


def get_json_format_features(input_i_feature):
    json_format = {
        'id': input_i_feature.guid,
        'tokens': input_i_feature.input_tokens,
        'input_ids': input_i_feature.input_ids,
        'input_mask': input_i_feature.input_mask,
        'segment_ids': input_i_feature.segment_ids
    }
    return json_format


# def save_multi_layers_embeddings(vocab_embeds_features, save_file, num_embed_layer, last_entry):
#     # vocab_embeds_features is (model_output_batch, input_features) tuple
#     output_batch, vocab_features_batch = vocab_embeds_features
#     mlm_embeds, _, hidden_layers = output_batch
#     # each term in the batch
#     for j in range(len(mlm_embeds)):
#         json_format_embed_j = get_json_format_embeds(mlm_embeds[j], vocab_features_batch[j],
#                                                      get_n_layers_j(j, hidden_layers, num_embed_layer))
#         if last_entry and j == len(mlm_embeds) - 1:
#             print(j)
#             save_file.write("%s\n" % json.dumps(json_format_embed_j))
#         else:
#             save_file.write("%s,\n" % json.dumps(json_format_embed_j))


def get_n_layers_j(term_j, hidden_layers, num_embed_layer):
    layers_embeds = {}
    num_total_layers = len(hidden_layers) - 1
    for i in range(num_embed_layer):
        layers_embeds['embed_l%s' % (num_total_layers - i)] = hidden_layers[num_total_layers - i - 1][
            term_j].numpy().tolist()
    return layers_embeds


def get_json_format_embeds(mlm_embeds, vocab_features_i, layers_embeds):
    # if features are loaded from a json file run
    json_format = {
        'id': vocab_features_i['id'],
        'tokens': vocab_features_i['tokens'],
        'embed_mlm': mlm_embeds.numpy().tolist()
    }
    json_format.update(layers_embeds)
    return json_format


def save_one_layer_embeddings(embedding_matrix, query_train_features, f, pooling_strategy):
    for i in range(len(embedding_matrix)):
        embedding_matrix_i = embedding_matrix[i]
        if pooling_strategy == PoolingStrategy.NONE:
            embedding_matrix_i = np.array(embedding_matrix_i).reshape((-1, 768))
            embedding_matrix_i_non_zero = []
            for k in embedding_matrix_i:
                if np.array_equal(k, np.zeros(768)):
                    break
                embedding_matrix_i_non_zero.append(k)
            embedding_matrix_i_non_zero = np.array(embedding_matrix_i_non_zero)
            # print(embedding_matrix_i.shape, embedding_matrix_i_non_zero.shape, len(query_train_features[i]['tokens']))
            assert embedding_matrix_i_non_zero.shape[0] == len(query_train_features[i]['tokens'])
            f.write("%s,\n" % json.dumps(
                {'id': query_train_features[i]['id'], 'tokens': query_train_features[i]['tokens'],
                 'embedding': embedding_matrix_i_non_zero.tolist()}))

        else:
            f.write(
                "%s,\n" % json.dumps({'id': query_train_features[i]['id'], 'embedding': embedding_matrix_i.tolist()}))


def save_multi_layers_embeddings(embedding_matrix, query_train_features, files, pooling_layers, pooling_strategy):
    for i in range(len(embedding_matrix)):
        # embedding matrix for one instance
        embedding_matrix_i = embedding_matrix[i]
        embedding_matrix_i = np.array(embedding_matrix_i)
        if pooling_strategy == PoolingStrategy.NONE:
            # print(embedding_matrix_i.size())
            embedding_matrix_i_non_zero = []
            # print(embedding_matrix_i.shape[1])
            for k in embedding_matrix_i:
                if np.array_equal(k, np.zeros(embedding_matrix_i.shape[1])):
                    break
                embedding_matrix_i_non_zero.append(k)
            embedding_matrix_i_non_zero = np.array(embedding_matrix_i_non_zero)
            # print(embedding_matrix_i.shape, embedding_matrix_i_non_zero.shape, len(query_train_features[i]['tokens']))
            assert embedding_matrix_i_non_zero.shape[0] == len(query_train_features[i]['tokens'])
            number_layers = embedding_matrix_i_non_zero.shape[1] / 768
            assert number_layers == len(pooling_layers)
            # print(number_layers)
            multi_layer_embedding_matrix_i_non_zero = np.split(embedding_matrix_i_non_zero, number_layers, axis=1)
            # print(len(multi_layer_embedding_matrix_i_non_zero))
            # to save layers are concatenated in order originally, so we are saving them in different layers based
            # on that order
            for l_i in range(len(pooling_layers)):
                f = files[l_i]
                f.write("%s,\n" % json.dumps(
                    {'id': query_train_features[i]['id'], 'tokens': query_train_features[i]['tokens'],
                     'embedding': multi_layer_embedding_matrix_i_non_zero[l_i].tolist()}))

        else:
            number_layers = embedding_matrix_i.shape[0] / 768
            assert number_layers == len(pooling_layers)
            multi_layer_embedding_matrix_i = np.split(embedding_matrix_i, number_layers, axis=0)
            for l_i in range(len(pooling_layers)):
                f = files[l_i]
                f.write(
                    "%s,\n" % json.dumps(
                        {'id': query_train_features[i]['id'], 'embedding': multi_layer_embedding_matrix_i[l_i].tolist()}))
