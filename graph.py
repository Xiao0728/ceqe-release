import torch
from transformers import BertTokenizer, BertModel, BertConfig

from enum import Enum


# This class is from bert-as-a-service github repository: https://github.com/hanxiao/bert-as-service
class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


class Graph():
    def __init__(self, model_config, pooling_layer, pooling_strategy, attention_pooling):
        self.model_config = model_config
        self.pooling_layer = pooling_layer
        self.pooling_strategy = pooling_strategy
        self.attention_pooling = attention_pooling
        self.model = None
        self.device = None      

    def initiate_model(self):
        bert_model, bert_model_config, device, output_hidden_states, output_attentions, state_dict = self.model_config
        config = BertConfig.from_pretrained(bert_model_config, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        self.model = BertModel.from_pretrained(bert_model, config=config, state_dict=state_dict) 
        assert self.model.config.output_hidden_states == output_hidden_states
        assert self.model.config.output_attentions == output_attentions
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_embedding_matrix(self, input_features):
        input_ids = torch.tensor([f['input_ids'] for f in input_features], dtype=torch.long)
        input_mask = torch.tensor([f['input_mask'] for f in input_features], dtype=torch.long)
        segment_ids = torch.tensor([f['segment_ids'] for f in input_features], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)       

        with torch.no_grad():
            mlm_output, last_layer, all_encoder_layers = self.model(input_ids, token_type_ids=segment_ids,
                                                                    attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers[:-1]  # last element is the mlm_output

        if len(self.pooling_layer) == 1:
            encoder_layer = all_encoder_layers[self.pooling_layer[0]]
        else:
            # print('multiple layers')
            all_layers = [all_encoder_layers[l] for l in self.pooling_layer]
            encoder_layer = torch.cat(all_layers, -1)  # ToDo check if the concat's axis is correct ??
            # raise

        minus_mask = lambda x, m: x - (1.0 - m.unsqueeze(-1)) * 1e5
        mul_mask = lambda x, m: x * m.unsqueeze(-1)
        masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1).values
        masked_reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (
                torch.sum(m, dim=1, keepdim=True) + 1e-10)

        input_mask = input_mask.float()
        if self.pooling_strategy == PoolingStrategy.REDUCE_MEAN:
            pooled = masked_reduce_mean(encoder_layer, input_mask)
        elif self.pooling_strategy == PoolingStrategy.REDUCE_MAX:
            pooled = masked_reduce_max(encoder_layer, input_mask)
        elif self.pooling_strategy == PoolingStrategy.NONE:
            pooled = mul_mask(encoder_layer, input_mask)
        else:
            raise NotImplementedError()

        return pooled

    def get_attended_embedding_matrix(self, input_features):
        print(self.attention_pooling, 'transposed')
        input_ids = torch.tensor([f['input_ids'] for f in input_features], dtype=torch.long)
        input_mask = torch.tensor([f['input_mask'] for f in input_features], dtype=torch.long)
        segment_ids = torch.tensor([f['segment_ids'] for f in input_features], dtype=torch.long)
        with torch.no_grad():
            mlm_output, _, all_encoder_layers, attentions = self.model(input_ids, token_type_ids=segment_ids,
                                                                       attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers[:-1]  # last element is the mlm_output

        if len(self.pooling_layer) == 1:
            encoder_layer = all_encoder_layers[self.pooling_layer[0]]
            attention_layer = attentions[self.pooling_layer[0]]

            batch_size, num_att_head, seq_len, seq_len = attention_layer.size()
            embedding_size = encoder_layer.size()[-1]
            # Get attended embedding for
            attended_embedding = torch.matmul(torch.transpose(attention_layer, 2, 3),
                                              encoder_layer.unsqueeze(1).expand(batch_size, num_att_head, seq_len,
                                                                                embedding_size))
            # Avg pool the different attention heads
            if self.attention_pooling == PoolingStrategy.REDUCE_MEAN:
                pooled_attended_embedding = torch.sum(attended_embedding, dim=1) / num_att_head
            # Max pool the different attention heads
            elif self.attention_pooling == PoolingStrategy.REDUCE_MAX:
                pooled_attended_embedding = torch.max(attended_embedding, dim=1).values

        else:
            raise
            # print('multiple layers')
            all_layers = [all_encoder_layers[l] for l in self.pooling_layer]
            encoder_layer = torch.cat(all_layers, -1)  # ToDo check if the concat's axis is correct ??

        minus_mask = lambda x, m: x - (1.0 - m.unsqueeze(-1)) * 1e5
        mul_mask = lambda x, m: x * m.unsqueeze(-1)
        masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1).values
        masked_reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (
                torch.sum(m, dim=1, keepdim=True) + 1e-10)

        input_mask = input_mask.float()
        if self.pooling_strategy == PoolingStrategy.REDUCE_MEAN:
            pooled = masked_reduce_mean(pooled_attended_embedding, input_mask)
        elif self.pooling_strategy == PoolingStrategy.REDUCE_MAX:
            pooled = masked_reduce_max(pooled_attended_embedding, input_mask)
        elif self.pooling_strategy == PoolingStrategy.NONE:
            pooled = mul_mask(pooled_attended_embedding, input_mask)
        else:
            raise NotImplementedError()
        return pooled
