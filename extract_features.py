"""
This file is adopted from the utils_glue.py that can be found here:
https://github.com/huggingface/pytorch-transformers/blob/f31154cb9df44b9535bd21eb5962e7a91711e9d1/examples/utils_glue.py#L53
"""
from __future__ import absolute_import, division, print_function

import logging
import tokenization
import math

logger = logging.getLogger()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class GetEmbeddingProcessor():

    def get_input_examples(self, data):
        return self._create_examples(data)

    def _create_examples(self, lines):
        """Creates examples for the the training and dev sets"""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_tokens, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_tokens = input_tokens
        self.guid = guid


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_input_features(tokens_a, tokens_b, tokenizer, max_seq_length, example):
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

    if tokens_b:
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    else:
        segment_ids = [1] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # To avoid padding for the extracting the vocab embeddings -> In other words no padding
    # max_seq_length = min(len(input_ids), max_seq_length)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    logger.debug('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens]))
    logger.debug('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
    logger.debug('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
    logger.debug('input_type_ids: %s' % ' '.join([str(x) for x in segment_ids]))

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         input_tokens=tokens,
                         guid=example.guid)


def convert_example_to_feature(example_row):

    # return example_row
    example, max_seq_length, tokenizer, chunk = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # ToDo
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if not chunk:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            input_features = get_input_features(tokens_a, tokens_b, tokenizer, max_seq_length, example)
            return input_features

        else:
            chunks_input_features = []
            begin_pointer = 0
            while begin_pointer < len(tokens_a):
                end_pointer = begin_pointer + (max_seq_length - 2)
                if end_pointer < len(tokens_a):
                    # check to see if part of a token is will be in the next token
                    while tokens_a[end_pointer].startswith("##"):
                        end_pointer -= 1
                chunked_tokens_a = tokens_a[begin_pointer:end_pointer]
                begin_pointer = end_pointer
                chunks_input_features.append(get_input_features(chunked_tokens_a, tokens_b, tokenizer,
                                                                    max_seq_length, example))

            #number_chunk = math.ceil(len(tokens_a) / (max_seq_length - 2))
            #for i in range(number_chunk):
            #    chunked_tokens_a = tokens_a[i * (max_seq_length - 2): (i + 1) * (max_seq_length - 2)]
            #    chunks_input_features.append(
            #        get_input_features(chunked_tokens_a, tokens_b, tokenizer, max_seq_length, example))
            return chunks_input_features
