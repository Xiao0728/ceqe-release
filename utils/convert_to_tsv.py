import pandas as pd


def convert_vocab_to_tsv(vocab):
    vocab_df_bert = pd.DataFrame({
        'id': range(len(vocab)),
        'label': [0] * len(vocab),  #
        'alpha': ['a'] * len(vocab),
        'text': vocab
    })
    return vocab_df_bert
