import tqdm
import pandas as pd


def load_qrels(path):
    # TODO implement loading queries
    pass


def load_queries(path):
    # panda datframe format: id and query text
    queries = pd.read_csv(path, delimiter='\t', encoding='utf-8', header=None)
    # list of (id,query) tuples
    queries_list = list(queries.itertuples(index=False, name=None))
    return queries_list


def load_queries_dict(path):
    queries = {}
    with open(path) as f:
        for line in f:
            splitted = line.strip().split('\t')
            q_id = splitted[0]
            q_text = splitted[1]
            queries[q_id] = q_text
    return queries


def load_collection(path):
    collection = []
    with open(path) as f:
        for line in f:
            doc_id, doc_text = line.strip().split('\t')
            collection.append((doc_id, doc_text))
    return collection


def load_run(path):
    result = {}
    with open(path) as f:
        for line in f:
            splitted  = line.strip().split(' ')
            rank = int(splitted[-3])
            score = float(splitted[-2])
            q = splitted[0]
            d = " ".join(splitted[2:-3]).strip()
            result.setdefault(q, {})
            result[q][d] = score
    return result


def get_term_df(path):
    term_df = {}
    with(open(path)) as f:
        for line in tqdm.tqdm(f):
            term, freq, doc_freq = line.strip().split('\t')
            term_df[term] = int(doc_freq)
    return term_df


def get_doc_stats(path):
    doc_stats = []
    with(open(path)) as f:
        for line in tqdm.tqdm(f):
            in_id, ex_id, doc_len = line.strip().split('\t')
            doc_stats.append([in_id, ex_id, int(doc_len)])
    return doc_stats


class Vocab:
    """
    This class contains a list of vocabulary terms and their mappings to term ids. The ids are zero indexed. The index
    zero is corresponding to 'UNKNOWN' terms. The terms that have frequency less than min_freq are the 'UNKNoWN'. The 'UNKNOWN'
    is only defined to start the indexing of the words from 1.

    Attributes:
        id_to_term (list): a list of string containing vocabulary terms.
        term_to_id (dict): a dict of terms (str) to their ids (int).
    """

    def __init__(self):
        self.id_and_term = []
        self.term_frequency = {}

    def load_vocab(self, path, min_freq):
        """
        loading vocabulary terms from the output of Galago's 'dump_term_stats' function. For more information, visit
        https://sourceforge.net/p/lemur/wiki/Galago%20Functions/.

        Args:
            path: The file address of the Galago's dump output.
            min_freq: Minimum term frequency for each valid term. Other terms are assumed to be 'UNKNOWN'.
        """
        id = 1
        with open(path) as f:
            for line in tqdm.tqdm(f):
                term, freq, doc_freq = line.rstrip().split('\t')
                freq = int(freq)
                if freq >= min_freq:
                    self.id_and_term.append([id, term])
                    id += 1
        print(str(id) + ' terms have been loaded to the dictionary with the minimum frequency of ' + str(min_freq))

    def load_term_freq(self, path):
        """ This function read the vocabulary file and save the frequencies"""
        with open(path) as f:
            for line in tqdm.tqdm(f):
                term, freq, doc_freq = line.rstrip().split('\t')
                freq = int(freq)
                self.term_frequency[term] = freq

    def size(self):
        return len(self.id_to_term)

    def get_freq_dist(self):
        # TODO find the distribution and assign min_freq based on it
        pass
