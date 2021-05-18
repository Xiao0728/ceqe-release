import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def normalize(query):
    query = query.lower().replace("-", " ").replace("'", " '").encode("ascii", errors="ignore").decode()
    return clean_string(query)


"""
    Ensure a safe galago query term.
"""


def clean_string(query):
    return re.sub("[^a-zA-Z0-9 ]", "", query)


def remove_stop_words(query_text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(query_text)
    query_stopped = [w for w in word_tokens if not w in stop_words]
    return ' '.join(query_stopped)


def remove_punct(query_text, translator):
    return query_text.translate(translator)
