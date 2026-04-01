
__author__ = 'Anushka Basu'

# vectorize

from src.bag_of_words_model import vocabulary
from src.term_weighing import read_corpus, document_frequencies, tf_idf

if __name__ == '__main__':
    corpus = read_corpus('dat/chronicles_of_narnia.txt')
    vocab = vocabulary(corpus)
    dfs = document_frequencies(vocab, corpus)
    D = len(corpus)

    documents = [
        'I like this movie very much'.split(),
        'I hate this movie very much'.split(),
        'I love this movie so much'.split()
    ]

    vs = [tf_idf(vocab, dfs, D, document) for document in documents]
    for v in vs: print(vs)


# euclidean similarity

import math
from src.bag_of_words_model import SparseVector

def euclidean_distance(v1: SparseVector, v2: SparseVector) -> float:
    d = sum((v - v2.get(k, 0)) ** 2 for k, v in v1.items())
    d += sum(v ** 2 for k, v in v2.items() if k not in v1)
    return math.sqrt(d)

print(euclidean_distance(vs[0], vs[0]))
print(euclidean_distance(vs[0], vs[1]))
print(euclidean_distance(vs[0], vs[2]))

# cosine similarity

def cosine_similarity(v1: SparseVector, v2: SparseVector) -> float:
    n = sum(v * v2.get(k, 0) for k, v in v1.items())
    d = math.sqrt(sum(v ** 2 for k, v in v1.items()))
    d *= math.sqrt(sum(v ** 2 for k, v in v2.items()))
    return n / d

print(cosine_similarity(vs[0], vs[0]))
print(cosine_similarity(vs[0], vs[1]))
print(cosine_similarity(vs[0], vs[2]))

