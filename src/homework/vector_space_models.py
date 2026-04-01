# ========================================================================

__author__ = 'Anushka Basu'

import math
from collections import Counter

def read(filename: str) -> list[tuple[int, str]]:
    def aux(s: str) -> tuple[int, str]:
        t = s.split('\t')
        return int(t[0]), t[1]

    return [aux(line) for line in open(filename)]


def vocabulary(documents: list[list[str]]) -> dict[str, int]:
    vocab = set()

    for document in documents:
        vocab.update(document)

    return {word: i for i, word in enumerate(sorted(list(vocab)))}


def bag_of_words(vocab: dict[str, int], document: list[str]) -> dict[int, int]:
    counts = Counter(document)
    return {vocab[word]: count for word, count in sorted(counts.items()) if word in vocab}


def document_frequencies(vocab: dict[str, int], corpus: list[list[str]]) -> dict[int, int]:
    counts = Counter()

    for document in corpus:
        counts.update(set(document))

    return {vocab[word]: count for word, count in sorted(counts.items()) if word in vocab}


def tf_idf(vocab: dict[str, int], dfs: dict[int, int], D: int, document: list[str]) -> dict[int, float]:
    tf = lambda count: count / len(document)
    idf = lambda tid: math.log(D / dfs[tid])
    return {tid: tf(count) * idf(tid) for tid, count in bag_of_words(vocab, document).items()}


def cosine_similarity(v1: dict[int, float], v2: dict[int, float]) -> float:
    n = sum(v * v2.get(k, 0) for k, v in v1.items())
    d = math.sqrt(sum(v ** 2 for k, v in v1.items()))
    d *= math.sqrt(sum(v ** 2 for k, v in v2.items()))
    return n / d if d != 0 else 0.0


def vectorize(dat: list[tuple[int, str]], vocab: dict[str, int], dfs: dict[int, int], D: int) -> list[tuple[int, dict[int, float]]]:
    vs = []

    for label, text in dat:
        vs.append((label, tf_idf(vocab, dfs, D, text.split())))

    return vs


def knn(trn_vs: list[tuple[int, dict[int, float]]], v: dict[int, float], k: int = 1) -> tuple[int, float]:
    sims = [(label, cosine_similarity(v, t)) for label, t in trn_vs]
    sims.sort(key=lambda x: x[1], reverse=True)

    topk = sims[:k]
    pred = Counter(label for label, _ in topk).most_common(1)[0][0]
    score = topk[0][1] if topk else 0.0
    return pred, score


# k was selected by searching odd values on the development set.
# in the tested range 131 to 151, the best-performing value was 131.
def sentiment_analyzer(trn_dat: list[tuple[int, str]], tst_dat: list[tuple[int, str]]) -> list[tuple[int, float]]:
    k = 131

    trn_docs = [text.split() for _, text in trn_dat]
    vocab = vocabulary(trn_docs)
    dfs = document_frequencies(vocab, trn_docs)
    D = len(trn_docs)

    trn_vs = vectorize(trn_dat, vocab, dfs, D)
    tst_vs = vectorize(tst_dat, vocab, dfs, D)

    return [knn(trn_vs, v, k) for _, v in tst_vs]


### extra credit

def tf_idf_sublinear(vocab: dict[str, int], dfs: dict[int, int], D: int, document: list[str]) -> dict[int, float]:
    counts = bag_of_words(vocab, document)
    tf = lambda count: 1 + math.log(count) if count > 0 else 0
    idf = lambda tid: math.log(D / dfs[tid])
    return {tid: tf(count) * idf(tid) for tid, count in counts.items()}


def vectorize_extra(dat: list[tuple[int, str]], vocab: dict[str, int], dfs: dict[int, int], D: int) -> list[tuple[int, dict[int, float]]]:
    vs = []

    for label, text in dat:
        vs.append((label, tf_idf_sublinear(vocab, dfs, D, text.split())))

    return vs


def knn_weighted(trn_vs: list[tuple[int, dict[int, float]]], v: dict[int, float], k: int = 1) -> tuple[int, float]:
    sims = [(label, cosine_similarity(v, t)) for label, t in trn_vs]
    sims.sort(key=lambda x: x[1], reverse=True)

    topk = sims[:k]
    if not topk:
        return 2, 0.0

    scores = {}
    counts = {}

    for label, score in topk:
        scores[label] = scores.get(label, 0.0) + score
        counts[label] = counts.get(label, 0) + 1

    pred = max(scores, key=lambda label: (scores[label], counts[label], label))
    avg_score = scores[pred] / counts[pred]
    return pred, avg_score


def sentiment_analyzer_extra(trn_dat: list[tuple[int, str]], tst_dat: list[tuple[int, str]]) -> list[tuple[int, float]]:
    k = 131

    trn_docs = [text.split() for _, text in trn_dat]
    vocab = vocabulary(trn_docs)
    dfs = document_frequencies(vocab, trn_docs)
    D = len(trn_docs)

    trn_vs = vectorize_extra(trn_dat, vocab, dfs, D)
    tst_vs = vectorize_extra(tst_dat, vocab, dfs, D)

    return [knn_weighted(trn_vs, v, k) for _, v in tst_vs]


## accuracy check

if __name__ == '__main__':
    trn_dat = read('dat/sentiment_treebank/sst_trn.tsv')
    dev_dat = read('dat/sentiment_treebank/sst_dev.tsv')

    print('Training size:', len(trn_dat))
    print('Development size:', len(dev_dat))
    print('Baseline k = 131')
    print('Extra-credit k = 131')
    print()

    pred = sentiment_analyzer(trn_dat, dev_dat)

    correct = 0
    for i, (pred_label, score) in enumerate(pred):
        if pred_label == dev_dat[i][0]:
            correct += 1
    print('Baseline Accuracy: {} ({}/{})'.format(100 * correct / len(dev_dat), correct, len(dev_dat)))

    predx = sentiment_analyzer_extra(trn_dat, dev_dat)

    correct = 0
    for i, (pred_label, score) in enumerate(predx):
        if pred_label == dev_dat[i][0]:
            correct += 1
    print('Extra Accuracy: {} ({}/{})'.format(100 * correct / len(dev_dat), correct, len(dev_dat)))

    # Baseline Accuracy: 39.69118982742961(437 / 1101)
    # Extra Accuracy: 40.87193460490463(450 / 1101)