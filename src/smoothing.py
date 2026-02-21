
from src.ngram_models import unigram_count, test_unigram, bigram_count, test_bigram, Unigram, Bigram

UNKNOWN = ''


def unigram_smoothing(filepath: str) -> Unigram:
    counts = unigram_count(filepath)
    total = sum(counts.values()) + len(counts)
    unigrams = {word: (count + 1) / total for word, count in counts.items()}
    unigrams[UNKNOWN] = 1 / total
    return unigrams


def smoothed_unigram(probs: Unigram, word: str) -> float:
    return probs.get(word, probs[UNKNOWN])


def bigram_smoothing(filepath: str) -> Bigram:
    counts = bigram_count(filepath)
    vocab = set(counts.keys())
    for _, css in counts.items():
        vocab.update(css.keys())

    bigrams = dict()
    for prev, ccs in counts.items():
        total = sum(ccs.values()) + len(vocab)
        d = {curr: (count + 1) / total for curr, count in ccs.items()}
        d[UNKNOWN] = 1 / total
        bigrams[prev] = d

    bigrams[UNKNOWN] = 1 / len(vocab)
    return bigrams


def smoothed_bigram(probs: Bigram, prev: str, curr: str) -> float:
    d = probs.get(prev, None)
    return probs[UNKNOWN] if d is None else d.get(curr, d[UNKNOWN])


if __name__ == '__main__':
    corpus = 'dat/chronicles_of_narnia.txt'

    # Unigram Smoothing
    test_unigram(corpus, unigram_smoothing)
    unigram = unigram_smoothing(corpus)
    for word in ['Aslan', 'Jinho']:
        print(f'{word} {smoothed_unigram(unigram, word):.6f}')

    # Bigram Smoothing
    test_bigram(corpus, bigram_smoothing)
    bigram = bigram_smoothing(corpus)
    for word in [('Aslan', 'is'), ('Aslan', 'Jinho'), ('Jinho', 'is')]:
        print(f'{word} {smoothed_bigram(bigram, *word):.6f}')
