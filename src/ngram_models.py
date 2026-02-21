
from typing import TypeAlias

Unigram: TypeAlias = dict[str, float]

from collections import Counter

def unigram_count(filepath: str) -> Counter:
    unigrams = Counter()

    for line in open(filepath):
        words = line.split()
        unigrams.update(words)

    return unigrams

def unigram_estimation(filepath: str) -> Unigram:
    counts = unigram_count(filepath)
    total = sum(counts.values())
    return {word: count / total for word, count in counts.items()}

### test unigram with text file, get top 300 unigrams with highest probabilities

from collections.abc import Callable

def test_unigram(filepath: str, estimator: Callable[[str], Unigram]):
    unigrams = estimator(filepath)
    unigram_list = [(word, prob) for word, prob in sorted(unigrams.items(), key=lambda x: x[1], reverse=True)]

    for word, prob in unigram_list[:300]:
        if word[0].isupper() and word.lower() not in unigrams:
            print(f'{word:>10} {prob:.6f}')

corpus = 'dat/chronicles_of_narnia.txt'
test_unigram(corpus, unigram_estimation)

#### bigram estimation

# ddefine typr
Bigram: TypeAlias = dict[str, Unigram | float]

from collections import Counter, defaultdict

def bigram_count(filepath: str) -> dict[str, Counter]:
    bigrams = defaultdict(Counter)

    for line in open(filepath):
        words = line.split()
        for i in range(1, len(words)):
            bigrams[words[i - 1]].update([words[i]])

    return bigrams

# return dictionary with bigram and probabilities
from typing import TypeAlias

Unigram: TypeAlias = dict[str, float]
Bigram: TypeAlias = dict[str, Unigram | float]

def bigram_estimation(filepath: str) -> Bigram:
    counts = bigram_count(filepath)
    bigrams = dict()

    for prev, ccs in counts.items():
        total = sum(ccs.values())
        bigrams[prev] = {curr: count / total for curr, count in ccs.items()}

    return bigrams

## define function test

def test_bigram(filepath: str, estimator: Callable[[str], Bigram]):
    bigrams = estimator(filepath)
    for prev in ['I', 'the', 'said']:
        print(prev)
        bigram_list = [(curr, prob) for curr, prob in sorted(bigrams[prev].items(), key=lambda x: x[1], reverse=True)]
        for curr, prob in bigram_list[:10]:
            print("{:>10} {:.6f}".format(curr, prob))

test_bigram(corpus, bigram_estimation)