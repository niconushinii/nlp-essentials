import json
import os
from types import SimpleNamespace

def get_lexica(res_dir: str) -> SimpleNamespace:
    with open(os.path.join(res_dir, 'nouns.txt')) as fin: nouns = {noun.strip() for noun in fin}
    with open(os.path.join(res_dir, 'verbs.txt')) as fin: verbs = {verb.strip() for verb in fin}
    with open(os.path.join(res_dir, 'nouns_irregular.json')) as fin: nouns_irregular = json.load(fin)
    with open(os.path.join(res_dir, 'verbs_irregular.json')) as fin: verbs_irregular = json.load(fin)
    with open(os.path.join(res_dir, 'nouns_rules.json')) as fin: nouns_rules = json.load(fin)
    with open(os.path.join(res_dir, 'verbs_rules.json')) as fin: verbs_rules = json.load(fin)

    return SimpleNamespace(
        nouns=nouns,
        verbs=verbs,
        nouns_irregular=nouns_irregular,
        verbs_irregular=verbs_irregular,
        nouns_rules=nouns_rules,
        verbs_rules=verbs_rules
    )
RES_DIR = "dat"
lexica = get_lexica(RES_DIR)

print(len(lexica.nouns))
print(len(lexica.verbs))
print(lexica.nouns_irregular)
print(lexica.verbs_irregular)
print(lexica.nouns_rules)
print(lexica.verbs_rules)

### lematizing

def lemmatize(lexica: SimpleNamespace, word: str) -> str:
    def aux(word: str, vocabs: dict[str, str], irregular: dict[str, str], rules: list[tuple[str, str]]):
        lemma = irregular.get(word, None)
        if lemma is not None: return lemma

        for p, s in rules:
            lemma = word[:-len(p)] + s
            if lemma in vocabs: return lemma

        return None

    word = word.lower()
    lemma = aux(word, lexica.verbs, lexica.verbs_irregular, lexica.verbs_rules)

    if lemma is None:
        lemma = aux(word, lexica.nouns, lexica.nouns_irregular, lexica.nouns_rules)

    return lemma if lemma else word

## test for nouns verbs

nouns = ['studies', 'crosses', 'areas', 'gentlemen', 'vertebrae', 'alumni', 'children', 'crises']
nouns_lemmatized = [lemmatize(lexica, word) for word in nouns]
for word, lemma in zip(nouns, nouns_lemmatized): print('{} -> {}'.format(word, lemma))

verbs = ['applies', 'cried', 'pushes', 'entered', 'takes', 'heard', 'lying', 'studying', 'taking', 'drawn', 'clung', 'was', 'bought']
verbs_lemmatized = [lemmatize(lexica, word) for word in verbs]
for word, lemma in zip(verbs, verbs_lemmatized): print('{} -> {}'.format(word, lemma))

### recount word types

from collections import Counter
from src.tokenization import tokenize

corpus = 'dat/emory-wiki.txt'
delims = {'"', "'", '(', ')', '[', ']', ':', '-', ',', '.'}
words = [lemmatize(lexica, word) for word in tokenize(corpus, delims)]
counts = Counter(words)

print(f'# of word tokens: {len(words)}')
print(f'# of word types: {len(counts)}')

output = 'dat/word_types-token-lemma.txt'
with open(output, 'w') as fout:
    for key in sorted(counts.keys()): fout.write(f'{key}\n')