from collections import Counter
from src.frequency_analysis import save_output
from pathlib import Path

def delimit(word: str, delimiters: set[str]) -> list[str]:
    i = next((i for i, c in enumerate(word) if c in delimiters), -1)
    if i < 0: return [word]
    tokens = []

    if i > 0: tokens.append(word[:i])
    tokens.append(word[i])

    if i + 1 < len(word):
        tokens.extend(delimit(word[i + 1:], delimiters))

    return tokens

### L1: Support for type hints.
    # L2: Find the index of the first character in word that is in a set of delimiters (enumerate(), next()). If no delimiter is found in word, return -1 (generator expressions).
   # L3: If no delimiter is found, return a list containing word as a single token.
   # L4: If a delimiter is found, create a list tokens to store the individual tokens.
    # L6: If the delimiter is not at the beginning of word, add the characters before the delimiter as a token to tokens.
   # L7: Add the delimiter itself as a separate token to tokens.
   #  L9-10: If there are characters after the delimiter, call delimit() recursively on the remaining part of word and extend() the tokens list with the result.
###

delims = {'"', "'", '(', ')', '[', ']', ':', '-', ',', '.'}

input = [
    '"R1:',
    '(R&D)',
    '15th-largest',
    'Atlanta,',
    "Department's",
    'activity"[26]',
    'centers.[21][22]',
    '149,000',
    'U.S.'
]

output = [delimit(word, delims) for word in input]

for word, tokens in zip(input, output):
    print('{:<16} -> {}'.format(word, tokens))

    ### postproess

def postprocess(tokens: list[str]) -> list[str]:
    i, new_tokens = 0, []

    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == "'" and tokens[i + 1].lower() == "s":
            new_tokens.append("".join(tokens[i:i + 2]))
            i += 1

        elif i + 2 < len(tokens) and (
            (tokens[i] == "[" and tokens[i + 1].isnumeric() and tokens[i + 2] == "]")
            or (tokens[i].isnumeric() and tokens[i + 1] == "," and tokens[i + 2].isnumeric())
        ):
            new_tokens.append("".join(tokens[i:i + 3]))
            i += 2

        elif i + 3 < len(tokens) and "".join(tokens[i:i + 4]) == "U.S.":
            new_tokens.append("".join(tokens[i:i + 4]))
            i += 3

        else:
            new_tokens.append(tokens[i])

        i += 1

    return new_tokens

output = [postprocess(delimit(word, delims)) for word in input]

for word, tokens in zip(input, output):
    print('{:<16} -> {}'.format(word, tokens))

def tokenize(corpus: str, delimiters: set[str]) -> list[str]:
    with open(corpus) as fin:
        words = fin.read().split()
    return [token for word in words for token in postprocess(delimit(word, delimiters))]


from collections import Counter
from src.frequency_analysis import save_output

corpus = "C:/Users/Abasu/PycharmProjects/nlp-essentials/dat/emory-wiki.txt"
output = "C:/Users/Abasu/PycharmProjects/nlp-essentials/dat/word_types-token.txt"

words = tokenize(corpus, delims)
counts = Counter(words)

print(f'# of word tokens: {len(words)}')
print(f'# of word types: {len(counts)}')

save_output(counts, output)