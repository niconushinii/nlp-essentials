from collections import Counter, defaultdict
from math import log
from string import punctuation

UNKNOWN = ''
INIT = '[INIT]'


# ==================================================
# Task 1: Bigram Modeling
# ==================================================

def bigram_model(filepath):
    """
    Build a bigram model with:
    - INIT before the first token of each line
    - Laplace smoothing (normalized)
    - UNKNOWN fallback
    """
    counts = defaultdict(Counter)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue

            prev_word = INIT
            for curr_word in tokens:
                counts[prev_word][curr_word] += 1
                prev_word = curr_word

    model = {}
    unknown_probs = []

    for prev_word, curr_counts in counts.items():
        total_count = sum(curr_counts.values())
        v_i = len(curr_counts)

        # Include an explicit UNKNOWN bucket in each row
        denominator = total_count + v_i + 1

        row = {}
        for curr_word, count in curr_counts.items():
            row[curr_word] = (count + 1) / denominator

        row[UNKNOWN] = 1 / denominator
        model[prev_word] = row
        unknown_probs.append(row[UNKNOWN])

    # Fallback for unseen previous words
    model[UNKNOWN] = {UNKNOWN: min(unknown_probs) if unknown_probs else 1.0}

    return model


# ==================================================
# Task 2: Sequence Generation
# ==================================================

def _is_punctuation(token):
    return bool(token) and all(ch in punctuation for ch in token)


def _get_bigram_prob(model, prev_word, curr_word):
    if prev_word in model:
        row = model[prev_word]
        return row.get(curr_word, row[UNKNOWN])
    return model[UNKNOWN][UNKNOWN]


def _get_vocabulary(model):
    vocab = set()

    for prev_word, row in model.items():
        if prev_word not in {INIT, UNKNOWN}:
            vocab.add(prev_word)

        if isinstance(row, dict):
            for curr_word in row:
                if curr_word not in {INIT, UNKNOWN}:
                    vocab.add(curr_word)

    return sorted(vocab)


def sequence_generator(model, initial_word, length):
    """
    Generate a greedy sequence starting with initial_word.
    """
    if length <= 0:
        return [], 0.0

    vocab = _get_vocabulary(model)
    sequence = [initial_word]

    max_punctuation = length // 5
    punctuation_count = 1 if _is_punctuation(initial_word) else 0

    used_non_punct = set()
    if not _is_punctuation(initial_word):
        used_non_punct.add(initial_word)

    current_word = initial_word
    score = log(_get_bigram_prob(model, INIT, initial_word))

    while len(sequence) < length:
        candidates = []

        for token in vocab:
            if _is_punctuation(token):
                if punctuation_count >= max_punctuation:
                    continue
            else:
                if token in used_non_punct:
                    continue

            prob = _get_bigram_prob(model, current_word, token)
            candidates.append((prob, token))

        if not candidates:
            raise ValueError(
                "No valid next token found. Try a different initial word or a shorter length."
            )

        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_prob, next_word = candidates[0]

        sequence.append(next_word)
        score += log(best_prob)

        if _is_punctuation(next_word):
            punctuation_count += 1
        else:
            used_non_punct.add(next_word)

        current_word = next_word

    return sequence, score


# ==================================================
# Extra Credit
# ==================================================

def sequence_generator_plus(model, initial_word, length, beam_width=5):
    """
    Beam-search version for better global sequences.
    """
    if length <= 0:
        return [], 0.0

    vocab = _get_vocabulary(model)
    max_punctuation = length // 5

    initial_punct = 1 if _is_punctuation(initial_word) else 0
    initial_used = set()
    if not _is_punctuation(initial_word):
        initial_used.add(initial_word)

    initial_score = log(_get_bigram_prob(model, INIT, initial_word))
    beams = [([initial_word], initial_score, initial_used, initial_punct)]

    for _ in range(1, length):
        new_beams = []

        for seq, score, used_non_punct, punct_count in beams:
            current_word = seq[-1]

            for token in vocab:
                if _is_punctuation(token):
                    if punct_count >= max_punctuation:
                        continue
                    new_punct_count = punct_count + 1
                    new_used = used_non_punct
                else:
                    if token in used_non_punct:
                        continue
                    new_punct_count = punct_count
                    new_used = set(used_non_punct)
                    new_used.add(token)

                prob = _get_bigram_prob(model, current_word, token)
                new_seq = seq + [token]
                new_score = score + log(prob)

                new_beams.append((new_seq, new_score, new_used, new_punct_count))

        if not new_beams:
            raise ValueError(
                "No valid continuation found. Try a different initial word or a shorter length."
            )

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    best_seq, best_score, _, _ = max(beams, key=lambda x: x[1])
    return best_seq, best_score


if __name__ == '__main__':
    model = bigram_model('dat/chronicles_of_narnia.txt')

    print("INIT:")
    for word, prob in sorted(model[INIT].items(), key=lambda x: -x[1])[:10]:
        print(word, prob)

    print("\nSequence from Lucy:")
    seq, score = sequence_generator(model, 'Lucy', 10)
    print(seq)
    print(score)