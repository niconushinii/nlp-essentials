from collections import Counter
from pathlib import Path

def count_words(corpus: str) -> Counter:
    with open(corpus, encoding="utf-8") as fin:
        words = fin.read().split()
    return Counter(words)

def save_output(counts: Counter, outfile: str):
    with open(outfile, "w", encoding="utf-8") as fout:
        for word in sorted(counts.keys()):
            fout.write(f"{word}\n")

if __name__ == "__main__":
    corpus = "dat/emory-wiki.txt"
    counts = count_words(corpus)

    n_tokens = sum(counts.values())
    n_types = len(counts)

    print(f"# of word tokens: {n_tokens}")
    print(f"# of word types: {n_types}")

    des = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    asc = sorted(counts.items(), key=lambda x: x[1])

    for word, count in des[:10]:
        print(word, count)
    for word, count in asc[:10]:
        print(word, count)

    save_output(counts, "dat/word_types.txt")
    print("Saved to:", Path("dat/word_types.txt").resolve())
