import numpy as np


def read_word_embeddings(path: str) -> dict[str, np.ndarray]:
    embeddings = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=float)
            embeddings[word] = vector

    return embeddings


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def similar_words(
    word_embeddings: dict[str, np.ndarray],
    target_word: str,
    threshold: float
) -> list[tuple[str, float]]:
    if target_word not in word_embeddings:
        return []

    target_vector = word_embeddings[target_word]
    results = []

    for word, vector in word_embeddings.items():
        if word == target_word:
            continue

        similarity = _cosine_similarity(target_vector, vector)
        if similarity >= threshold:
            results.append((word, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _document_embedding(
    word_embeddings: dict[str, np.ndarray],
    document: str
) -> np.ndarray | None:
    words = document.split()
    vectors = [word_embeddings[word] for word in words if word in word_embeddings]

    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def document_similarity(
    word_embeddings: dict[str, np.ndarray],
    document1: str,
    document2: str
) -> float:
    doc_vector1 = _document_embedding(word_embeddings, document1)
    doc_vector2 = _document_embedding(word_embeddings, document2)

    if doc_vector1 is None or doc_vector2 is None:
        return 0.0

    return _cosine_similarity(doc_vector1, doc_vector2)