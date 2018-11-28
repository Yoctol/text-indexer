from typing import List

import numpy as np
import umsgpack


def load_w2v(path: str):
    with open(path, 'rb') as filep:
        word2vec = umsgpack.unpack(filep)
    return word2vec


def export_word2vec(words: List[str], path: str):
    assert len(words) == len(set(words))
    word2vec = {
        'token2index': {word: i for i, word in enumerate(words)},
        'vector': np.random.rand(len(words), 10).tolist(),
    }
    with open(path, 'wb') as filep:
        umsgpack.pack(word2vec, filep)
