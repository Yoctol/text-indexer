import abc
import os
import json
from typing import List, Tuple

from .base import BaseIndexer


class PipeIndexer(BaseIndexer):

    def __init__(
            self,
            sos_token: str,
            eos_token: str,
            pad_token: str,
            unk_token: str,
            maxlen: int = 50,
        ):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.maxlen = maxlen
        self.pipe = self._build_pipe()

    @abc.abstractmethod
    def _build_pipe(self):
        pass

    def word2index(self, word):
        token2index = self.pipe.get_state(3)['token2index']
        if word not in token2index:
            raise KeyError(f'{word} is not in vocab.')
        return token2index[word]

    def index2word(self, index):
        index2token = self.pipe.get_state(3)['index2token']
        if index not in index2token:
            raise KeyError(f'{index} is not in index map.')
        return index2token[index]

    def transform(
            self,
            utterances: List[str],
        ) -> Tuple[List[List[int]], dict]:
        result, tx_info, intermediates = self.pipe.transform(utterances)
        output_info = {
            'seqlen': self._compute_seqlen(intermediates[0], maxlen=self.maxlen),
            'inv_info': tx_info,
        }
        return result, output_info

    @staticmethod
    def _compute_seqlen(
            sentences: List[List[str]],
            maxlen: int,
        ) -> List[int]:
        output = [0] * len(sentences)
        for i, sent in enumerate(sentences):
            output[i] = min(len(sent), maxlen)
        return output

    def inverse_transform(
            self,
            indices: List[List[int]],
            tx_info: List[dict],
        ) -> List[str]:
        return self.pipe.inverse_transform(indices, tx_info)
