from typing import List, Tuple, Dict
import os

import strpipe as sp

from .base import BaseIndexer
from .utils import save_json, load_json


# class Mixin:

#     def seqlen(self, intermediates):
#         # depend on maxlen
#         sentences = intermediates[0]
#         output = [0] * len(sentences)
#         for i, sent in enumerate(sentences):
#             output[i] = min(len(sent), self.maxlen)
#         return output

#     def word2index(self, word):
#         token2index = self.pipe.get_state(3)['token2index']
#         if word not in token2index:
#             raise KeyError('The word [{}] is not in vocab'.format(word))
#         return token2index[word]

#     def index2token(self, index):
#         index2token = self.pipe.get_state(3)['index2token']
#         return index2token[index]


class CharwtWord2Vec(BaseIndexer):

    def __init__(
            self,
            sos_token: str = '<sos>',
            eos_token: str = '</s>',
            pad_token: str = '<pad>',
            unk_token: str = '<unk>',
            maxlen: int = 50,
            token2index: Dict[str, int] = None,
        ):
        self.maxlen = maxlen
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token2index = token2index
        super().__init__()

    def _build_pipe(self):
        p = sp.Pipe()
        p.add_step_by_op_name('CharTokenizer')
        p.add_step_by_op_name(
            'AddSosEos',
            op_kwargs={
                'sos_token': self.sos_token,
                'eos_token': self.eos_token,
            },
        )
        p.add_checkpoint()
        p.add_step_by_op_name(
            'Pad',
            op_kwargs={
                'pad_token': self.pad_token,
                'maxlen': self.maxlen,
            },
        )
        p.add_step_by_op_name(
            'TokenToIndex',
            op_kwargs={
                'unk_token': self.unk_token,
                'token2index': self.token2index,
            },
        )
        return p

    def fit(self, data: List[str] = None):
        if data is None:
            data = ['dummy', 'dummy']

        if not self.is_fitted:
            self.pipe = self._build_pipe()
            self.pipe.fit(data)
            self.is_fitted = True
        else:
            print('Has been fitted')

    def transform(
            self,
            data: List[str],
        ) -> Tuple[List[List[int]], dict, List[object]]:
        output, tx_info, intermediates = self.pipe.transform(data)
        return output, tx_info, intermediates

    def inverse_transform(
            self,
            data: List[List[int]],
            tx_info: List[dict],
        ) -> List[str]:
        restored_output = self.pipe.inverse_transform(data, tx_info)
        return restored_output

    def save(self, output_dir: str):
        os.mkdir(output_dir)
        params = {
            "maxlen": self.maxlen,
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "token2index": self.token2index,
        }
        save_json(params, os.path.join(output_dir, 'indexer.json'))
        self.pipe.save_json(os.path.join(output_dir, 'pipe.json'))

    @classmethod
    def load(cls, output_dir):
        params = load_json(os.path.join(output_dir, 'indexer.json'))
        indexer = cls(**params)
        indexer.pipe = sp.Pipe.restore_from_json(os.path.join(output_dir, 'pipe.json'))
        indexer.is_fitted = True
        return indexer
