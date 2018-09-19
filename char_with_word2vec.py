from typing import List, Tuple

import strpipe as sp
import umsgpack

from .base import BaseIndexer


class CharwtWord2Vec(BaseIndexer):

    def __init__(
            self,
            word2vec_path: str,
            sos_token: str = '<sos>',
            eos_token: str = '</s>',
            pad_token: str = '<pad>',
            unk_token: str = '<unk>',
            maxlen: int = 50,
        ):

        self.maxlen = maxlen
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2vec_path = word2vec_path
        super().__init__()

    def build(self):
        if not self.is_built:
            self.word2vec = self._load_word2vec(path=self.word2vec_path)
            self.pipe = self._build_pipe()
            self.is_built = True

    def _load_word2vec(self, path: str):
        with open(path, 'rb') as filep:
            word2vec = umsgpack.unpack(filep)
        return word2vec

    def _build_pipe(self):
        p = sp.Pipe()
        p.add_step_by_op_name('CharTokenizer')
        p.add_step_by_op_name(
            'Pad',
            op_kwargs={
                'sos_token': self.sos_token,
                'eos_token': self.eos_token,
                'pad_token': self.pad_token,
                'maxlen': self.maxlen,
            },
        )
        p.add_step_by_op_name(
            'TokenToIndexWithUNK',
            op_kwargs={
                'unk_token': self.unk_token,
                'token2index': self.word2vec['token2index'],
            },
        )
        p.fit(['test test'])
        return p

    def transform(
            self,
            utterances: List[str],
        ) -> Tuple[List[List[int]], dict]:
        tx_utt, tx_info = self.pipe.transform(utterances)
        seqlen = [info['sentlen'] for info in tx_info[1]]
        output_info = {
            'seqlen': seqlen,
            'inv_info': tx_info,
        }
        return tx_utt, output_info

    def inverse_transform(
            self,
            indices: List[List[int]],
            tx_info: List[dict],
        ) -> List[str]:
        return self.pipe.inverse_transform(indices, tx_info)

    def get_embedding(self):
        return self.word2vec['vector']
