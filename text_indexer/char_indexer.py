from typing import List
import warnings

import strpipe as sp

from .pipe_indexer import PipeIndexer


class CharIndexer(PipeIndexer):

    def __init__(
            self,
            word2vec=None,
            sos_token: str = '<sos>',
            eos_token: str = '</s>',
            pad_token: str = '<pad>',
            unk_token: str = '<unk>',
            maxlen: int = 50,
        ):
        self.word2vec = word2vec
        super().__init__(
            sos_token=sos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            maxlen=maxlen,
        )

    @classmethod
    def create_without_word2vec(
            cls,
            sos_token: str = '<sos>',
            eos_token: str = '</s>',
            pad_token: str = '<pad>',
            unk_token: str = '<unk>',
            maxlen: int = 50,
        ):
        indexer = cls(
            sos_token=sos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            maxlen=maxlen,
        )
        return indexer

    @classmethod
    def create_with_word2vec(
            cls,
            word2vec,
            sos_token: str = '<sos>',
            eos_token: str = '</s>',
            pad_token: str = '<pad>',
            unk_token: str = '<unk>',
            maxlen: int = 50,
        ):
        indexer = cls(
            word2vec=word2vec,
            sos_token=sos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            maxlen=maxlen,
        )
        return indexer

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
        if self.word2vec is None:
            p.add_step_by_op_name(
                'TokenToIndex',
                op_kwargs={
                    'unk_token': self.unk_token,
                },
            )
        else:
            p.add_step_by_op_name(
                'TokenToIndex',
                op_kwargs={
                    'unk_token': self.unk_token,
                    'token2index': self.word2vec['token2index'],
                },
            )
        return p

    def fit(self, utterances: List[str]):
        if self.word2vec is None:
            self.pipe.fit(utterances)
        else:
            warnings.warn(
                "CharwtWord2Vec fit function doesn't actually fit on utterances.",
                UserWarning,
            )
            self.pipe.fit(['dummy fit'])
