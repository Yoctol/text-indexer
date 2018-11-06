from unittest import TestCase

from pathlib import Path
import umsgpack

from ..char_indexer import (
    CharIndexer,
    CharwtWord2Vec,
)


def load_w2v(path: str):
    with open(path, 'rb') as filep:
        word2vec = umsgpack.unpack(filep)
    return word2vec


class CharwtWord2VecTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxlen = 7
        cls.test_emb = load_w2v(
            Path(__file__).resolve().parent.joinpath('data/example.msg'),
        )
        cls.input_data = [
            '克安是牛肉大粉絲',  # longer than 7 after adding sos eos
            '繼良喜歡喝星巴巴',  # longer than 7 after adding sos eos
            '安靜的祥睿',  # equal to 7 after adding sos eos
            '喔',  # shorter than 7 after adding sos eos
        ]

    def setUp(self):
        self.indexer = CharwtWord2Vec(
            word2vec=self.test_emb,
            maxlen=self.maxlen,
        )
        self.indexer.fit(self.input_data)

    def test_transform_and_fit_dont_change(self):
        tx_data, meta = self.indexer.transform(self.input_data)
        self.assertEqual(
            [
                [0, 4, 5, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3, 6],
                [0, 5, 8, 3, 10, 3, 1],
                [0, 3, 1, 2, 2, 2, 2],
            ],
            tx_data,
        )
        self.assertEqual(
            [7, 7, 7, 3],
            meta['seqlen'],
        )
        self.indexer.fit(self.input_data)
        self.assertEqual(
            [
                [0, 4, 5, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3, 6],
                [0, 5, 8, 3, 10, 3, 1],
                [0, 3, 1, 3, 3, 3, 3],
            ],
            tx_data,
        )

    def test_inverse_transform(self):
        tx_data, meta = self.indexer.transform(self.input_data)
        output = self.indexer.inverse_transform(tx_data, meta['inv_info'])
        self.assertEqual(
            output,
            self.input_data,
        )

    def test_embedding_correct(self):
        self.assertEqual(
            len(self.indexer.word2vec),
            len(self.test_emb),
        )
