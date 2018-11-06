from unittest import TestCase
import shutil

from pathlib import Path
from ..char_with_word2vec import CharwtWord2Vec


class CharwtWord2VecTestCase(TestCase):

    def setUp(self):
        self.maxlen = 7
        self.indexer_class = CharwtWord2Vec
        self.indexer = CharwtWord2Vec(
            token2index={
                "<sos>": 0,
                "</s>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "克": 4,
                "安": 5,
                "星": 6,
                "巴": 7,
                "靜": 8,
                "絲": 9,
                "祥": 10,
            },
            maxlen=self.maxlen,
        )
        self.input_data = [
            '克安是牛肉大粉絲',  # longer than 7 after adding sos eos
            '繼良喜歡喝星巴巴',  # longer than 7 after adding sos eos
            '安靜的祥睿',  # equal to 7 after adding sos eos
            '喔',  # shorter than 7 after adding sos eos
        ]
        self.output_dir = Path(__file__).parent / 'example_indexer/'

    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(str(self.output_dir))

    def test_correctly_init(self):
        self.assertFalse(self.indexer.is_fitted)

    def test_fit(self):
        self.indexer.fit()
        self.assertTrue(self.indexer.is_fitted)

    def test_transform(self):
        self.indexer.fit()
        tx_data, _, intermediate = self.indexer.transform(self.input_data)
        self.assertEqual(
            [
                [0, 4, 5, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3, 6],
                [0, 5, 8, 3, 10, 3, 1],
                [0, 3, 1, 3, 3, 3, 3],
            ],
            tx_data,
        )
        self.assertEqual(
            [
                [
                    ['<sos>', '克', '安', '是', '牛', '肉', '大', '粉', '絲', '</s>'],
                    ['<sos>', '繼', '良', '喜', '歡', '喝', '星', '巴', '巴', '</s>'],
                    ['<sos>', '安', '靜', '的', '祥', '睿', '</s>'],
                    ['<sos>', '喔', '</s>'],
                ],
            ],
            intermediate,
        )

    def test_inverse_transform(self):
        self.indexer.fit()
        tx_data, meta, _ = self.indexer.transform(self.input_data)
        output = self.indexer.inverse_transform(tx_data, meta)
        self.assertEqual(output, self.input_data)

    def test_save(self):
        self.indexer.fit()
        self.indexer.save(str(self.output_dir))

    def test_load(self):
        self.indexer.fit()
        self.indexer.save(str(self.output_dir))
        indexer = CharwtWord2Vec.load(str(self.output_dir))
        tx_data, _, _ = indexer.transform(self.input_data)
        self.assertEqual(
            [
                [0, 4, 5, 3, 3, 3, 3],
                [0, 3, 3, 3, 3, 3, 6],
                [0, 5, 8, 3, 10, 3, 1],
                [0, 3, 1, 3, 3, 3, 3],
            ],
            tx_data,
        )

    # def test_seqlen(self):
    #     self.indexer.fit()
    #     _, _, intermediate = self.indexer.transform(self.input_data)
    #     seqlen = self.indexer.seqlen(intermediate)
    #     self.assertEqual([7, 7, 7, 3], seqlen)

    # def test_word2index(self):
    #     self.indexer.fit()
    #     output = self.indexer.word2index('克')

    # def test_index2word(self):
    #     self.indexer.fit()
    #     output = self.indexer.index2word(3)
