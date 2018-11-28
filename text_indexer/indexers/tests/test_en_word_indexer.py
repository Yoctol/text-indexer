from unittest import TestCase

from pathlib import Path

from .template import TestTemplate
from ..en_word_indexer import EnWordIndexer
from .utils import load_w2v, export_word2vec


class EnWordIndexerWithoutW2vTestCase(TestTemplate, TestCase):

    @classmethod
    def get_data(cls):
        cls.maxlen = 10
        cls.input_data = [
            "Jsaon is craving for working in Angel Cafe.",
            "GB loves ramen!",
            "Alvin enthuses over Soylent.",
            "Girls shorter than 150cm are CPH's desire?",
            "Liang's family owns a bank.",
        ]
        cls.tokenized_input_data = [
            ["Jsaon", "is", "craving", "for", "working", "in", "Angel", "Cafe", "."],
            ["GB", "loves", "ramen", "!"],
            ["Alvin", "enthuses", "over", "Soylent", "."],
            ["Girls", "shorter", "than", "150cm", "are", "CPH", "'", "s", "desire", "?"],
            ["Liang", "'", "s", "family", "owns", "a", "bank", "."],
        ]

    def get_indexer_class(self):
        return EnWordIndexer

    def get_indexer(self):
        return EnWordIndexer.create_without_word2vec(
            sos_token=self.sos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            maxlen=self.maxlen,
        )

    def get_correct_idxs_and_seqlen_of_input_data(self):
        correct_idxs = []
        for sent in self.tokenized_input_data:
            sent_idxs = [self.indexer.word2index(self.indexer.sos_token)]
            for word in sent:
                try:
                    sent_idxs.append(
                        self.indexer.word2index(word),
                    )
                except KeyError:
                    sent_idxs.append(
                        self.indexer.word2index(self.indexer.unk_token),
                    )
            sent_idxs.append(self.indexer.word2index(self.indexer.eos_token))
            if len(sent_idxs) > self.maxlen:
                sent_idxs = sent_idxs[:self.maxlen]
            while len(sent_idxs) < self.maxlen:
                sent_idxs.append(self.indexer.word2index(self.indexer.pad_token))
            assert len(sent_idxs) == self.maxlen
            correct_idxs.append(sent_idxs)

        correct_seqs = [
            min(len(sent) + 2, self.maxlen)
            for sent in self.tokenized_input_data
        ]
        return correct_idxs, correct_seqs

    def test_embedding_correct(self):
        self.assertIsNone(self.indexer.word2vec)


class EnWordIndexerWithW2vTestCase(EnWordIndexerWithoutW2vTestCase):

    @classmethod
    def setUpClass(cls):
        word2vec_path = str(Path(__file__).resolve().parent.joinpath('data/en_example.msg'))
        export_word2vec(
            words=["<sos>", "</s>", "<pad>", "<unk>", ".", "'", "?", "!",
                   "Jsaon", "GB", "Alvin", "CPH", "Liang", "is", "are",
                   "loves", "working", "bank", "Girls", "than"],
            path=word2vec_path,
        )
        cls.test_emb = load_w2v(word2vec_path)
        super().setUpClass()

    def get_indexer(self):
        return EnWordIndexer.create_with_word2vec(
            word2vec=self.test_emb,
            sos_token=self.sos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            maxlen=self.maxlen,
        )

    def test_embedding_correct(self):
        self.assertEqual(self.indexer.word2vec, self.test_emb)

    def test_transform_and_fit_dont_change(self):
        tx_data, meta = self.indexer.transform(self.input_data)
        correct_idxs, correct_seqs = self.get_correct_idxs_and_seqlen_of_input_data()
        self.assertEqual(correct_idxs, tx_data)
        self.assertEqual(correct_seqs, meta['seqlen'])
        self.indexer.fit(self.input_data)
        self.assertEqual(correct_idxs, tx_data)
        self.assertEqual(correct_seqs, meta['seqlen'])
