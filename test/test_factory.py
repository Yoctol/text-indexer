from unittest import TestCase

from pathlib import Path
from ..char_with_word2vec import CharwtWord2Vec
from ..factory import IndexerFactory


class IndexerFactoryTestCase(TestCase):

    def setUp(self):
        self.maxlen = 7
        self.indexer = CharwtWord2Vec(
            word2vec_path=str(
                Path(__file__).resolve().parent.joinpath('data/example.msg')),
            maxlen=self.maxlen,
        )

    def test_register_exists(self):
        factory = IndexerFactory()
        factory.register('test indexer', self.indexer)
        self.assertEqual(
            self.indexer,
            factory.indexers['test indexer'],
        )

    def test_regitster_already_exists(self):
        factory = IndexerFactory()
        factory.register('test indexer', self.indexer)
        with self.assertRaise(KeyError):
            factory.register('test indexer', self.indexer)

    def test_get_indexer_not_exists(self):
        factory = IndexerFactory()
        with self.assertRaise(KeyError):
            factory.get_indexer('some random key')

    def test_get_indexer_exists_correctly(self):
        factory = IndexerFactory()
        factory.register('test indexer', self.indexer)
        self.assertEqual(
            self.indexer,
            factory.get_indexer('test indexer'),
        )

    def test__getitem__and__setitem(self):
        factory = IndexerFactory()
        factory['test indexer'] = self.indexer
        self.assertEqual(
            self.indexer,
            factory['test indexer'],
        )
