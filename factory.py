from .base import BaseIndexer


class IndexerFactory:

    def __init__(self):
        self.indexers = {}

    def list_all(self):
        return list(self.indexers.keys())

    def register(self, name: str, indexer: BaseIndexer):
        if name in self.indexers:
            raise KeyError(
                'Indexer [{}] has existed. Please use another name',
            )
        self.indexers[name] = indexer

    def get_indexer(self, name):
        if name in self.indexers:
            raise KeyError(
                'Indexer [{}] has not found.',
            )
            self.indexers[name].build()
        return self.indexers[name]
