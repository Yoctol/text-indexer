from typing import List, Tuple

import strpipe as sp

from .base import Indexer


class LoadedStrpipe(Indexer):

    def __init__(self, path: str):
        self._path = path
        super().__init__()

    def fit(self):
        if not self.is_fitted:
            # restore strpipe
            self.pipe = sp.Pipe.restore_from_json(self._path)
            self.is_fitted = True

    def transform(
            self,
            utterances: List[str],
        ) -> Tuple[List[List[int]], List[dict]]:
        return self.pipe.transform(utterances)

    def inverse_transform(
            self,
            indices: List[List[int]],
            tx_info: List[dict],
        ) -> List[str]:
        return self.pipe.inverse_transform(indices, tx_info)

    def save(self, output_path):
        pass

    @classmethod
    def load(cls, path):
        pass
