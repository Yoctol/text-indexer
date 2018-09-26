from typing import List, Tuple

import strpipe as sp

from .base import BaseIndexer


class StrpipeLoader(BaseIndexer):

    def __init__(self, path: str):
        self._path = path
        super().__init__()

    def build(self):
        if not self.is_built:
            # restore strpipe
            self.pipe = sp.Pipe.restore_from_json(self._path)
            self.is_built = True

    def transform(
            self,
            utterances: List[str],
        ) -> Tuple[List[List[int]], List[dict]]:
        tx_utt, tx_info = self.pipe.transform(utterances)
        return tx_utt, tx_info

    def inverse_transform(
            self,
            indices: List[List[int]],
            tx_info: List[dict],
        ) -> List[str]:
        return self.pipe.inverse_transform(indices, tx_info)
