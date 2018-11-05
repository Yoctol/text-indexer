import abc
from typing import List, Tuple

from .base import BaseIndexer


class PipeIndexer(BaseIndexer):

    @abc.abstractmethod
    def _build_pipe(self):
        pass

    def transform(
            self,
            utterances: List[str],
        ) -> Tuple[List[List[int]], dict]:
        result, tx_info, intermediates = self.pipe.transform(utterances)
        output_info = {
            'seqlen': self._compute_seqlen(intermediates[0], maxlen=self.maxlen),
            'inv_info': tx_info,
        }
        return result, output_info

    @staticmethod
    def _compute_seqlen(
            sentences: List[List[str]],
            maxlen: int,
        ) -> List[int]:
        output = [0] * len(sentences)
        for i, sent in enumerate(sentences):
            output[i] = min(len(sent), maxlen)
        return output

    def inverse_transform(
            self,
            indices: List[List[int]],
            tx_info: List[dict],
        ) -> List[str]:
        return self.pipe.inverse_transform(indices, tx_info)
