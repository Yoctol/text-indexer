import abc
import os
import json
from typing import List, Tuple

from .base import BaseIndexer


class PipeIndexer(BaseIndexer):

    def __init__(
            self,
            sos_token: str,
            eos_token: str,
            pad_token: str,
            unk_token: str,
            maxlen: int = 50,
        ):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.maxlen = maxlen

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

    def save(self, output_dir: str):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        self.pipe.save_json(os.path.join(output_dir, 'pipe.json'))
        with open(os.path.join(output_dir, 'params.json'), 'w') as filp:
            json.dump(
                {
                    'sos_token': self.sos_token,
                    'eos_token': self.eos_token,
                    'pad_token': self.pad_token,
                    'unk_token': self.unk_token,
                    'maxlen': self.maxlen,
                },
                filp,
            )
