from typing import List, Tuple

from abc import abstractmethod, ABC


class BaseIndexer(ABC):

    @abstractmethod
    def fit(self, utterances: List[str]):
        pass

    @abstractmethod
    def transform(
            self,
            data: List[str],
        ) -> Tuple[List[List[int]], dict]:
        """Transform strings to indices"""
        pass

    @abstractmethod
    def inverse_transform(
            self,
            data: List[List[int]],
        ) -> List[str]:
        """Restore indices to strings"""
        pass
