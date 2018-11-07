from typing import List, Tuple

from abc import abstractmethod, ABC


class Indexer(ABC):

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: List[str]) -> None:
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

    @abstractmethod
    def save(self, output_path: str):
        pass

    @abstractmethod
    def load(self, output_path: str):
        pass
