from abc import ABC, abstractmethod
import numpy as np

class ContentEmbedder(ABC):
    @abstractmethod
    def embed(self, content_dict: dict) -> np.ndarray:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass
