from abc import ABC, abstractmethod
import numpy as np

# Base class for content embedding

class BaseContentEmbedder(ABC):
    @abstractmethod
    def embed(self, content_dict: dict) -> np.ndarray:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass
