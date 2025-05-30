from abc import ABC, abstractmethod
import numpy as np

# Base class for user state embedding

class BaseUserEmbedder(ABC):
    @abstractmethod
    def embed(self, user_dict: dict) -> np.ndarray:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass
    