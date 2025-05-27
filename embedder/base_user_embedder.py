from abc import ABC, abstractmethod
import numpy as np

class BaseUserEmbedder(ABC):
    @abstractmethod
    def embed(self, user_dict: dict) -> np.ndarray:
        """
        Args:
            user_dict: {
                'recent_logs': [...],
                'current_time': datetime
            }
        Returns:
            np.ndarray: state vector
        """
        pass

    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of the state vector"""
        pass