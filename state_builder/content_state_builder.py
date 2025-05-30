from abc import ABC, abstractmethod
import numpy as np

class ContentStateBuilder(ABC):
    @abstractmethod
    def build(self, user_state: np.ndarray, meta_action: int) -> np.ndarray:
        pass
