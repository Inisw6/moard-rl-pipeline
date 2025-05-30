from abc import ABC, abstractmethod

class RewardFunction(ABC):
    @abstractmethod
    def calculate(self, content_info: dict) -> float:
        pass