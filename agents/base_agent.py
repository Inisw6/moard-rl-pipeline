from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state): ...
    @abstractmethod
    def store(self, *args): ...
    @abstractmethod
    def learn(self): ...
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError
    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError
