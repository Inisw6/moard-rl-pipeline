from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass
    @abstractmethod
    def store(self, *args):
        pass
    @abstractmethod
    def learn(self):
        pass
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError
    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError
