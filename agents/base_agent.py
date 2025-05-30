from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def store(self, *args, **kwargs):
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