from abc import ABC, abstractmethod

class RewardFunction(ABC):
    @abstractmethod
    def calculate(self, content_info: dict) -> float:
        pass

class DefaultRewardFunction(RewardFunction):
    def calculate(self, content_info: dict) -> float:
        """
        예시 보상: 클릭 여부*1.0 + 체류시간*0.01 + 감정*0.1
        content_info 예시: {'clicked':1, 'dwell':25, 'emotion':0.8}
        """
        click  = content_info.get("clicked", 0)
        dwell  = content_info.get("dwell", 0)
        emotion= content_info.get("emotion", 0)
        return click * 1.0 + dwell * 0.01 + emotion * 0.1
