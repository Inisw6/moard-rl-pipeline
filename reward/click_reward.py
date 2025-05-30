from reward.reward_function import RewardFunction

class ClickReward(RewardFunction):
    def calculate(self, content_info):
        return float(content_info.get("clicked", 0))