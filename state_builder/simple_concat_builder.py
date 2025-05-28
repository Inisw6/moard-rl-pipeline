from state_builder.content_state_builder import ContentStateBuilder
import numpy as np

class SimpleConcatBuilder(ContentStateBuilder):
    def __init__(self, meta_action_dim):
        self.meta_action_dim = meta_action_dim

    def build(self, user_state, meta_action):
        meta_onehot = np.eye(self.meta_action_dim)[meta_action]
        return np.concatenate([user_state, meta_onehot])
