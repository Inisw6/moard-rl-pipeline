from embedder.base_content_embedder import BaseContentEmbedder
import numpy as np

class DummyContentEmbedder(BaseContentEmbedder):
    def __init__(self, dim=10):
        self._dim = dim

    def embed(self, content_dict):
        return np.random.randn(self._dim)  # or fixed content vector

    def output_dim(self):
        return self._dim
