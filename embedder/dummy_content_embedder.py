from embedder.content_embedder import ContentEmbedder
import numpy as np

class DummyContentEmbedder(ContentEmbedder):
    def __init__(self, dim=10):
        self._dim = dim

    def embed(self, content_dict):
        return np.random.randn(self._dim)  # or fixed content vector

    def output_dim(self):
        return self._dim
