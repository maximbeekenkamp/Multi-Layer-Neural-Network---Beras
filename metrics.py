import numpy as np

from .core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        """Categorical accuracy forward pass!"""
        super().__init__()
        true_pred = np.argmax(labels, axis=-1)
        model_pred = np.argmax(probs, axis=-1)
        return np.mean(true_pred == model_pred)
