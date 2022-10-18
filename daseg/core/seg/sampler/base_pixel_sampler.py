# Obtained from: https://github.com/open-mmlab/dasegmentation/tree/v0.16.0

from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):
    """Base class of pixel sampler."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        """Placeholder for sample function."""
