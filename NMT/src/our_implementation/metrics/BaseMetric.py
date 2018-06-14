from abc import ABCMeta, abstractmethod


class BaseMetric(metaclass=ABCMeta):
    """
    This provides the base off of which all metrics are created.
    """

    def __init__(self):
        self.params = {}

    @abstractmethod
    def evaluate_hypothesis_single(self, hypothesis: str, reference: str or list):
        """
        This evaluates a single hypothesis against one or multiple references
        :param hypothesis:
        :param reference:
        :return:
        """
        pass

    @abstractmethod
    def evaluate_hypothesis_batch_single(self, hypothesis, references):
        pass

    @abstractmethod
    def evaluate_hypothesis_corpus(self, hypothesis, references):
        pass
