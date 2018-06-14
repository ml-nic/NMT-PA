from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.params = {}

    @abstractmethod
    def start_training(self):
        """
        Starts training and saves the model in a directory with a unique name
        """
        pass

    @abstractmethod
    def predict_one_sentence(self, sentence):
        """
        Predicts a translation based on the source sentence.
        Preprocessing and postprocessing is also handled.
        :param sentence: The sentence which should be translated
        :return: the translated sentence as a string
        """
        pass

    @abstractmethod
    def predict_batch(self, sentences):
        pass

    @abstractmethod
    def calculate_hiddenstate_after_encoder(self, sentence):
        pass

    @abstractmethod
    def calculate_hiddenstate_after_decoder(self, sentence):
        pass

    @abstractmethod
    def calculate_every_hiddenstate_after_encoder(self, sentence):
        pass

    @abstractmethod
    def calculate_every_hiddenstate(self, sentence):
        pass
