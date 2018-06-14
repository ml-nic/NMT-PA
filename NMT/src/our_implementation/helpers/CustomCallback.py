import keras
import numpy as np

from metrics.Bleu import Bleu


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, de_word_index, start_token, end_token, val_input_data_preprocessed, val_target_data,
                 real_epochs):
        super(CustomCallback, self).__init__()

        self.de_word_index = de_word_index
        self.START_TOKEN = start_token
        self.END_TOKEN = end_token

        self.val_input_data_preprocessed = val_input_data_preprocessed
        self.val_target_data = val_target_data
        self.real_epochs = real_epochs


    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.real_epochs == 0:
            print("now callback:")

            predict_batch_size = 512
            from_idx = 0
            to_idx = from_idx + predict_batch_size
            self.predicted_sentences = []
            while True:
                to_predict = self.val_input_data_preprocessed[from_idx:to_idx]
                prediction = self.model.predict(to_predict, batch_size=predict_batch_size)
                from_idx += predict_batch_size
                to_idx = from_idx + predict_batch_size
                self.convert_prediction_to_sentence(prediction)
                if to_idx >= len(self.val_input_data_preprocessed):
                    break

            # TODO: may write some predictions to file

            bleu_score = Bleu("WordBasedSeq2Seq1000Units20EpochsGLOVE", "Bleu_corpus",
                              epoch / self.real_epochs).evaluate_hypothesis_corpus(
                self.predicted_sentences, self.val_target_data)

    def convert_prediction_to_sentence(self, predictions):
        print(predictions.shape)
        print("predictions finished")
        reverse_word_index = dict((i, word) for word, i in self.de_word_index.items())

        for sentence in predictions:
            predicted_sentence = ""
            for token in sentence:
                max_idx = np.argmax(token)
                if max_idx == 0:
                    print("id of max token = 0")
                    print("second best prediction is ", reverse_word_index[np.argmax(np.delete(token, max_idx))])
                else:
                    next_word = reverse_word_index[max_idx]
                    if next_word == self.END_TOKEN:
                        break
                    elif next_word == self.START_TOKEN:
                        continue
                    predicted_sentence += next_word + " "
            self.predicted_sentences.append(predicted_sentence)
