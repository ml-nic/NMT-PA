import os

import numpy as np
from helpers.BleuCallback import BleuCallback
from helpers.EvalCallback import EvalCallback
from helpers.Tokenizer import Tokenizer
from keras import callbacks
from keras.engine import Model
from keras.layers import Embedding
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'WordBasedSeq2Seq1000Units20EpochsGLOVEReverseInput'

        self.params['batch_size'] = 64
        self.params['val_batch_size'] = 256
        self.params['epochs'] = 20
        self.params['latent_dim'] = 1000
        self.params['MAX_SEQ_LEN'] = 100
        self.params['EMBEDDING_DIM'] = 300
        self.params['MAX_WORDS_DE'] = 40000
        self.params['MAX_WORDS_EN'] = 40000
        self.params['P_DENSE_DROPOUT'] = 0.2
        self.params['VALIDATION_FREQ'] = 1

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENCE_DIR = '../../Persistence/' + self.identifier
        if not os.path.exists(self.BASIC_PERSISTENCE_DIR):
            os.makedirs(self.BASIC_PERSISTENCE_DIR)
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR, 'Graph')
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.WEIGHT_FILES = []

        dir = os.listdir(self.MODEL_CHECKPOINT_DIR)
        for file in dir:
            if file.endswith("hdf5"):
                self.WEIGHT_FILES.append(os.path.join(self.MODEL_CHECKPOINT_DIR, file))
        self.WEIGHT_FILES.sort(key=lambda x: int(x.split('model.')[1].split('-')[0]))
        if len(self.WEIGHT_FILES) == 0:
            print("no weight files found")
        else:
            self.LATEST_MODELCHKPT = self.WEIGHT_FILES[len(self.WEIGHT_FILES) - 1]
        self.TRAIN_DATA_FILE = os.path.join(self.BASE_DATA_DIR, 'Training/DE_EN_(tatoeba)_train.txt')
        self.VAL_DATA_FILE = os.path.join(self.BASE_DATA_DIR, 'Validation/DE_EN_(tatoeba)_validation.txt')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.PRETRAINED_GLOVE_FILE = os.path.join(self.BASE_DATA_DIR, 'glove.6B.300d.txt')
        self.START_TOKEN = "_GO"
        self.END_TOKEN = "_EOS"
        self.UNK_TOKEN = "_UNK"

        self.preprocessing = False
        self.use_bleu_callback = False

    def __insert_valid_token_at_last_position(self, texts):
        for sent in texts:
            if not (sent[self.params['MAX_SEQ_LEN'] - 1] == 0 or sent[self.params['MAX_SEQ_LEN'] - 1] == 2):
                sent[self.params['MAX_SEQ_LEN'] - 1] = 2

    def __create_vocab(self):
        en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_EN'])
        en_tokenizer.fit_on_texts(self.train_input_texts + self.val_input_texts)
        self.train_input_texts = en_tokenizer.texts_to_sequences(self.train_input_texts)
        self.train_input_texts = pad_sequences(self.train_input_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                               padding='post',
                                               truncating='post')
        self.__insert_valid_token_at_last_position(self.train_input_texts)

        self.val_input_texts = en_tokenizer.texts_to_sequences(self.val_input_texts)
        self.val_input_texts = pad_sequences(self.val_input_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                             padding='post',
                                             truncating='post')
        self.__insert_valid_token_at_last_position(self.val_input_texts)
        self.en_word_index = en_tokenizer.word_index

        de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_DE'])
        de_tokenizer.fit_on_texts(self.train_target_texts + self.val_target_texts)
        self.train_target_texts = de_tokenizer.texts_to_sequences(self.train_target_texts)
        self.train_target_texts = pad_sequences(self.train_target_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                                padding='post',
                                                truncating='post')
        self.__insert_valid_token_at_last_position(self.train_target_texts)
        self.val_target_texts = de_tokenizer.texts_to_sequences(self.val_target_texts)
        self.val_target_texts = pad_sequences(self.val_target_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                              padding='post',
                                              truncating='post')
        self.__insert_valid_token_at_last_position(self.val_target_texts)
        self.de_word_index = de_tokenizer.word_index

        embeddings_index = {}
        filename = self.PRETRAINED_GLOVE_FILE
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        self.num_train_words = self.params['MAX_WORDS_EN'] + 3
        self.en_embedding_matrix = np.zeros((self.num_train_words, self.params['EMBEDDING_DIM']))
        for word, i in self.en_word_index.items():
            if i >= self.params['MAX_WORDS_EN'] + 3:
                continue
            embedding_vector = None
            if word == self.START_TOKEN:
                embedding_vector = self.START_TOKEN_VECTOR
            elif word == self.END_TOKEN:
                embedding_vector = self.END_TOKEN_VECTOR
            elif word == self.UNK_TOKEN:
                embedding_vector = self.UNK_TOKEN_VECTOR
            else:
                embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_vector = self.UNK_TOKEN_VECTOR
            self.en_embedding_matrix[i] = embedding_vector

    def start_training(self):
        if self.preprocessing is True:
            self.START_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
            self.END_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
            self.UNK_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])

            self.train_input_texts, self.train_target_texts = self.__split_data(self.TRAIN_DATA_FILE)
            self.num_train_samples = len(self.train_input_texts)
            self.val_input_texts, self.val_target_texts = self.__split_data(self.VAL_DATA_FILE,
                                                                            save_unpreprocessed_targets=self.use_bleu_callback)
            self.__create_vocab()
            if self.use_bleu_callback is False:
                np.save(self.BASIC_PERSISTENCE_DIR + '/val_target_texts.npy', self.val_target_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/train_target_texts.npy', self.train_target_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/train_input_texts.npy', self.train_input_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/val_input_texts.npy', self.val_input_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy', self.en_word_index)
            np.save(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy', self.de_word_index)
            np.save(self.BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy', self.en_embedding_matrix)

        else:
            self.train_input_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/train_input_texts.npy')
            self.train_target_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/train_target_texts.npy')
            self.val_input_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/val_input_texts.npy')
            self.en_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
            self.de_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy')
            self.en_word_index = self.en_word_index.item()
            self.de_word_index = self.de_word_index.item()
            self.en_embedding_matrix = np.load(self.BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy')

            if self.use_bleu_callback:
                self.val_target_texts_no_preprocessing = []
                lines = open(self.VAL_DATA_FILE, encoding='UTF-8').read().split('\n')
                for line in lines:
                    _, target_text = line.split('\t')
                    self.val_target_texts_no_preprocessing.append(target_text)
            else:
                self.val_target_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/val_target_texts.npy')

        self.num_train_samples = len(self.train_input_texts)

        self.__setup_model(mode='training')

        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_grads=True, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(
            self.MODEL_CHECKPOINT_DIR + '/model.{epoch:03d}-{loss:.3f}.hdf5',
            monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=True, mode='auto',
            period=self.params['epochs'])

        if self.use_bleu_callback:
            bleuCallback = BleuCallback(self.de_word_index, self.START_TOKEN, self.END_TOKEN, self.val_input_texts,
                                        self.val_target_texts_no_preprocessing, self.params['epochs'])
            used_callbacks = [tbCallBack, modelCallback, bleuCallback]
        evalCallback = EvalCallback(self.__serve_batch(self.val_input_texts, self.val_target_texts, 'val'),
                                    int(len(self.val_input_texts) / self.params['val_batch_size']), self.identifier,
                                    frequency=self.params['VALIDATION_FREQ'])
        used_callbacks = [tbCallBack, modelCallback, evalCallback]
        self.M.fit_generator(self.__serve_batch(self.train_input_texts, self.train_target_texts, 'train'),
                             int(len(self.train_input_texts) / self.params['batch_size']),
                             epochs=self.params['epochs'], verbose=2, callbacks=used_callbacks,
                             max_queue_size=1)
        self.M.save(self.model_file)

    def __split_data(self, file, save_unpreprocessed_targets=False):
        """
        Reads the data from the given file.
        The two languages in the file have to be splitted by a tab
        :param file: file which should be read from
        :return: (input_texts, target_texts)
        """
        input_texts = []
        target_texts = []
        lines = open(file, encoding='UTF-8').read().split('\n')
        for line in lines:
            input_text, target_text = line.split('\t')
            input_texts.append(input_text)
            target_texts.append(target_text)
        if save_unpreprocessed_targets is True:
            self.val_target_texts_no_preprocessing = target_texts.copy()
        assert len(input_texts) == len(target_texts)
        return input_texts, target_texts

    def load(file):
        """
        Loads the given file into a list.
        :param file: the file which should be loaded
        :return: list of data
        """
        with(open(file, encoding='utf8')) as file:
            data = file.readlines()
            # data = []
            # for i in range(MAX_SENTENCES):
            #    data.append(lines[i])
        print('Loaded', len(data), "lines of data.")
        return data

    def __serve_batch(self, input_texts, target_texts, mode):
        batch_size = self.params['batch_size']
        if mode != 'train':
            batch_size = self.params['val_batch_size']

        counter = 0
        batch_X = np.zeros((batch_size, self.params['MAX_SEQ_LEN']), dtype='int16')
        batch_Y = np.zeros((batch_size, self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
                           dtype='int16')
        while True:
            for i in range(input_texts.shape[0]):
                in_X = input_texts[i]
                out_Y = np.zeros((1, target_texts.shape[1], self.params['MAX_WORDS_DE'] + 3), dtype='int16')
                token_counter = 0
                for token in target_texts[i]:
                    out_Y[0, token_counter, :] = to_categorical(token, num_classes=self.params['MAX_WORDS_DE'] + 3)
                    token_counter += 1
                batch_X[counter] = in_X
                batch_Y[counter] = out_Y
                counter += 1
                if counter == batch_size:
                    print("counter == batch_size", i, mode)
                    counter = 0
                    yield batch_X, batch_Y

    def __setup_model(self, mode=None):
        if mode not in ['predict', 'training']:
            exit("wrong mode for setup_model")

        if mode == 'predict':
            try:
                test = self.en_embedding_matrix
                test = self.M
                return
            except AttributeError:
                pass
            self.en_embedding_matrix = np.load(self.BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy')

        self.M = Sequential()
        self.M.add(
            Embedding(self.params['MAX_WORDS_EN'] + 3, self.params['EMBEDDING_DIM'], weights=[self.en_embedding_matrix],
                      mask_zero=True, trainable=False))

        self.M.add(LSTM(self.params['latent_dim'], return_sequences=True, name='encoder', go_backwards=True))

        self.M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        # M.add(LSTM(self.params['latent_dim'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))
        self.M.add(LSTM(self.params['latent_dim'], return_sequences=True))

        self.M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        self.M.add(TimeDistributed(Dense(self.params['MAX_WORDS_DE'] + 3,
                                         input_shape=(
                                             None, self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
                                         activation='softmax')))

        print('compiling')

        self.M.compile(optimizer='Adam', loss='categorical_crossentropy')
        self.M.summary()

        print('compiled')

        if mode == 'predict':
            self.M.load_weights(self.LATEST_MODELCHKPT)

    def __setup_helpers(self):
        try:
            self.en_word_index
            return
        except Exception as e:
            pass
        self.en_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
        self.de_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy')
        self.de_word_index = self.de_word_index.item()
        self.en_word_index = self.en_word_index.item()

        self.en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                      num_words=self.params['MAX_WORDS_EN'])
        self.en_tokenizer.word_index = self.en_word_index
        self.en_tokenizer.num_words = self.params['MAX_WORDS_EN'] + 3

        self.de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                      num_words=self.params['MAX_WORDS_DE'])
        self.de_tokenizer.word_index = self.de_word_index
        self.de_tokenizer.num_words = self.params['MAX_WORDS_DE'] + 3

    def predict_one_sentence(self, sentence):
        self.__setup_model(mode='predict')
        self.__setup_helpers()

        sentence = self.en_tokenizer.texts_to_sequences([sentence], print_unk_warning=True)
        sentence = pad_sequences(sentence, maxlen=self.params['MAX_SEQ_LEN'],
                                 padding='post',
                                 truncating='post')
        self.__insert_valid_token_at_last_position(sentence)
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])

        prediction = self.M.predict(sentence)

        predicted_sentence = ""
        reverse_word_index = dict((i, word) for word, i in self.de_word_index.items())
        for sentence in prediction:
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

        return predicted_sentence

    def predict_batch(self, sentences, all_weights=False):
        self.__setup_model(mode='predict')
        self.__setup_helpers()

        sentences = self.en_tokenizer.texts_to_sequences(sentences, print_unk_warning=True)
        sentences = pad_sequences(sentences, maxlen=self.params['MAX_SEQ_LEN'],
                                  padding='post',
                                  truncating='post')
        self.__insert_valid_token_at_last_position(sentences)
        sentences = sentences.reshape(sentences.shape[0], sentences.shape[1])

        predictions_for_weights = {}
        if all_weights is True:
            for weight_file in self.WEIGHT_FILES:
                self.M.load_weights(weight_file)
                predictions_for_weights[weight_file.split('model.')[1]] = self.__predict_batch_for_specific_weight(
                    sentences)
        else:
            predictions_for_weights[
                self.LATEST_MODELCHKPT.split('model.')[1]] = self.__predict_batch_for_specific_weight()

        return predictions_for_weights

    def __predict_batch_for_specific_weight(self, sentences):
        batch_size = sentences.shape[0]
        if batch_size > 20:
            batch_size = 20
        reverse_word_index = dict((i, word) for word, i in self.de_word_index.items())

        predicted_sentences = []
        from_idx = 0
        to_idx = batch_size
        while True:
            print("from_idx, to_idx, hm_sentences", from_idx, to_idx, sentences.shape[0])
            current_batch = sentences[from_idx:to_idx]
            prediction = self.M.predict(current_batch, batch_size=batch_size)

            for sentence in prediction:
                predicted_sent = ""
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
                        predicted_sent += next_word + " "
                predicted_sentences.append(predicted_sent)
            from_idx += batch_size
            to_idx += batch_size

            if from_idx > sentences.shape[0]:
                break
            elif from_idx == sentences.shape[0]:
                to_idx = from_idx + 1
            elif to_idx > sentences.shape[0] and from_idx < sentences.shape[0]:
                to_idx = sentences.shape[0] + 1
            elif to_idx > sentences.shape[0]:
                break
        return predicted_sentences

    def calculate_hiddenstate_after_encoder(self, sentences):
        self.__setup_model(mode='predict')
        self.__setup_helpers()

        sentences = self.en_tokenizer.texts_to_sequences(sentences, print_unk_warning=True)
        sentences = pad_sequences(sentences, maxlen=self.params['MAX_SEQ_LEN'],
                                  padding='post',
                                  truncating='post')
        self.__insert_valid_token_at_last_position(sentences)
        sentences = sentences.reshape(sentences.shape[0], sentences.shape[1])
        encoder = Model(inputs=self.M.input, outputs=self.M.get_layer('encoder').output)

        batch_size = sentences.shape[0]
        if batch_size > 20:
            batch_size = 20

        predicted_sentences = []
        from_idx = 0
        to_idx = batch_size

        hiddenstates = []
        while True:
            print("from_idx, to_idx, hm_sentences", from_idx, to_idx, sentences.shape[0])
            current_batch = sentences[from_idx:to_idx]
            hiddenstates.append(encoder.predict(current_batch, batch_size=batch_size))

            from_idx += batch_size
            to_idx += batch_size

            if from_idx > sentences.shape[0]:
                break
            elif from_idx == sentences.shape[0]:
                to_idx = from_idx + 1
            elif to_idx > sentences.shape[0] and from_idx < sentences.shape[0]:
                to_idx = sentences.shape[0] + 1
            elif to_idx > sentences.shape[0]:
                break
        return hiddenstates

    def calculate_every_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate(self, sentence):
        raise NotImplementedError()

    def calculate_hiddenstate_after_decoder(self, sentence):
        raise NotImplementedError()

    def setup_inference(self):
        raise NotImplementedError()
