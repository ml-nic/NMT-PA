import os

import numpy as np
from helpers.BleuCallback import BleuCallback
from keras import callbacks
from keras.engine import Model
from keras.layers import Embedding
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from models.BaseModel import BaseModel


# TODO: Use FastText
class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'WordBasedSeq2Seq1000Units20EpochsFastText'

        self.params['batch_size'] = 64
        self.params['epochs'] = 20
        self.params['latent_dim'] = 1000
        self.params['MAX_SEQ_LEN'] = 100
        self.params['EMBEDDING_DIM'] = 300
        self.params['MAX_WORDS_DE'] = 35000
        self.params['MAX_WORDS_EN'] = 20000
        self.params['P_DENSE_DROPOUT'] = 0.2

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENCE_DIR = '../../Persistence/' + self.identifier
        if not os.path.exists(self.BASIC_PERSISTENCE_DIR):
            os.makedirs(self.BASIC_PERSISTENCE_DIR)
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR, 'Graph')
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)

        self.TRAIN_DATA_FILE = os.path.join(self.BASE_DATA_DIR, 'Training/DE_EN_(tatoeba)_train.txt')
        self.VAL_DATA_FILE = os.path.join(self.BASE_DATA_DIR, 'Validation/DE_EN_(tatoeba)_validation.txt')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.LATEST_MODELCHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR, 'model.878-1.90.hdf5')

        self.START_TOKEN = "_GO"
        self.END_TOKEN = "_EOS"
        self.UNK_TOKEN = "_UNK"

        self.preprocessing = True

    def __create_vocab(self):
        en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_EN'])
        en_tokenizer.fit_on_texts(self.train_input_texts + self.val_input_texts)
        self.train_input_texts = en_tokenizer.texts_to_sequences(self.train_input_texts)
        self.train_input_texts = pad_sequences(self.train_input_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                               padding='post',
                                               truncating='post')
        self.val_input_texts = en_tokenizer.texts_to_sequences(self.val_input_texts)
        self.val_input_texts = pad_sequences(self.val_input_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                             padding='post',
                                             truncating='post')
        self.en_word_index = en_tokenizer.word_index

        de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_DE'])
        de_tokenizer.fit_on_texts(self.train_target_texts + self.val_target_texts)
        self.train_target_texts = de_tokenizer.texts_to_sequences(self.train_target_texts)
        self.train_target_texts = pad_sequences(self.train_target_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                                padding='post',
                                                truncating='post')
        self.val_target_texts = de_tokenizer.texts_to_sequences(self.val_target_texts)
        self.val_target_texts = pad_sequences(self.val_target_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                              padding='post',
                                              truncating='post')
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
            else:
                embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                # TODO: test if this is nesessari
                similar_word = self.__find_similar_word_in_embeddings(word)
                embedding_vector = embeddings_index.get(similar_word)
            self.en_embedding_matrix[i] = embedding_vector

    def start_training(self):
        if self.preprocessing is True:
            self.START_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
            self.END_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
            self.UNK_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])

            self.train_input_texts, self.train_target_texts = self.__split_data(self.TRAIN_DATA_FILE)
            self.num_train_samples = len(self.train_input_texts)
            self.val_input_texts, self.val_target_texts = self.__split_data(self.VAL_DATA_FILE)
            self.__create_vocab()

            np.save(self.BASIC_PERSISTENCE_DIR + '/train_input_texts.npy', self.train_input_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/train_target_texts.npy', self.train_target_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/val_input_texts.npy', self.val_input_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/val_target_texts.npy', self.val_target_texts)
            np.save(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy', self.en_word_index)
            np.save(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy', self.de_word_index)
            np.save(self.BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy', self.en_embedding_matrix)

        else:
            self.train_input_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/train_input_texts.npy')
            self.train_target_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/train_target_texts.npy')
            self.val_input_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/val_input_texts.npy')
            self.val_target_texts = np.load(self.BASIC_PERSISTENCE_DIR + '/val_target_texts.npy')
            self.en_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
            self.de_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy')
            self.en_embedding_matrix = np.load(self.BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy')

        self.num_train_samples = len(self.train_input_texts)

        M = Sequential()
        M.add(
            Embedding(self.params['MAX_WORDS_EN'] + 3, self.params['EMBEDDING_DIM'], weights=[self.en_embedding_matrix],
                      mask_zero=True, trainable=False))

        M.add(LSTM(self.params['latent_dim'], return_sequences=True, name='encoder'))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        # M.add(LSTM(self.params['latent_dim'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))
        M.add(LSTM(self.params['latent_dim'], return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(TimeDistributed(Dense(self.params['MAX_WORDS_DE'] + 3,
                                    input_shape=(None, self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
                                    activation='softmax')))

        print('compiling')

        M.compile(optimizer='Adam', loss='categorical_crossentropy')
        M.summary()

        print('compiled')

        steps_per_epoch = 1
        mod_epochs = int(np.math.floor(
            self.num_train_samples / self.params['batch_size'] / steps_per_epoch * self.params['epochs']))
        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_grads=True, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(
            self.MODEL_CHECKPOINT_DIR + '/model.{epoch:03d}-{loss:.3f}-{val-loss:.3f}.hdf5',
            monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=True, mode='auto',
            period=mod_epochs / self.params['epochs'])
        callback_steps = 2  # int(np.floor(len(self.val_input_texts) / (self.params['batch_size'] * 2)))
        validation_steps = 2  # int(np.floor(len(self.val_input_texts) / (self.params['batch_size'] * 2)))

        bleuCallback = BleuCallback(self.__serve_batch(self.val_input_texts, self.val_target_texts, "val_callback"),
                                    callback_steps, self.de_word_index, self.START_TOKEN, self.END_TOKEN)
        M.fit_generator(self.__serve_batch(self.train_input_texts, self.train_target_texts, "train"), steps_per_epoch,
                        epochs=mod_epochs, verbose=2, callbacks=[tbCallBack, modelCallback, bleuCallback],
                        validation_data=self.__serve_batch(self.val_input_texts, self.val_target_texts, "val"),
                        validation_steps=validation_steps,
                        max_queue_size=1)
        M.save(self.model_file)

    def __split_data(self, file):
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
            target_text = target_text
            target_texts.append(target_text)

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
        if mode == 'val' or mode == 'val_callback':
            batch_size *= 2
        print(input_texts.shape, target_texts.shape, mode)
        counter = 0
        batch_X = np.zeros((batch_size, self.params['MAX_SEQ_LEN']), dtype='int32')
        batch_Y = np.zeros(
            (batch_size, self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
            dtype='int32')
        while True:
            for i in range(input_texts.shape[0]):
                in_X = input_texts[i]
                out_Y = np.zeros((1, target_texts.shape[1], self.params['MAX_WORDS_DE'] + 3), dtype='int32')
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

    def __setup_model(self):
        try:
            test = self.en_embedding_matrix
            test = self.M
            return
        except AttributeError:
            pass

        self.en_embedding_matrix = np.load(self.BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy')

        M = Sequential()
        M.add(
            Embedding(self.params['MAX_WORDS_EN'] + 3, self.params['EMBEDDING_DIM'], weights=[self.en_embedding_matrix],
                      mask_zero=True, trainable=False))

        M.add(LSTM(self.params['latent_dim'], return_sequences=True, name='encoder'))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(LSTM(self.params['latent_dim'], return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(TimeDistributed(Dense(self.params['MAX_WORDS_DE'] + 3,
                                    input_shape=(None, self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
                                    activation='softmax')))

        print('compiling')

        M.compile(optimizer='Adam', loss='categorical_crossentropy')
        M.summary()

        self.M.load_weights(self.LATEST_MODELCHKPT)

    def predict_one_sentence(self, sentence):
        self.__setup_model()

        self.en_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
        self.de_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy')

        en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_EN'])
        en_tokenizer.word_index = self.en_word_index
        en_tokenizer.num_words = self.params['MAX_WORDS_EN'] + 3

        de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_DE'])
        de_tokenizer.word_index = self.de_word_index
        de_tokenizer.num_words = self.params['MAX_WORDS_DE'] + 3

        print(sentence)
        sentence = en_tokenizer.texts_to_sequences([sentence], search_related_word=True)
        print(sentence)
        sentence = pad_sequences(sentence, maxlen=self.params['MAX_SEQ_LEN'],
                                 padding='post',
                                 truncating='post')
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])
        print(sentence)

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

    def predict_batch(self, sentences):
        self.__setup_model()

        self.en_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
        self.de_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy')

        en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_EN'])
        en_tokenizer.word_index = self.en_word_index
        en_tokenizer.num_words = self.params['MAX_WORDS_EN'] + 3

        de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_DE'])
        de_tokenizer.word_index = self.de_word_index
        de_tokenizer.num_words = self.params['MAX_WORDS_DE'] + 3

        print(sentences)
        sentences = en_tokenizer.texts_to_sequences(sentences)
        print(sentences)
        sentences = pad_sequences(sentences, maxlen=self.params['MAX_SEQ_LEN'],
                                  padding='post',
                                  truncating='post')
        sentences = sentences.reshape(sentences.shape[0], sentences.shape[1])

        batch_size = sentences.shape[0]
        if batch_size > 10:
            batch_size = 10

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
            if to_idx > sentences.shape[0]:
                # todo accept not multiple of batchsize
                break
        return predicted_sentences

    def calculate_hiddenstate_after_encoder(self, sentence):
        self.__setup_model()

        self.en_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
        self.de_word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/de_word_index.npy')

        en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_EN'])
        en_tokenizer.word_index = self.en_word_index
        en_tokenizer.num_words = self.params['MAX_WORDS_EN'] + 3

        de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_DE'])
        de_tokenizer.word_index = self.de_word_index
        de_tokenizer.num_words = self.params['MAX_WORDS_DE'] + 3

        print(sentence)
        sentence = en_tokenizer.texts_to_sequences([sentence])
        print(sentence)
        sentence = pad_sequences(sentence, maxlen=self.params['MAX_SEQ_LEN'],
                                 padding='post',
                                 truncating='post')
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])
        print(sentence)

        encoder_name = 'encoder'

        encoder = Model(inputs=self.M.input, outputs=self.M.get_layer(encoder_name).output)

        prediction = encoder.predict(sentence, batch_size=1)
        print(prediction.shape)
        return prediction

    def calculate_every_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate(self, sentence):
        raise NotImplementedError()

    def calculate_hiddenstate_after_decoder(self, sentence):
        raise NotImplementedError()

    def setup_inference(self):
        raise NotImplementedError()
