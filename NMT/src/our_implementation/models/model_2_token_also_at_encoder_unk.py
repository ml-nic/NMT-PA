import os

import numpy as np
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


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'model_2_token_also_at_encoder_unk'

        self.params['batch_size'] = 128
        self.params['epochs'] = 5
        self.params['latent_dim'] = 256
        self.params['num_samples'] = 150000
        self.params['num_tokens'] = 91
        self.params['max_seq_length'] = 100
        self.params['EMBEDDING_DIM'] = 100
        self.params['MAX_WORDS'] = 20000
        self.params['P_DENSE_DROPOUT'] = 0.2

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENCE_DIR = '../../Persistence/' + self.identifier
        if not os.path.exists(self.BASIC_PERSISTENCE_DIR):
            os.makedirs(self.BASIC_PERSISTENCE_DIR)
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR, 'Graph')
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        # self.input_token_idx_file = os.path.join(self.BASIC_PERSISTENCE_DIR, "input_token_index.npy")
        # self.target_token_idx_file = os.path.join(self.BASIC_PERSISTENCE_DIR, "target_token_index.npy")
        self.data_path = os.path.join(self.BASE_DATA_DIR, 'Training/deu.txt')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.PRETRAINED_GLOVE_FILE = os.path.join(self.BASE_DATA_DIR, 'glove.6B.100d.txt')
        self.LATEST_MODELCHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR, 'model.878-1.90.hdf5')
        # self.TRAIN_EN_FILE = "europarl-v7.de-en.en"
        # self.TRAIN_DE_FILE = "europarl-v7.de-en.de"
        # self.VAL_EN_FILE = "newstest2013.en"
        # self.VAL_DE_FILE = "newstest2013.de"

        # english_train_file = os.path.join(BASE_DATA_DIR, "Training", TRAIN_EN_FILE)
        # german_train_file = os.path.join(BASE_DATA_DIR, "Training", TRAIN_DE_FILE)
        # english_val_file = os.path.join(BASE_DATA_DIR, "Validation", VAL_EN_FILE)
        # german_val_file = os.path.join(BASE_DATA_DIR, "Validation", VAL_DE_FILE)

        self.START_TOKEN = "_GO"
        self.END_TOKEN = "_EOS"
        self.UNK_TOKEN = "_UNK"

    def start_training(self):
        # data_en = self.load(self.english_train_file)
        # data_de = self.load(self.german_train_file)
        # val_data_en = self.load(self.english_val_file)
        # val_data_de = self.load(self.german_val_file)

        # train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size = self.preprocess_data(
        #    data_en, data_de, val_data_en, val_data_en)

        # if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
        #    print("length of input_data and target_data have to be the same")
        #    exit(-1)
        # num_samples = len(train_input_data)

        # print("Number of training data:", num_samples)
        # print("Number of validation data:", len(val_input_data))

        self.START_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
        self.END_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
        self.UNK_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
        np.save(self.BASIC_PERSISTENCE_DIR + '/start_token_vector.npy', self.START_TOKEN_VECTOR)
        np.save(self.BASIC_PERSISTENCE_DIR + '/end_token_vector.npy', self.END_TOKEN_VECTOR)
        np.save(self.BASIC_PERSISTENCE_DIR + '/unk_token_vector.npy', self.UNK_TOKEN_VECTOR)

        self._split_count_data()

        M = Sequential()
        M.add(Embedding(self.params['MAX_WORDS'] + 3, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                        mask_zero=True, trainable=False))

        M.add(LSTM(self.params['latent_dim'], return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(
            LSTM(self.params['latent_dim'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(TimeDistributed(Dense(self.params['MAX_WORDS'] + 3,
                                    input_shape=(None, self.params['num_tokens'], self.params['MAX_WORDS'] + 3),
                                    activation='softmax')))

        print('compiling')

        M.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        M.summary()

        print('compiled')

        steps = 4
        mod_epochs = np.math.floor(self.num_samples / self.params['batch_size'] / steps * self.params['epochs'])
        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(self.MODEL_CHECKPOINT_DIR + '/model.{epoch:02d}-{loss:.2f}.hdf5',
                                                  monitor='loss', verbose=1, save_best_only=False,
                                                  save_weights_only=True, mode='auto',
                                                  period=mod_epochs / self.params['epochs'])

        M.fit_generator(self.serve_batch(), steps, epochs=mod_epochs, verbose=2, max_queue_size=15,
                        callbacks=[tbCallBack, modelCallback])
        M.save(self.model_file)

        print('\n\n Test prediction:')
        print(self.input_texts[0])
        prediction = M.predict(self.input_texts[0])
        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
        predicted_sentence = ''
        for sentence in prediction:
            for token in sentence:
                print(token)
                print(token.shape)
                max_idx = np.argmax(token)
                print(max_idx)
                if max_idx == 0:
                    print("id of max token = 0")
                    predicted_sentence += reverse_word_index[np.argmax(np.delete(token, max_idx))]
                else:
                    print(reverse_word_index[max_idx])
                    predicted_sentence += reverse_word_index[max_idx]
        print(predicted_sentence)
        print("\n\n")
        print(self.input_texts[10000])
        prediction = M.predict(self.input_texts[10000])
        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
        predicted_sentence = ''
        for sentence in prediction:
            for token in sentence:
                print(token)
                print(token.shape)
                max_idx = np.argmax(token)
                print(max_idx)
                if max_idx == 0:
                    print("id of max token = 0")
                    predicted_sentence += reverse_word_index[np.argmax(np.delete(token, max_idx))]
                else:
                    print(reverse_word_index[max_idx])
                    predicted_sentence += reverse_word_index[max_idx]
        print(predicted_sentence)

    def _split_count_data(self):
        self.input_texts = []
        self.target_texts = []
        lines = open(self.data_path, encoding='UTF-8').read().split('\n')
        for line in lines[: min(self.params['num_samples'], len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            self.input_texts.append(input_text)
            target_text = target_text
            self.target_texts.append(target_text)
        self.num_samples = len(self.input_texts)
        tokenizer = Tokenizer(num_words=self.params['MAX_WORDS'])
        tokenizer.fit_on_texts(self.input_texts + self.target_texts)
        self.word_index = tokenizer.word_index
        for word in tokenizer.word_index:
            tokenizer.word_index[word] = tokenizer.word_index[word] + 3
        tokenizer.word_index[self.START_TOKEN] = 1
        tokenizer.word_index[self.END_TOKEN] = 2
        tokenizer.word_index[self.UNK_TOKEN] = 3
        tokenizer.num_words = tokenizer.num_words + 3
        self.word_index = tokenizer.word_index

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        self.input_texts = tokenizer.texts_to_sequences(self.input_texts)
        self.target_texts = tokenizer.texts_to_sequences(self.target_texts)
        for idx in range(len(self.target_texts)):
            self.input_texts[idx] = [self.word_index[self.START_TOKEN]] + self.input_texts[idx] + [
                self.word_index[self.END_TOKEN]]
            self.target_texts[idx] = [self.word_index[self.START_TOKEN]] + self.target_texts[idx] + [
                self.word_index[self.END_TOKEN]]
            if self.target_texts[idx][0] != 1:
                print(idx)
                print(self.target_texts[idx])
                exit(-1)

        self.input_texts = pad_sequences(self.input_texts, maxlen=self.params['max_seq_length'], padding='post')
        self.target_texts = pad_sequences(self.target_texts, maxlen=self.params['max_seq_length'], padding='post')

        embeddings_index = {}
        filename = self.PRETRAINED_GLOVE_FILE
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        self.num_words = self.params['MAX_WORDS'] + 3
        self.embedding_matrix = np.zeros((self.num_words, self.params['EMBEDDING_DIM']))
        for word, i in self.word_index.items():
            if i >= self.params['MAX_WORDS'] + 3 and word not in [self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
                continue
            embedding_vector = None
            if word == self.START_TOKEN:
                embedding_vector = self.START_TOKEN_VECTOR
            elif word == self.END_TOKEN:
                embedding_vector = self.END_TOKEN_VECTOR
            else:
                embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_vector = self.UNK_TOKEN_VECTOR
            self.embedding_matrix[i] = embedding_vector
        np.save(self.BASIC_PERSISTENCE_DIR + '/word_index.npy', self.word_index)
        np.save(self.BASIC_PERSISTENCE_DIR + '/embedding_matrix.npy', self.embedding_matrix)

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

    def convert_last_dim_to_one_hot_enc(self, target, vocab_size):
        """
        :param target: shape: (number of samples, max sentence length)
        :param vocab_size: size of the vocabulary
        :return: transformed target with shape: (number of samples, max sentence length, number of words in vocab)
        """
        x = np.ones((target.shape[0], target.shape[1], vocab_size), dtype='int32')
        for idx, s in enumerate(target):
            for token in s:
                x[idx, :len(target)] = to_categorical(token, num_classes=vocab_size)
        return x

    def serve_batch(self):
        counter = 0
        self.batch_X = np.zeros((self.params['batch_size'], self.params['max_seq_length']), dtype='int32')
        self.batch_Y = np.zeros(
            (self.params['batch_size'], self.params['max_seq_length'], self.params['MAX_WORDS'] + 3),
            dtype='int32')
        while True:
            for i in range(self.input_texts.shape[0]):
                in_X = self.input_texts[i]
                out_Y = np.zeros((1, self.target_texts.shape[1], self.params['MAX_WORDS'] + 3), dtype='int32')
                token_counter = 0
                for token in self.target_texts[i]:
                    out_Y[0, token_counter, :] = to_categorical(token, num_classes=self.params['MAX_WORDS'] + 3)
                    token_counter += 1
                self.batch_X[counter] = in_X
                self.batch_Y[counter] = out_Y
                counter += 1
                if counter == self.params['batch_size']:
                    print("counter == batch_size", i)
                    counter = 0
                    yield self.batch_X, self.batch_Y

    def preprocess_data(self, train_input_data, train_target_data, val_input_data, val_target_data):
        train_input_data, train_target_data, val_input_data, val_target_data, word_index = self.tokenize(
            train_input_data,
            train_target_data,
            val_input_data,
            val_target_data)

        train_input_data = pad_sequences(train_input_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        train_target_data = pad_sequences(train_target_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        val_input_data = pad_sequences(val_input_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        val_target_data = pad_sequences(val_target_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')

        embeddings_index = self.load_embedding()
        embedding_matrix, num_words = self.prepare_embedding_matrix(word_index, embeddings_index)

        # target_data = convert_last_dim_to_one_hot_enc(padded_target_data, num_words)

        return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words

    def tokenize(self, train_input_data, train_target_data, val_input_data, val_target_data):
        tokenizer = Tokenizer(num_words=self.params['MAX_NUM_WORDS'])
        tokenizer.fit_on_texts(train_input_data + train_target_data + val_input_data + val_target_data)

        train_input_data = tokenizer.texts_to_sequences(train_input_data)
        train_target_data = tokenizer.texts_to_sequences(train_target_data)
        val_input_data = tokenizer.texts_to_sequences(val_input_data)
        val_target_data = tokenizer.texts_to_sequences(val_target_data)

        return train_input_data, train_target_data, val_input_data, val_target_data, tokenizer.word_index

    def load_embedding(self):
        print('Indexing word vectors.')

        embeddings_index = {}
        filename = os.path.join(self.BASE_DATA_DIR, 'glove.6B.100d.txt')
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        return embeddings_index

    def prepare_embedding_matrix(self, word_index, embeddings_index):
        print('Preparing embedding matrix.')

        # prepare embedding matrix
        num_words = min(self.params['MAX_NUM_WORDS'], len(word_index)) + 1
        embedding_matrix = np.zeros((num_words, self.params['EMBEDDING_DIM']))
        for word, i in word_index.items():
            if i >= self.params['MAX_NUM_WORDS']:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix, num_words

    def __setup_model(self):
        try:
            test = self.embedding_matrix
            test = self.M
            return
        except AttributeError:
            pass

        self.embedding_matrix = np.load(self.BASIC_PERSISTENCE_DIR + '/embedding_matrix.npy')

        self.M = Sequential()
        self.M.add(Embedding(self.params['MAX_WORDS'] + 3, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                             mask_zero=True, trainable=False))

        self.M.add(LSTM(self.params['latent_dim'], return_sequences=True, name='encoder'))

        self.M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        self.M.add(
            LSTM(self.params['latent_dim'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))

        self.M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        self.M.add(TimeDistributed(Dense(self.params['MAX_WORDS'] + 3,
                                    input_shape=(None, self.params['num_tokens'], self.params['MAX_WORDS'] + 3),
                                    activation='softmax')))

        self.M.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.M.summary()

        self.M.load_weights(self.LATEST_MODELCHKPT)

    def predict_one_sentence(self, sentence):
        self.__setup_model()
        tokenizer = Tokenizer()
        self.word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/word_index.npy')
        self.word_index = self.word_index.item()
        tokenizer.word_index = self.word_index
        self.num_words = self.params['MAX_WORDS'] + 3
        tokenizer.num_words = self.num_words

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()
        print(sentence)
        sentence = tokenizer.texts_to_sequences([sentence])
        print(sentence)
        sentence = [self.word_index[self.START_TOKEN]] + sentence[0] + [self.word_index[self.END_TOKEN]]
        print(sentence)
        sentence = pad_sequences([sentence], maxlen=self.params['max_seq_length'], padding='post')
        print(sentence)
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])
        print(sentence)
        prediction = self.M.predict(sentence)

        predicted_sentence = ""
        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
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

        tokenizer = Tokenizer()
        self.word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/word_index.npy')
        self.word_index = self.word_index.item()
        tokenizer.word_index = self.word_index
        self.num_words = self.params['MAX_WORDS'] + 3
        tokenizer.num_words = self.num_words

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        sentences = tokenizer.texts_to_sequences(sentences)
        mod_sentences = []
        for sentence in sentences:
            mod_sentences.append([self.word_index[self.START_TOKEN]] + sentence + [self.word_index[self.END_TOKEN]])
        sentences = pad_sequences(mod_sentences, maxlen=self.params['max_seq_length'], padding='post')
        sentences = sentences.reshape(sentences.shape[0], sentences.shape[1])

        batch_size = sentences.shape[0]
        if batch_size > 10:
            batch_size = 10

        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
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
                # todo nicht multiple von batchsize trotzdem predicten
                break
        return predicted_sentences

    def calculate_hiddenstate_after_encoder(self, sentence):
        self.__setup_model()

        tokenizer = Tokenizer()
        self.word_index = np.load(self.BASIC_PERSISTENCE_DIR + '/word_index.npy')
        self.word_index = self.word_index.item()
        tokenizer.word_index = self.word_index
        self.num_words = self.params['MAX_WORDS'] + 3
        tokenizer.num_words = self.num_words

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = [self.word_index[self.START_TOKEN]] + sentence[0] + [self.word_index[self.END_TOKEN]]
        sentence = pad_sequences([sentence], maxlen=self.params['max_seq_length'], padding='post')
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])

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
