from __future__ import print_function

import os

import numpy as np
from keras import callbacks
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.models import load_model
from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'CharSeq2SeqTutOneHotInput'

        self.params['batch_size'] = 16
        self.params['epochs'] = 20
        self.params['latent_dim'] = 256
        self.params['num_encoder_tokens'] = 89
        self.params['num_decoder_tokens'] = 120
        self.params['max_encoder_seq_length'] = 350
        self.params['max_decoder_seq_length'] = 350

        self.BASE_DATA_DIR = "../../DataSets/"
        self.BASIC_PERSISTENCE_DIR = '../../Persistence/' + self.identifier
        if not os.path.exists(self.BASIC_PERSISTENCE_DIR):
            os.makedirs(self.BASIC_PERSISTENCE_DIR)
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.input_token_idx_file = os.path.join(self.BASIC_PERSISTENCE_DIR, "input_token_index.npy")
        self.target_token_idx_file = os.path.join(self.BASIC_PERSISTENCE_DIR, "target_token_index.npy")
        self.data_path = self.BASE_DATA_DIR + 'Training/DE_EN_(tatoeba)_train.txt'
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')

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

    def start_training(self):
        input_texts, target_texts, input_token_index, target_token_index = self.split_count_data()

        encoder_input_data = np.zeros(
            (len(input_texts), self.params['max_encoder_seq_length'], self.params['num_encoder_tokens']),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), self.params['max_decoder_seq_length'], self.params['num_decoder_tokens']),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), self.params['max_decoder_seq_length'], self.params['num_decoder_tokens']),
            dtype='int16')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_target_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['num_encoder_tokens']))
        encoder = LSTM(self.params['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['num_decoder_tokens']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['num_decoder_tokens'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.summary()

        tbCallBack = callbacks.TensorBoard(log_dir=self.BASIC_PERSISTENCE_DIR + '/Graph', histogram_freq=0,
                                           write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(
            self.MODEL_DIR + '/model.{epoch:03d}-{loss:.3f}.hdf5', monitor='loss', verbose=1,
            save_best_only=False, save_weights_only=True, mode='auto', period=1)

        model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=self.params['batch_size'],
                  epochs=self.params['epochs'], verbose=1, callbacks=[tbCallBack, modelCallback])
        #
        # Save model
        model.save(self.model_file)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling our_implementation.models
        encoder_model = Model(encoder_inputs, encoder_states)
        encoder_model.save(self.encoder_model_file)

        decoder_state_input_h = Input(shape=(self.params['latent_dim'],))
        decoder_state_input_c = Input(shape=(self.params['latent_dim'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        decoder_model.save(self.decoder_model_file)

    def split_count_data(self):
        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        lines = open(self.data_path, encoding='UTF-8').read().split('\n')
        for line in lines:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
        np.save(self.input_token_idx_file, input_token_index)
        np.save(self.target_token_idx_file, target_token_index)

        return input_texts, target_texts, input_token_index, target_token_index

    def setup_inferenceddf(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['num_encoder_tokens']))
        encoder = LSTM(self.params['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['num_decoder_tokens']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['num_decoder_tokens'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # self.model = load_model('./data/s2s2.h5')
        self.model.load_weights('weights_file')

        self.encoder_model = Model(encoder_inputs, encoder_states)
        # self.encoder_model = load_model('./data/encoder_model.h5')

        decoder_state_input_h = Input(shape=(self.params['latent_dim'],))
        decoder_state_input_c = Input(shape=(self.params['latent_dim'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # self.decoder_model = load_model('./data/decoder_model.h5')

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.input_token_index = np.load(self.input_token_idx_file)
        self.target_token_index = np.load(self.target_token_idx_file)
        self.input_token_index = self.input_token_index.item()
        self.target_token_index = self.target_token_index.item()

        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

    def _setup_inference(self):
        self.model = load_model(self.model_file)
        self.encoder_model = load_model(self.encoder_model_file)
        self.decoder_model = load_model(self.decoder_model_file)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.input_token_index = np.load(self.input_token_idx_file)
        self.target_token_index = np.load(self.target_token_idx_file)
        self.input_token_index = self.input_token_index.item()
        self.target_token_index = self.target_token_index.item()

        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

    def predict_one_sentence(self, sentence):
        self._setup_inference()
        # input_seq = np.zeros((1, 71, 91))
        input_seq = np.zeros((1, self.params['max_encoder_seq_length'], self.params['num_encoder_tokens']))

        index = 0
        for char in sentence:
            input_seq[0][index][self.input_token_index[char]] = 1.
            index += 1

        decoded_sentence = self._decode_sequence(input_seq, False)
        print(decoded_sentence)
        return decoded_sentence

    def __decode_for_specific_weight(self, input_sequences):
        decoded_sentences = []
        for i in range(input_sequences.shape[0]):
            if i % int(input_sequences.shape[0] / 100) == 0:
                print(i, "of", input_sequences.shape[0])
            current_sentence = np.array([input_sequences[i]])
            # Encode the input as state vectors.
            states_value = self.encoder_model.predict(current_sentence)

            # Generate empty target sequence of length 1.
            target_seqs = np.zeros((1, 1, self.params['num_decoder_tokens']))
            # Populate the first character of target sequence with the start character.
            target_seqs[0, 0, self.target_token_index['\t']] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = self.decoder_model.predict([target_seqs] + states_value)

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = self.reverse_target_char_index[sampled_token_index]
                if sampled_char != '\n':
                    decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '\n' or len(decoded_sentence) > self.params['max_decoder_seq_length']):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seqs = np.zeros((1, 1, self.params['num_decoder_tokens']))
                target_seqs[0, 0, sampled_token_index] = 1.

                # Update states
                states_value = [h, c]
            decoded_sentences.append(decoded_sentence)
        return decoded_sentences

    def _decode_sequence(self, input_sequences, all_weights):
        predictions_for_weights = {}
        if all_weights is True:
            for weight_file in self.WEIGHT_FILES:
                self.model.load_weights(weight_file)
                predictions_for_weights[weight_file.split('model.')[1]] = self.__decode_for_specific_weight(
                    input_sequences)
        else:
            predictions_for_weights[
                self.LATEST_MODELCHKPT.split('model.')[1]] = self.__decode_for_specific_weight(input_sequences)

        return predictions_for_weights

    def predict_batch(self, sentences, all_weights=False):
        self._setup_inference()

        input_seqs = np.zeros(
            (len(sentences), self.params['max_encoder_seq_length'], self.params['num_encoder_tokens']))
        sentence_idx = 0
        for sentence in sentences:
            index = 0
            for char in sentence:
                try:
                    input_seqs[sentence_idx][index][self.input_token_index[char]] = 1.
                except KeyError as e:
                    print("Attention:", char, "is unknown!!!")
                index += 1
            sentence_idx += 1

        decoded_sentences = self._decode_sequence(input_seqs, all_weights)
        return decoded_sentences

    def calculate_hiddenstate_after_encoder(self, sentences):
        self._setup_inference()

        input_seqs = np.zeros(
            (len(sentences), self.params['max_encoder_seq_length'], self.params['num_encoder_tokens']))
        sentence_idx = 0
        for sentence in sentences:
            index = 0
            for char in sentence:
                try:
                    input_seqs[sentence_idx][index][self.input_token_index[char]] = 1.
                except KeyError as e:
                    print("Attention:", char, "is unknown!!!")
                index += 1
            sentence_idx += 1

        batch_size = input_seqs.shape[0]
        if batch_size > 20:
            batch_size = 20

        from_idx = 0
        to_idx = batch_size

        hiddenstates = []
        while True:
            print("from_idx, to_idx, hm_sentences", from_idx, to_idx, input_seqs.shape[0])
            current_batch = input_seqs[from_idx:to_idx]
            prediction = self.encoder_model.predict(current_batch, batch_size=batch_size)
            hiddenstates.append(prediction[0])

            from_idx += batch_size
            to_idx += batch_size

            if from_idx >= input_seqs.shape[0]:
                break
            elif to_idx > input_seqs.shape[0] and from_idx < input_seqs.shape[0]:
                to_idx = input_seqs.shape[0] + 1
            elif to_idx > input_seqs.shape[0]:
                break
        return hiddenstates

    def calculate_every_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate(self, sentence):
        raise NotImplementedError()

    def calculate_hiddenstate_after_decoder(self, sentence):
        raise NotImplementedError()
