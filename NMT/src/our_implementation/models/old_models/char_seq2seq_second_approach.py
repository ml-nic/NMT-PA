from __future__ import print_function

import gc
import os

import numpy as np
from keras import callbacks
from keras.engine import Model
from keras.layers import Dense, Input, LSTM
from keras.models import load_model
from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'char_seq2seq_second_approach'

        self.params['BATCH_SIZE'] = 128
        self.params['EMBEDDING_DIM'] = 100
        self.params['EPOCHS'] = 15
        self.params['LATENT_DIM'] = 256
        self.params['NUM_TOKENS'] = 70
        self.params['MAX_NUM_SAMPLES'] = 1000000
        self.params['MAX_NUM_WORDS'] = 20000
        self.params['MAX_SENTENCES'] = 1000
        self.params['MAX_SEQ_LEN'] = 1800
        self.UNKNOWN_CHAR = '\r'
        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENCE_DIR = '../../persistent/persistentModelseq2seqbugfree'
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR, 'Graph')
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.LATEST_MODEL_CHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR,
                                               'chkp2prepro_64_100_15_256_1000000_20000_1000_1800_70_70_0.8_char___tfmodelprepro.35999-54.06.hdf5')
        self.token_idx_file = os.path.join(self.BASIC_PERSISTENCE_DIR, "input_token_idx_preprocessed.npy")
        self.train_en_file = os.path.join(self.BASE_DATA_DIR, 'Training/train.en')
        self.train_de_file = os.path.join(self.BASE_DATA_DIR, 'Training/train.de')
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')

    def start_training(self):
        train_input_texts, token_index, train_target_texts = self._split_data_and_count()
        gc.collect()
        np.save(self.token_idx_file, token_index)

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        encoder = LSTM(self.params['LATENT_DIM'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['LATENT_DIM'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['NUM_TOKENS'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        steps = 5
        mod_epochs = np.floor(
            len(train_input_texts) / self.params['BATCH_SIZE'] / steps * self.params['EPOCHS'])
        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(self.MODEL_CHECKPOINT_DIR + '/model.{epoch:02d}-{loss:.2f}.hdf5',
                                                  monitor='loss', verbose=1, save_best_only=False,
                                                  save_weights_only=False, mode='auto', period=mod_epochs/self.params['epochs'])

        print('steps', steps, 'mod_epochs', mod_epochs, 'len(train_input_texts)', len(train_input_texts), "batch_size",
              self.params['BATCH_SIZE'], 'epochs', self.params['EPOCHS'])
        model.fit_generator(self.serve_batch(train_input_texts, train_target_texts, token_index),
                            steps, epochs=mod_epochs, verbose=2, max_queue_size=5,
                            # validation_data=serve_batch(val_input_texts, val_target_texts, train_input_token_idx,
                            #                            train_target_token_idx),
                            # validation_steps=len(val_input_texts) / self.params['BATCH_SIZE'],
                            callbacks=[tbCallBack, modelCallback]
                            )

        # Save model
        model.save(self.model_file)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states
        # model = load_model('s2s.h5')

        # Define sampling our_implementation.models
        encoder_model = Model(encoder_inputs, encoder_states)
        encoder_model.save(self.encoder_model_file)

        decoder_state_input_h = Input(shape=(self.params['LATENT_DIM'],))
        decoder_state_input_c = Input(shape=(self.params['LATENT_DIM'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        decoder_model.save(self.decoder_model_file)

    def setup_inferencejlk(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        encoder = LSTM(self.params['LATENT_DIM'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['LATENT_DIM'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['NUM_TOKENS'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.load_weights(self.LATEST_MODEL_CHKPT)

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.params['LATENT_DIM'],))
        decoder_state_input_c = Input(shape=(self.params['LATENT_DIM'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        self.char_index = np.load(self.token_idx_file)
        self.char_index = self.char_index.item()
        self.reverse_char_index = dict((i, char) for char, i in self.char_index.items())

    def _setup_inference(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        encoder = LSTM(self.params['LATENT_DIM'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['LATENT_DIM'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['NUM_TOKENS'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model.load_weights(self.LATEST_MODEL_CHKPT)

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.params['LATENT_DIM'],))
        decoder_state_input_c = Input(shape=(self.params['LATENT_DIM'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        self.char_index = np.load(self.token_idx_file)
        self.char_index = self.char_index.item()
        self.reverse_char_index = dict((i, char) for char, i in self.char_index.items())

    def setup_inferenceold(self):
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states


        # input_texts, token_index, target_texts = self.split_data_and_count()

        # Define sampling our_implementation.models
        model = load_model(self.model_file)
        model.load_weights(self.LATEST_MODEL_CHKPT)
        self.encoder_model = load_model(self.encoder_model_file)
        self.decoder_model = load_model(self.decoder_model_file)

        # Reverse-lookup token index to decode sequences back to
        # something readable.


        self.char_index = np.load(self.token_idx_file)
        self.char_index = self.char_index.item()
        self.reverse_char_index = dict((i, char) for char, i in self.char_index.items())

    def predict_one_sentence(self, sentence):
        self._setup_inference()
        input_seq = np.zeros((1, self.params['MAX_SEQ_LEN'], self.params['NUM_TOKENS']))

        index = 0
        for char in sentence:
            try:
                input_seq[0][index][self.char_index[char]] = 1.
            except KeyError:
                input_seq[0][index][self.char_index['\r']] = 1.
            index += 1

        decoded_sentence = self.decode_sequence(input_seq, self.char_index, self.reverse_char_index)
        return decoded_sentence

    def _split_data_and_count(self):
        input_texts = []
        target_texts = []
        characters_dict = {}
        lines_en = open(self.train_en_file, encoding='UTF-8').read().split('\n')
        lines_de = open(self.train_de_file, encoding='UTF-8').read().split('\n')
        if len(lines_de) != len(lines_en):
            print("error length of both training files have to be the same")
            exit()
        for line_idx in range(len(lines_en) - 1):
            input_text = lines_en[line_idx]
            target_text = lines_de[line_idx]
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            if len(input_text) == 0 or len(target_text) == 0:
                continue
            if len(input_text) > self.params['MAX_SEQ_LEN']:
                input_text = input_text[0:self.params['MAX_SEQ_LEN']]
            if len(target_text) > self.params['MAX_SEQ_LEN']:
                target_text = target_text[0:self.params['MAX_SEQ_LEN']]
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                char = char.lower()
                try:
                    char.encode('ascii')
                    try:
                        characters_dict[char] = characters_dict[char] + 1
                    except KeyError:
                        characters_dict[char] = 1
                except Exception:
                    pass
            for char in target_text:
                char = char.lower()
                try:
                    char.encode('ascii')
                    try:
                        characters_dict[char] = characters_dict[char] + 1
                    except KeyError:
                        characters_dict[char] = 1
                except Exception:
                    pass
        lines = None

        characters = []
        sorted_char_dict = [(k, characters_dict[k]) for k in
                            sorted(characters_dict, key=characters_dict.get, reverse=True)]
        print(sorted_char_dict)
        counter = 0
        for tuple in sorted_char_dict:
            characters.append(tuple[0])
            counter += 1
            if counter == self.params['NUM_TOKENS']:
                break
        print(len(characters))
        if '\t' not in characters:
            characters.append('\t')
        if '\n' not in characters:
            characters.append('\n')
        if self.UNKNOWN_CHAR not in characters:
            characters.append(self.UNKNOWN_CHAR)
        for char in 'abcdefghijklmnopqrstuvwxyz0123456789.,äöü':
            if char not in characters:
                characters.append(char)
        characters = sorted(characters)
        print(len(characters))
        print('Number of samples:', len(input_texts))
        print('Number of unique tokens:', self.params['NUM_TOKENS'])
        print('Max sequence length for inputs:', self.params['MAX_SEQ_LEN'])

        token_index = dict([(char, i) for i, char in enumerate(characters)])
        return input_texts, token_index, target_texts

    def serve_batch(self, input_data, target_data, token_index):
        gc.collect()
        encoder_input_data = np.zeros((self.params['BATCH_SIZE'], self.params['MAX_SEQ_LEN'],
                                       self.params['NUM_TOKENS']), dtype='float32')
        decoder_input_data = np.zeros((self.params['BATCH_SIZE'], self.params['MAX_SEQ_LEN'],
                                       self.params['NUM_TOKENS']), dtype='float32')
        decoder_target_data = np.zeros((self.params['BATCH_SIZE'], self.params['MAX_SEQ_LEN'],
                                        self.params['NUM_TOKENS']), dtype='float32')
        start = 0
        print("serve_batch", "start", start)
        while True:
            for i, (input_text, target_text) in enumerate(
                    zip(input_data[start:start + self.params['BATCH_SIZE']],
                        target_data[start:start + self.params['BATCH_SIZE']])):
                for t, char in enumerate(input_text):
                    char = char.lower()
                    try:
                        encoder_input_data[i, t, token_index[char]] = 1.
                    except KeyError:
                        encoder_input_data[i, t, token_index[self.UNKNOWN_CHAR]] = 1.
                for t, char in enumerate(target_text):
                    char = char.lower()
                    # decoder_target_data is ahead of decoder_target_data by one timestep
                    try:
                        decoder_input_data[i, t, token_index[char]] = 1.
                    except KeyError:
                        decoder_input_data[i, t, token_index[self.UNKNOWN_CHAR]] = 1.
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        try:
                            decoder_target_data[i, t - 1, token_index[char]] = 1.
                        except KeyError:
                            decoder_target_data[i, t - 1, token_index[self.UNKNOWN_CHAR]] = 1.
            start += self.params['BATCH_SIZE']
            print("serve_batch", start, start / self.params['BATCH_SIZE'])
            yield [encoder_input_data, decoder_input_data], decoder_target_data

    def decode_sequence(self, input_sequence, char_index, reverse_char_index):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.params['NUM_TOKENS']))
        # Populate the first character of target sequence with the start character.

        target_seq[0, 0, char_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_char_index[sampled_token_index]

            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.params['MAX_SEQ_LEN']):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.params['NUM_TOKENS']))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def calculate_hiddenstate_after_decoder(self, sentence):
        input_seq = np.zeros((1, self.params['MAX_SEQ_LEN'], self.params['NUM_TOKENS']))

        index = 0
        for char in sentence:
            try:
                input_seq[0][index][self.char_index[char]] = 1.
            except KeyError:
                input_seq[0][index][self.char_index['\r']] = 1.
            index += 1

        states_value = self.encoder_model.predict(input_seq)
        return states_value

    def predict_batch(self, sentences):
        raise NotImplementedError()

    def calculate_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate(self, sentence):
        raise NotImplementedError()
