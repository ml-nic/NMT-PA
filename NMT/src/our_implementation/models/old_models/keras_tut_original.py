from __future__ import print_function

import os

from models.BaseModel import BaseModel


class KerasTutSeq2Seq(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.params['BATCH_SIZE'] = 256
        self.params['EPOCHS'] = 100
        self.params['LATENT_DIM'] = 256
        self.params['MAX_NUM_SAMPLES'] = 3000000
        self.params['NUM_TOKENS'] = 127
        self.params['MAX_ENCODER_SEQ_LEN'] = 286
        self.params['MAX_DECODER_SEQ_LEN'] = 382

        self.UNKNOWN_CHAR = '\r'
        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENCE_DIR = '../../persistent/model_keras_tut_original'
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENCE_DIR)
        self.LATEST_MODEL_CHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR,
                                               'chkp22_64_100_15_256_1000000_20000_1000_1800_150_150_0.8_char___tfmodel2.6999-47.37.hdf5')
        self.token_idx_file = os.path.join(self.BASIC_PERSISTENCE_DIR, "token_index.npy")
        self.train_data_file = os.path.join(self.BASE_DATA_DIR, 'Training/deu.txt')
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 's2s2.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')
