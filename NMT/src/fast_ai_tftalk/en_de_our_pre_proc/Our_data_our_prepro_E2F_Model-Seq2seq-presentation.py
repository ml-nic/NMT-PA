# coding: utf-8

# In[46]:

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback
from recurrentshop import *
# import sutils; importlib.reload(sutils)
from sutils import *

# In[47]:


print(keras.__version__)
print(tf.__version__)

# In[48]:


# In[50]:


path = '/data/TensorFlowTalks/neural_translation_my_pre_proc/'
dpath = '/data/TensorFlowTalks/neural_translation_my_pre_proc/translate/'

# ### Set up Regex and tokenize for use later

# In[51]:


# re_mult_space = re.compile(r"  *")
# re_mw_punc = re.compile(r"(\w[’'])(\w)")
# re_punc = re.compile("([\"().,;:/_?!—])")
# re_apos = re.compile(r"(\w)'s\b")


# def simple_toks(sent):
#    sent = re_apos.sub(r"\1 's", sent)
#    sent = re_mw_punc.sub(r"\1 \2", sent)
#    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
#    sent = re_mult_space.sub(' ', sent)
#    return sent.lower().split()


# ## Load the PreProcessed data
# 
# Here we load all the data 

# In[52]:


# path = '/data/TensorFlowTalks/neural_translation_en_de/'
# dpath = '/data/TensorFlowTalks/neural_translation_en_de/translate/'

# data = load(dpath+'nmt_data.pkl')
# look_ups = load(dpath+'look_ups.pkl')

# fr_train = data['fr_train']
# fr_test = data['fr_test']
# en_train = data['en_train']
# en_test = data['en_test']
# en_w2id = look_ups['en_w2id']
# fr_vocab = look_ups['fr_vocab']
# en_vocab = look_ups['en_vocab']
# en_embs = look_ups['en_embs']
# fr_embs = look_ups['fr_embs']

# questions = load(dpath+'questions.pkl')
# print(questions[10])
# en_qs, fr_qs = zip(*questions)


# In[53]:


# print(fr_train[0:2])
# print(en_train[0:2])
# print(en_vocab[0:10])
# print(en_qs[0], fr_qs[0])
# print(en_w2id)


# In[55]:


path = '/data/TensorFlowTalks/neural_translation_my_pre_proc/'
dpath = '/data/TensorFlowTalks/neural_translation_my_pre_proc/translate/'

BASIC_PERSISTENCE_DIR = "/data/TensorFlowTalks/neural_translation_my_pre_proc/translate/"

en_train = np.load(BASIC_PERSISTENCE_DIR + '/train_target_texts.npy')
fr_train = np.load(BASIC_PERSISTENCE_DIR + '/train_input_texts.npy')
en_val = np.load(BASIC_PERSISTENCE_DIR + '/val_target_texts.npy')
fr_val = np.load(BASIC_PERSISTENCE_DIR + '/val_input_texts.npy')
en_w2id = np.load(BASIC_PERSISTENCE_DIR + '/en_word_index.npy')
fr_w2id = np.load(BASIC_PERSISTENCE_DIR + '/de_word_index.npy')
en_embs = np.load(BASIC_PERSISTENCE_DIR + '/en_embedding_matrix.npy')

# In[56]:


print(fr_train[0:2])
print(en_train[0:2])
# print(en_vocab[0:10])
# print(en_qs[0], fr_qs[0])


# In[57]:


# for running model test on small set of data
# fr_train = fr_train[:5000]
# en_train = fr_train[:5000]

fr_train.shape

# In[58]:


en_train.shape

# ## Model

# #### Create some Keras Callbacks to handle early stopping and Learning Rate changes

# In[59]:


reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0, cooldown=1, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')

import math


# learning rate schedule for dropping every 10 epochs
def LRDropping(epoch):
    initial_lrate = 0.001
    drop = 0.9
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# try at manual setting of LR for Epochs
def fixed_dropping(epoch):
    if epoch < 2:
        lrate = 0.01
    elif epoch < 4:
        lrate = 0.001
    elif epoch < 7:
        lrate = 0.0005
    else:
        lrate = 0.0001
    print(lrate)
    return lrate


LRDrop = LearningRateScheduler(fixed_dropping)

# In[60]:


modelCallback = ModelCheckpoint(
    'our_data_our_preproc_e2f_last.{epoch:03d}-{loss:.3f}.hdf5',
    monitor='val_loss', verbose=1, save_best_only=False,
    save_weights_only=True, mode='auto',
    period=1)

# creating different sets of Params to easily import into the model at train time
params = {'verbose': 1, 'callbacks': [TQDMNotebookCallback(), reduce_LR, early_stopping, modelCallback]}
params2 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}
params3 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}

# #### Set some parameters for the model

# In[61]:


lr = 1e-3
maxlen = 30
dim_en_vec = 200
n_en_vec = 400000
dim_fr_vec = 200

vocab_size = 40000  # len(fr_vocab) #the output vocab # embeddings.shape[0]
embedding_size = 100  # The english inputs embeddings embeddings.shape[1]

# In[62]:


# fr_wgts = [fr_embs.T, np.zeros((len(fr_vocab,)))]


# ### The model itself

# In[70]:


# Test different settings:

# - only two LSTM's and one TimeDistributed
# - LSTM instead of Bidirectional
# - only one timeDistributed
# - without weights for german embeddings
# - categorical_crossentropy instead of sparse



# while training implement my preprocessing into jupyter notebook and try my prepro with this standard model


# In[71]:


# without fr_wgts
inp = Input((100,))
x = Embedding(40003, dim_en_vec, input_length=100,
              weights=[en_embs], trainable=False)(inp)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LSTM(128, return_sequences=True)(x)
x = TimeDistributed(Dense(dim_fr_vec))(x)
x = TimeDistributed(Dense(40003))(x)
x = Activation('softmax')(x)

model = Model(inp, x)
model.compile('adam', 'sparse_categorical_crossentropy')

K.set_value(model.optimizer.lr, lr)

# In[73]:


model.summary()

# In[ ]:


hist = model.fit(en_train, np.expand_dims(fr_train, axis=-1), batch_size=64, epochs=20, **params,
                 validation_data=[en_test, np.expand_dims(fr_test, axis=-1)])

weight_identifier = "our_data_our_preproc_e2f_last"
model.save_weights(dpath + weight_identifier + '.h5')

# In[74]:


weight_identifier = "our_data_our_preproc_e2f_last"
model.load_weights(dpath + weight_identifier + '.h5')

# In[75]:


en_test[0]

# In[76]:


print(en_val[0])
print(fr_val[0])


# ## Testing

# In[77]:


def sent2ids(sent):
    sent = simple_toks(sent)
    ids = [en_w2id[t] for t in sent]
    return pad_sequences([ids], maxlen, padding="post", truncating="post")


# In[87]:


def en2fr(sent):
    # ids = sent2ids(sent)
    ids = sent
    tr_ids = np.argmax(model.predict([ids]), axis=-1)
    return ' '.join(fr_vocab[i] for i in tr_ids[0] if i > 0)


# In[79]:


en2fr("what is the size of canada?")

# In[ ]:


en2fr("what is the size of australia?")

# In[ ]:



print(fr_qs[0])
en2fr("What is light?")

# In[ ]:


print(fr_qs[50000])
en2fr("Why is the Arctic ozone layer thicker than the Antarctic ozone layer?")

# In[ ]:


print(qs[9])
en2fr("Which province is the most populated?")

# In[ ]:


en2fr("Who are we?")

# In[ ]:


print(fr_qs[3])
en2fr("What would we do without it?")

# In[80]:


print(en_val[0])

# ## Predict validation data

# In[89]:


import os

translated_sentences = []
for sent in en_val:
    translated_sentences.append(en2fr(sent))
print(len(translated_sentences))

# In[ ]:


out_file = os.path.join(os.path.abspath(os.path.join(source_file, os.pardir)), weight_identifier + ".pred")
with(open(out_file, 'w')) as file:
    for sent in translated_sentences:
        file.write(sent + '\n')
