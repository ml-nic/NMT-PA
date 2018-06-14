# coding: utf-8

# # English to French using Neural Machine Translation
# 
# This example was taken from the wonderful Cutting Edge Deep Learning for Coders course as taught by Jeremy Howard http://course.fast.ai/part2.html The course is now live and I encourage you to check it out.

# In[3]:


import re

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback
from recurrentshop import *
from seq2seq.models import AttentionSeq2Seq
# import sutils; importlib.reload(sutils)
from sutils import *

# In[4]:


print(keras.__version__)
print(tf.__version__)

# In[5]:


# In[6]:


path = '../neural_translation_en_de_attention/'
dpath = '../neural_translation_en_de_attention/translate/'

# ### Set up Rex and tokenize for use later

# In[7]:


re_mult_space = re.compile(r"  *")
re_mw_punc = re.compile(r"(\w[’'])(\w)")
re_punc = re.compile("([\"().,;:/_?!—])")
re_apos = re.compile(r"(\w)'s\b")


def simple_toks(sent):
    sent = re_apos.sub(r"\1 's", sent)
    sent = re_mw_punc.sub(r"\1 \2", sent)
    sent = re_punc.sub(r" \1 ", sent).replace('-', ' ')
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()


# ## Load the PreProcessed data
# 
# Here we load all the data 

# In[8]:


data = load(dpath + 'nmt_datawmtsmall_sos_eos_unk_att.pkl')
look_ups = load(dpath + 'look_upswmtsmall_sos_eos_unk_att.pkl')
fr_train = data['fr_train']
fr_test = data['fr_test']
en_train = data['en_train']
en_test = data['en_test']
en_w2id = look_ups['en_w2id']
fr_vocab = look_ups['fr_vocab']
en_vocab = look_ups['en_vocab']
en_embs = look_ups['en_embs']
fr_embs = look_ups['fr_embs']

questions = load(dpath + 'questionswmt.pkl')
# print(questions[10])
en_qs, fr_qs = zip(*questions)

# In[9]:


# for running model test on small set of data
# fr_train = fr_train[:5000]
# en_train = fr_train[:5000]

fr_train.shape

# In[10]:


en_train.shape

# ## Model

# #### Create some Keras Callbacks to handle early stopping and Learning Rate changes

# In[11]:


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

# In[20]:


modelCallback = ModelCheckpoint(
    'model_checkpoint_advsmall_sos_eos_unk_att.{epoch:03d}-{loss:.3f}.hdf5',
    monitor='val_loss', verbose=1, save_best_only=False,
    save_weights_only=True, mode='auto',
    period=1)

# creating different sets of Params to easily import into the model at train time
params = {'verbose': 1, 'callbacks': [TQDMNotebookCallback(), reduce_LR, early_stopping, modelCallback]}
params2 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}
params3 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}

# #### Set some parameters for the model

# In[13]:


lr = 1e-3
maxlen = 100
dim_en_vec = 200
n_en_vec = 400000
dim_fr_vec = 200

vocab_size = 40003  # the output vocab # embeddings.shape[0]
embedding_size = dim_en_vec  # The english inputs embeddings embeddings.shape[1]

# In[14]:


fr_wgts = [fr_embs.T, np.zeros((len(fr_vocab, )))]

# ### The model itself

# In[15]:


# attention seq2seq
inp = Input((maxlen,))
x = Embedding(vocab_size, dim_en_vec, input_length=maxlen,
              weights=[en_embs], trainable=False)(inp)

x = AttentionSeq2Seq(input_dim=dim_en_vec, input_length=maxlen, hidden_dim=128, output_length=maxlen, output_dim=128,
                     depth=3, bidirectional=True, unroll=False, stateful=False, dropout=0.1)(x)
x = TimeDistributed(Dense(dim_fr_vec))(x)
x = TimeDistributed(Dense(vocab_size, weights=fr_wgts))(x)
x = Activation('softmax')(x)

# In[16]:


model = Model(inp, x)
model.compile('adam', 'sparse_categorical_crossentropy')

# In[17]:


K.set_value(model.optimizer.lr, lr)

# In[18]:


model.summary()

# In[19]:


hist = model.fit(en_train, np.expand_dims(fr_train, axis=-1), batch_size=64, epochs=20, **params,
                 validation_data=[en_test, np.expand_dims(fr_test, axis=-1)])

# In[43]:





# In[44]:


model.save_weights(dpath + 'trans_testing_basic2_sos_eos_unk_att.h5')

# In[55]:


model.load_weights(dpath + 'trans_testing_basic2_sos_eos_unk_att.h5')


# ## Testing

# In[46]:


def sent2ids(sent):
    sent = simple_toks(sent)
    ids = [en_w2id[t] for t in sent]
    return pad_sequences([ids], maxlen, padding="post", truncating="post")


# In[47]:


def en2fr(sent):
    ids = sent2ids(sent)
    tr_ids = np.argmax(model.predict(ids), axis=-1)
    return ' '.join(fr_vocab[i] for i in tr_ids[0] if i > 0)


# In[19]:


en2fr("what is the size of canada?")

# In[20]:


en2fr("what is the size of australia?")

# In[21]:



print(fr_qs[0])
en2fr("What is light?")

# In[22]:


print(qs[50000])
en2fr("Why is the Arctic ozone layer thicker than the Antarctic ozone layer?")

# In[25]:


print(qs[9])
en2fr("Which province is the most populated?")

# In[24]:


en2fr("Who are we?")

# In[23]:


print(qs[3])
en2fr("What would we do without it?")
