# coding: utf-8

# # English to French using Neural Machine Translation
# 
# This example was taken from the wonderful Cutting Edge Deep Learning for Coders course as taught by Jeremy Howard http://course.fast.ai/part2.html The course is now live and I encourage you to check it out.

# In[1]:



import re

import keras
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback
from recurrentshop import *
# import sutils; importlib.reload(sutils)
from sutils import *

# In[2]:


print(keras.__version__)
print(tf.__version__)

# In[3]:


# In[4]:


# limit_gpu_mem()


# In[5]:


path = '/data/TensorFlowTalks/neural_translation_en_de/'
dpath = '/data/TensorFlowTalks/neural_translation_en_de/translate/'

# ### Set up Regex and tokenize for use later

# In[6]:


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

# In[7]:


data = load(dpath + 'nmt_data.pkl')
look_ups = load(dpath + 'look_ups.pkl')
fr_train = data['fr_train']
fr_test = data['fr_test']
en_train = data['en_train']
en_test = data['en_test']
en_w2id = look_ups['en_w2id']
fr_vocab = look_ups['fr_vocab']
en_vocab = look_ups['en_vocab']
en_embs = look_ups['en_embs']
fr_embs = look_ups['fr_embs']

questions = load(dpath + 'questions.pkl')
# print(questions[10])
en_qs, fr_qs = zip(*questions)

# In[8]:


# for running model test on small set of data
# fr_train = fr_train[:5000]
# en_train = fr_train[:5000]

print(fr_train.shape)

# In[9]:


print(en_train.shape)

# ## Model

# #### Create some Keras Callbacks to handle early stopping and Learning Rate changes

# In[10]:


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

# In[11]:


# creating different sets of Params to easily import into the model at train time
params = {'verbose': 1, 'callbacks': [TQDMNotebookCallback(), reduce_LR, early_stopping]}
params2 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}
params3 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}

# #### Set some parameters for the model

# In[12]:


lr = 1e-3
maxlen = 30
dim_en_vec = 100
n_en_vec = 400000
dim_fr_vec = 200

vocab_size = len(fr_vocab)  # the output vocab # embeddings.shape[0]
embedding_size = 100  # The english inputs embeddings embeddings.shape[1]

# In[13]:


fr_wgts = [fr_embs.T, np.zeros((len(fr_vocab, )))]

# ### The model itself

# In[ ]:


# Test different settings:

# - only two LSTM's and one TimeDistributed
# - LSTM instead of Bidirectional
# - only one timeDistributed
# - without weights for german embeddings
# - categorical_crossentropy instead of sparse



# while training implement my preprocessing into jupyter notebook and try my prepro with this standard model


# In[ ]:




# - without fr_wgts
inp = Input((maxlen,))
x = Embedding(len(en_vocab), dim_en_vec, input_length=maxlen,
              weights=[en_embs], trainable=False)(inp)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LSTM(128, return_sequences=True)(x)
x = TimeDistributed(Dense(dim_fr_vec))(x)
x = TimeDistributed(Dense(len(fr_vocab)))(x)
x = Activation('softmax')(x)

model = Model(inp, x)
model.compile('adam', 'sparse_categorical_crossentropy')

# In[15]:


K.set_value(model.optimizer.lr, lr)

# In[16]:


model.summary()

# In[ ]:


hist = model.fit(en_train, np.expand_dims(fr_train, axis=-1), batch_size=64, epochs=20, **params,
                 validation_data=[en_test, np.expand_dims(fr_test, axis=-1)])

# In[ ]:


# plot_train(hist)


# In[ ]:


model.save_weights(dpath + 'trans_testing_basic_without_fr_weights.h5')

# In[ ]:


model.load_weights(dpath + 'trans_testing_basic_without_fr_weights.h5')


# ## Testing

# In[ ]:


def sent2ids(sent):
    sent = simple_toks(sent)
    ids = [en_w2id[t] for t in sent]
    return pad_sequences([ids], maxlen, padding="post", truncating="post")


# In[ ]:


def en2fr(sent):
    ids = sent2ids(sent)
    tr_ids = np.argmax(model.predict(ids), axis=-1)
    return ' '.join(fr_vocab[i] for i in tr_ids[0] if i > 0)


# In[ ]:


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


# ## Predict validation data

# In[ ]:


# TODO implement prediction of val data, to calculate bleu
