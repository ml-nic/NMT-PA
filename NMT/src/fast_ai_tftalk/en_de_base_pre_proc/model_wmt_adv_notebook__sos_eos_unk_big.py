# coding: utf-8

# In[ ]:


import re

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras_tqdm import TQDMNotebookCallback
from recurrentshop import *
# import sutils; importlib.reload(sutils)
from sutils import *

# In[ ]:


print(keras.__version__)
print(tf.__version__)

# In[ ]:


path = '../neural_translation_en_de/'
dpath = '../neural_translation_en_de/translate/'

# In[ ]:



# ### Set up Regex and tokenize for use later

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


# In[ ]:



# ## Load the PreProcessed data
# 
# Here we load all the data 

data = load(dpath + 'nmt_datawmt_sos_eos_unk.pkl')
look_ups = load(dpath + 'look_upswmt_sos_eos_unk.pkl')
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

# In[ ]:


print(fr_train.shape)
print(en_train.shape)

# In[ ]:



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

tbCallBack = TensorBoard(log_dir='/data/model_newGraphadvbig_sos_eos_unk', write_graph=True)
modelCallback = ModelCheckpoint(
    'model_checkpoint_advbig_sos_eos_unk.{epoch:03d}-{loss:.3f}.hdf5',
    monitor='val_loss', verbose=1, save_best_only=False,
    save_weights_only=True, mode='auto',
    period=1)

# creating different sets of Params to easily import into the model at train time
params = {'verbose': 1, 'callbacks': [TQDMNotebookCallback(), reduce_LR, early_stopping, tbCallBack, modelCallback]}
params2 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}
params3 = {'verbose': 1, 'callbacks': [LRDrop, TQDMNotebookCallback(), reduce_LR, early_stopping]}

# In[ ]:




# #### Set some parameters for the model

# In[12]:


lr = 1e-3
maxlen = 100
dim_en_vec = 200
n_en_vec = 400000
dim_fr_vec = 200

vocab_size = len(fr_vocab)  # the output vocab # embeddings.shape[0]
embedding_size = 200  # The english inputs embeddings embeddings.shape[1]

fr_wgts = [fr_embs.T, np.zeros((len(fr_vocab, )))]

# In[9]:


# ### The model itself

# Base Model big
inp = Input((maxlen,))
x = Embedding(40003, dim_en_vec, input_length=maxlen,
              weights=[en_embs], trainable=False)(inp)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LSTM(128, return_sequences=True)(x)
x = TimeDistributed(Dense(dim_fr_vec))(x)
x = TimeDistributed(Dense(40003, weights=fr_wgts))(x)
x = Activation('softmax')(x)

model = Model(inp, x)
model.compile('adam', 'sparse_categorical_crossentropy')

K.set_value(model.optimizer.lr, lr)
model.summary()

# In[10]:


hist = model.fit(en_train, np.expand_dims(fr_train, axis=-1), batch_size=96, epochs=50, **params,
                 validation_data=[en_test, np.expand_dims(fr_test, axis=-1)])

# In[12]:


# plot_train(hist)55000epoch2
for a in hist.history:
    print(a, hist.history[a])

# In[13]:


weight_identifier = "trans_testing_basic_wmt_advbig_sos_eos_unk"
model.save_weights(dpath + weight_identifier + '.h5')
model.load_weights(dpath + weight_identifier + '.h5')

# In[23]:


# ## Testing
SOS = True
EOS = True
UNK = True


def sent2ids(sent):
    sent = simple_toks(sent)
    ids = []
    if SOS:
        ids.append(en_w2id["<SOS>"])
    for t in sent:
        try:
            ids.append(en_w2id[t])
        except KeyError:
            if UNK:
                ids.append(en_w2id["<UNK>"])
            else:
                pass
    if EOS:
        ids.append(en_w2id["<EOS>"])
    return pad_sequences([ids], maxlen, padding="post", truncating="post")


def en2fr(sent):
    ids = sent2ids(sent)
    tr_ids = np.argmax(model.predict(ids), axis=-1)
    return ' '.join(fr_vocab[i] for i in tr_ids[0] if i > 0 and fr_vocab[i] not in ["<SOS>", "<EOS>"])


# In[24]:


print(en2fr("what is the size of canada?"))

# In[25]:


print(en2fr("what is the size of australia?"))

# In[26]:


print(en2fr("What is light?"))

# In[27]:


print(en2fr("Why is the Arctic ozone layer thicker than the Antarctic ozone layer?"))

# In[28]:


print(en2fr("Which province is the most populated?"))

# In[29]:


print(en2fr("Who are we?"))

# In[30]:


print(en2fr("What would we do without it?"))

# In[33]:


print(en2fr("Hello Tom"))

# ## Prediction

# In[ ]:


import os

source_file = "/data/wrapper/PA_BA/DataSets/Validation/DE_EN_(tatoeba)_validation_english_only.txt"
if os.path.exists(source_file) is False:
    exit("source file does not exists")

source_sentences = open(source_file, encoding='UTF-8').read().split('\n')
print(len(source_sentences))

translated_sentences = []
i = 0
for sent in source_sentences:
    if i % int((len(source_sentences) / 100)) == 0:
        print(i)
    translated_sentences.append(en2fr(sent))
    i += 1
print(len(translated_sentences))

# In[ ]:


out_file = os.path.join(os.path.abspath(os.path.join(source_file, os.pardir)), weight_identifier + ".pred")
with(open(out_file, 'w', encoding='utf8')) as file:
    for sent in translated_sentences:
        file.write(sent + '\n')
