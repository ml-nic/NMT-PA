from helpers.Tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def get_gradients(model):
    for layer in model.layers:
        print(layer.get_weights())
        print("\n")


from models import WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet

seq2seq = WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet.Seq2Seq2()
seq2seq._setup_model(mode='predict')
get_gradients(seq2seq.M)

exit()
lines_en = open('C:/Users/Nicolas/Desktop/DE_EN_(tatoeba)_validation_english_only.txt', encoding='UTF-8').read().split(
    '\n')
lines_de = open('C:/Users/Nicolas/Desktop/DE_EN_(tatoeba)_validation_german_only.txt', encoding='UTF-8').read().split(
    '\n')

assert len(lines_en) == len(lines_de)
new_en = []
new_de = []
for idx in range(len(lines_en)):
    if len(lines_de[idx]) <= 30 and len(lines_en[idx]) <= 30:
        new_en.append(lines_en[idx])
        new_de.append(lines_de[idx])
with(
        open('C:/Users/Nicolas/Desktop/DE_EN_(tatoeba)_validation_english_only_shorted.txt', 'w',
             encoding='UTF-8')) as out:
    for line in new_en:
        out.write(line)
        out.write('\n')
with(open('C:/Users/Nicolas/Desktop/DE_EN_(tatoeba)_validation_german_only_shorted.txt', 'w', encoding='UTF-8')) as out:
    for line in new_de:
        out.write(line)
        out.write('\n')

exit()


def concat_en_and_de():
    lines_de = open('../../DataSets/Training/train.clean.de', encoding='UTF-8').read().split('\n')
    lines_en = open('../../DataSets/Training/train.clean.en', encoding='UTF-8').read().split('\n')
    assert len(lines_de) == len(lines_en)

    with(open('../../DataSets/Training/wmt_16_de_en_google_nmt.txt', 'w', encoding='utf8')) as file:
        for i in range(len(lines_de)):
            file.write(lines_en[i] + '\t' + lines_de[i] + '\n')


import random

with \
        open('DE_EN_(wmt16_google_nmt)_test.txt', 'w', encoding='utf8') as test, \
        open('DE_EN_(wmt16_google_nmt)_validation.txt', 'w', encoding='utf8') as val, \
        open('DE_EN_(wmt16_google_nmt)_train.txt', 'w', encoding='utf8') as train, \
        open('../../DataSets/Training/wmt_16_de_en_google_nmt.txt', 'r', encoding='utf8') as file:
    data = file.readlines()
    random.shuffle(data)

    lines = 1500000
    train_lines = int(lines * 0.7)
    val_lines = int(lines * 0.2)
    test_lines = int(lines * 0.1)
    print('Train: {} \t Val: {} \t Test: {}'.format(train_lines, val_lines, test_lines))
    for i in range(lines):
        if i <= train_lines:
            print(data[i], file=train, end='')
        elif train_lines < i < train_lines + val_lines:
            print(data[i], file=val, end='')
        else:
            print(data[i], file=test, end='')

exit()
input_texts = []
target_texts = []
lines = open('../../DataSets/Training/deu.txt', encoding='UTF-8').read().split('\n')
for line in lines:
    input_text, target_text = line.split('\t')
    input_texts.append(input_text)
    target_text = target_text
    target_texts.append(target_text)
num_samples = len(input_texts)

en_tokenizer = Tokenizer("GO_", "_EOS", "_UNK",
                         num_words=30000)
en_tokenizer.fit_on_texts(input_texts)
train_input_texts = en_tokenizer.texts_to_sequences(input_texts)
lengths = []
for text in train_input_texts:
    lengths.append(len(text))
print(len(lengths))
import numpy as np

print(np.max(np.array(lengths)))
train_input_texts = pad_sequences(train_input_texts, maxlen=100,
                                  padding='post',
                                  truncating='post')
en_word_index = en_tokenizer.word_index

exit()

input_texts = []
target_texts = []
lines = open('../../DataSets/Training/deu.txt', encoding='UTF-8').read().split('\n')
for line in lines:
    input_text, target_text = line.split('\t')
    input_texts.append(input_text)
    target_text = target_text
    target_texts.append(target_text)
num_samples = len(input_texts)

tokenizer = Tokenizer(num_words=20000000)
tokenizer.fit_on_texts(input_texts + target_texts)
hm_words_at_all = len(tokenizer.word_index)
print(hm_words_at_all)

tokenizer = Tokenizer(num_words=20000000)
tokenizer.fit_on_texts(input_texts)
print(len(tokenizer.word_index))

tokenizer = Tokenizer(num_words=20000000)
tokenizer.fit_on_texts(target_texts)
print(len(tokenizer.word_index))

# Gesamt 42789 Wörter
# Englisch 14434 Wörter
# Deutsch 30328 Wörter
