import nltk.probability as p
import nltk.tokenize as tk
from nltk.corpus import stopwords

file_path = '../../DataSets/Test/DE_EN_(tatoeba)_test.txt'

text = ''
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        text += line.split('\t')[0].lower().replace('.', '').replace('?', '') + ' '

blob = tk.casual_tokenize(text, strip_handles=True)

better_blob = []
stop_words = set(stopwords.words('english'))

for word in blob:
    if not (len(word) <= 3 or word == 'mary' or word == "tom's" or word == "mary's"):
        better_blob.append(word)

filtered_blob = list(filter(lambda w: not w in stop_words, better_blob))


heu = p.FreqDist(filtered_blob).most_common(100)

for i in range(len(heu)):
    print(heu[i])

