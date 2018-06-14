"""
Merges two seperate training files, where each one contains one language, into one tab seperated file.
"""

BASE_DATA_DIR = '../../DataSets/Validation/'

en = 'newstest2013.en'
de = 'newstest2013.de'

out = 'merged_en_de.txt'
import os

with(open(os.path.join(BASE_DATA_DIR, en), encoding='utf8')) as file:
    data_en = file.readlines()
with(open(os.path.join(BASE_DATA_DIR, de), encoding='utf8')) as file:
    data_de = file.readlines()

print(len(data_en) == len(data_de))
out_data = []
for line_idx in range(len(data_en)):
    out_data.append(data_en[line_idx].replace("\n", "") + "\t" + data_de[line_idx].replace("\n", "") + "\n")
print(out_data[0])
with(open(os.path.join(BASE_DATA_DIR, out), 'w', encoding='utf8')) as file:
    file.writelines(out_data)
print(len(data_en) == len(data_de) == len(out_data))
