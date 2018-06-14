import numpy as np

"""
Simple Filesplitter that reads the source into memory and splits it into three separate files 
with the ration 70-20-10 
"""
with \
        open('DE_EN_(tatoeba)_test.txt', 'w', encoding='utf8') as test, \
        open('DE_EN_(tatoeba)_validation.txt', 'w', encoding='utf8') as val, \
        open('DE_EN_(tatoeba)_train.txt', 'w', encoding='utf8') as train, \
        open('../../../DataSets/Training/deu.txt', 'r', encoding='utf8') as file:
    data = np.array(file.readlines())
    np.random.shuffle(data)
    #    data = random.shuffle(data)
    lines = len(data)
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
