from bs4 import BeautifulSoup as bs

input_file_path_name = 'newstest2015-deen-src.de.sgm'
output_file_path = 'newstest2015.de'
file = open(input_file_path_name, 'r', encoding='utf-8')
data = file.read()
soup = bs(data)

sentences = soup.find_all('seg')
print(sentences)

with open(output_file_path, 'w', encoding='utf-8') as output:
    for sentence in sentences:
        print(sentence.text, file=output)
