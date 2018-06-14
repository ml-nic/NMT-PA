import sys

from helpers import model_repo

BASE_DIR = "C:/Users/Nicolas/Desktop/"


def save_hiddenstates(hiddenstates, output_file, sentences):
    with(open(output_file, 'a')) as out_file:
        counter = 0
        for batch in hiddenstates:
            for i in range(batch.shape[0]):
                out_file.write(sentences[counter] + ' ')
                if len(batch.shape) == 2:
                    for element in batch[i]:
                        out_file.write(str(element) + ' ')
                else:
                    for element in batch[i][batch.shape[1] - 1]:
                        out_file.write(str(element) + ' ')
                out_file.write('\n')
                counter += 1

if __name__ == '__main__':
    if len(sys.argv) > 2:
        model = model_repo.argument_model_selection(sys.argv[1])
        in_file = sys.argv[2]
        split_flag = sys.argv[3]
        suffix = sys.argv[4]

        source_sentences = open(in_file, encoding='UTF-8').read().split('\n')
        if split_flag in ['y', 'yes']:
            temp_source_sentences = source_sentences.copy()
            source_sentences = []
            for line in temp_source_sentences:
                input_text, _ = line.split('\t')
                source_sentences.append(input_text)

        hiddenstates = model.calculate_hiddenstate_after_encoder(source_sentences)
        save_hiddenstates(hiddenstates, BASE_DIR + "hidden_states_" + suffix + ".txt",
                          [sent.replace(' ', '_') for sent in source_sentences])

    else:
        model = model_repo.interactive_model_selection()

    mode = input("sent or batch\n")
    suffix = input("File suffix:")
    if mode == '0':
        while True:
            sentence = input("Which sentence should be used?")
            if sentence == '\q':
                exit()
            sentence = [sentence]
            source_sentences = sentence.copy()
            hiddenstates = model.calculate_hiddenstate_after_encoder(sentence)
            save_hiddenstates(hiddenstates, BASE_DIR + "hidden_states_" + suffix + ".txt",
                              [sent.replace(' ', '_') for sent in source_sentences])
    elif mode == '1':
        in_file = input("source file\n")
        lines = open(in_file, encoding='UTF-8').read().split('\n')
        print(len(lines))
        print(lines[0])
        source_sentences = lines.copy()
        hiddenstates = model.calculate_hiddenstate_after_encoder(lines)
        save_hiddenstates(hiddenstates, BASE_DIR + "hidden_states_" + suffix + ".txt",
                          [sent.replace(' ', '_') for sent in source_sentences])

    else:
        exit("Only mode 0 and 1 are available")
