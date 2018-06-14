import os
import sys

from helpers import model_repo

"""
Interactive inference of pretrained our_implementation.models via terminal.
"""


# C:/Users/Nicolas/Desktop/own_validation_data.en
# C:/Users/Nicolas/Desktop/deu_val_data.en

def predict_interactive_per_sentence(inference_model):
    """
    Asks the user for a source sentence and translates them with the given model.
    :param inference_model: the model which should be used for prediction
    """
    while True:
        print("\n\nPlease type in the sentence which should be translated:")
        source_sentence = input()
        if source_sentence == '\q':
            exit()
        if source_sentence == '\m':
            inference_model = model_repo.interactive_model_selection()
            continue
        target_sentence = inference_model.predict_one_sentence(source_sentence)

        print("Source sentence:\n", source_sentence)
        print("Translated sentence:\n", target_sentence)


def predict_interactive_from_file(model):
    source_file = input('Path of source file:\n')
    if os.path.exists(source_file) is False:
        exit("source file does not exists")
    out_file_name = input('suffix of output file:\n')
    all_weights = input("predict for every weight file?\n")

    split_flag = input("split source file with tab?\n")

    source_sentences = open(source_file, encoding='UTF-8').read().split('\n')
    print(len(source_sentences))
    if split_flag in ['y', 'yes']:
        temp_source_sentences = source_sentences.copy()
        source_sentences = []
        for line in temp_source_sentences:
            input_text, _ = line.split('\t')
            source_sentences.append(input_text)

    if all_weights in ['y', 'yes']:
        predictions_per_weight_file = model.predict_batch(source_sentences, all_weights=True)
    else:
        predictions_per_weight_file = model.predict_batch(source_sentences)
    for weight_identifier in predictions_per_weight_file:
        out_file = os.path.join(os.path.abspath(os.path.join(source_file, os.pardir)),
                                model.identifier + "_" + weight_identifier.split('.hdf5')[0] + ".pred")
        with(open(out_file, 'w')) as file:
            for sent in predictions_per_weight_file[weight_identifier]:
                file.write(sent + '\n')


if len(sys.argv) > 1:
    model = model_repo.argument_model_selection(sys.argv[1])
    model.predict_batch(sys.argv[2])
else:
    model = model_repo.interactive_model_selection()
    mode = -1
    while mode == -1:
        print("Model", model.identifier, "selected")
        print("Mode: \n0: per sentence prediction\n1: batch prediction")
        mode = input()

    if mode == '0':
        predict_interactive_per_sentence(model)
    elif mode == '1':
        predict_interactive_from_file(model)
