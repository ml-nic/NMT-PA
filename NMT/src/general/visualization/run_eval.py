import os
import sys

from helpers import model_repo
from metrics.Bleu import Bleu

if len(sys.argv) > 1:
    model = model_repo.argument_model_selection(sys.argv[1])
else:
    mode = input("model selection (1) or name (2)")
    if mode == 1:
        model = model_repo.interactive_model_selection()
        model_name = model.identifier
    else:
        model_name = input("name of the model")

references_file = input("reference_file:\n")
if references_file == "":
    references_file = "C:/Users/Nicolas/Desktop/DE_EN_(tatoeba)_validation_german_only.txt"

hypothesis_file = input("hypothesis file:\n")
if len(hypothesis_file) < 2:
    base_hypothesis_path = "C:/Users/Nicolas/Desktop/Predictions"
    hypothesis_file = base_hypothesis_path + model.identifier + "_validation_predictions" + '.txt'

if not os.path.exists(hypothesis_file):
    with open(hypothesis_file, 'w'):
        pass

bleu_evaluator = Bleu(model_name, 'bleu', timestamp=True, epoch=1)
bleu_evaluator.evaluate_hypothesis_corpus(hypothesis_file, references_file)
