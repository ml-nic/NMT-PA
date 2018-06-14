import sys

from helpers import model_repo

"""
Starts training of a model.
Accepts model code as an argument, to run in non interactive mode.
If model code is not provided, the interactive model selection starts. 
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = model_repo.argument_model_selection(sys.argv[1])
    else:
        model = model_repo.interactive_model_selection()

    model.start_training()
