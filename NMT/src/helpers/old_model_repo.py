from models import CharSeq2SeqTutOneHotInput
from models import WordBasedSeq2Seq1000Units20EpochsFastText
from models import WordBasedSeq2Seq1000Units20EpochsGLOVE
from models import WordBasedSeq2Seq1000Units20EpochsGLOVEBig
from models import WordBasedSeq2Seq1000Units20EpochsGLOVEBigReverse
from models import WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet
from models import WordBasedSeq2Seq1000Units20EpochsGLOVEReverseInput
from models import model_2_token_also_at_encoder_unk

"""
models = {'WordBasedSeq2Seq1000Units20EpochsGLOVE': '0',
          'WordBasedSeq2Seq1000Units20EpochsFastText': '1',
          'model_2_token_also_at_encoder_unk': '2',
          'WordBasedSeq2Seq1000Units20EpochsGLOVEReverseInput': '3',
          'WordBasedSeq2Seq1000Units20EpochsGLOVEBig': '4',
          'WordBasedSeq2Seq1000Units20EpochsGLOVEBigReverse': '5',
          'CharSeq2SeqTutOneHotInput': '6',
          'WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet': '7',
          'google_baseline_en_de': '8',
          'EN2DE_fastai_model_fastai_pre_proc': '9',
          'EN2DE_fastai_model_our_pre_proc' : '10',
          'EN2DE_our_model_fastai_pre_proc' : '11',
          'EN2DE_our_model_our_pre_proc' : '12',
          'EN2FR_fastai_model_fastai_pre_proc' : '13',
          'EN2FR_our_model_fastai_pre_proc' : '14'
          }
"""

models = {'CharSeq2SeqTutOneHotInput': '0',
          'google_baseline_en_de': '1',
          'EN2DE_fastai_model_fastai_pre_proc': '2',
          'EN2DE_fastai_model_our_pre_proc': '3',
          'EN2DE_our_model_fastai_pre_proc': '4',
          'EN2DE_our_model_our_pre_proc': '5',
          'EN2FR_fastai_model_fastai_pre_proc': '6',
          'EN2FR_our_model_fastai_pre_proc': '7'
          }


def determine_model(searched_model):
    model = -1
    if searched_model == 'WordBasedSeq2Seq1000Units20EpochsGLOVE' or searched_model == models[
        'WordBasedSeq2Seq1000Units20EpochsGLOVE']:
        model = WordBasedSeq2Seq1000Units20EpochsGLOVE.Seq2Seq2()

    elif searched_model == 'WordBasedSeq2Seq1000Units20EpochsFastText' or searched_model == models[
        'WordBasedSeq2Seq1000Units20EpochsFastText']:
        model = WordBasedSeq2Seq1000Units20EpochsFastText.Seq2Seq2()

    elif searched_model == 'model_2_token_also_at_encoder_unk' or searched_model == models[
        'model_2_token_also_at_encoder_unk']:
        model = model_2_token_also_at_encoder_unk.Seq2Seq2()

    elif searched_model == 'WordBasedSeq2Seq1000Units20EpochsGLOVEReverseInput' or searched_model == models[
        'WordBasedSeq2Seq1000Units20EpochsGLOVEReverseInput']:
        model = WordBasedSeq2Seq1000Units20EpochsGLOVEReverseInput.Seq2Seq2()

    elif searched_model == 'WordBasedSeq2Seq1000Units20EpochsGLOVEBig' or searched_model == models[
        'WordBasedSeq2Seq1000Units20EpochsGLOVEBig']:
        model = WordBasedSeq2Seq1000Units20EpochsGLOVEBig.Seq2Seq2()

    elif searched_model == 'WordBasedSeq2Seq1000Units20EpochsGLOVEBigReverse' or searched_model == models[
        'WordBasedSeq2Seq1000Units20EpochsGLOVEBigReverse']:
        model = WordBasedSeq2Seq1000Units20EpochsGLOVEBigReverse.Seq2Seq2()

    elif searched_model == 'CharSeq2SeqTutOneHotInput' or searched_model == models[
        'CharSeq2SeqTutOneHotInput']:
        identifier = interactive_identifier_selection(CharSeq2SeqTutOneHotInput.get_identifiers())
        model = CharSeq2SeqTutOneHotInput.Seq2Seq2(identifier)

    elif searched_model == 'WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet' or searched_model == models[
        'WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet']:
        model = WordBasedSeq2Seq1000Units20EpochsGLOVELargeDataSet.Seq2Seq2()

    elif searched_model == 'google_baseline_en_de' or searched_model == models[
        'google_baseline_en_de']:
        model = google.Seq2Seq()

    elif searched_model == 'EN2DE_fastai_model_fastai_pre_proc' or searched_model == models[
        'EN2DE_fastai_model_fastai_pre_proc']:
        from models.en_de import EN2DE_fastai_model_fastai_pre_proc
        model = EN2DE_fastai_model_fastai_pre_proc.Seq2Seq('EN2DE_fastai_model_fastai_pre_proc')

    return model


def print_models():
    for model in models:
        print("- ", model, ":", models[model])


def interactive_model_selection():
    print("Which model do you want to use?")

    model = -1
    while model == -1:
        print_models()
        choosed_model_code = input()
        if choosed_model_code == '\q':
            exit(0)

        model = determine_model(choosed_model_code)
        print("choosed model", model.identifier)
        if model == -1:
            print("\nThis model doesn't exists. Following models are allowed:")
    return model


def argument_model_selection(argument):
    model = determine_model(argument)
    if model == -1:
        exit("Error: Model does not exists")
    return model


def interactive_identifier_selection(identifiers):
    for i, identifier in enumerate(identifiers):
        print(i, identifier)

    i = input("Which config should be used?\n")
    return identifiers[int(i)]
