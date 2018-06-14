import os


def get_max_score_for_epoch(file):
    our_max_bleu = -1.0
    google_max_bleu = -1.0
    smoothed_max_bleu = -1.0
    epoch_of_our_max_bleu = -1
    epoch_of_google_max_bleu = -1
    epoch_of_smoothed_max_bleu = -1
    with(open(file)) as in_file:
        lines = in_file.readlines()
    for line in lines:
        if line.startswith("ini"):
            continue
        our_bleu = float(line.split("\t")[1].split(" ")[1])
        google_bleu = float(line.split("\t")[5].split(" ")[1])
        smoothed_bleu = float(line.split("\t")[6].split(" ")[1])
        epoch = int(line.split("\t")[2].split(" ")[1])
        if our_bleu > our_max_bleu:
            our_max_bleu = our_bleu
            epoch_of_our_max_bleu = epoch
        if google_bleu > google_max_bleu:
            google_max_bleu = google_bleu
            epoch_of_google_max_bleu = epoch
        if smoothed_bleu > smoothed_max_bleu:
            smoothed_max_bleu = smoothed_bleu
            epoch_of_smoothed_max_bleu = epoch

    return epoch, our_max_bleu, google_max_bleu, smoothed_max_bleu, epoch_of_our_max_bleu, epoch_of_google_max_bleu, epoch_of_smoothed_max_bleu


models = [
    # Question dataset:
    "EN2FR_fastai_model_fastai_pre_proc",
    "EN2FR_fastai_model_fastai_pre_proc_attention",
    "EN2FR_fastai_model_fastai_pre_proc_hidden_dim_1024",
    "EN2FR_our_model_fastai_pre_proc",
    "EN2FR_our_model_fastai_pre_proc_hidden_dim_1024",
    # Tatoeba dataset:
    "EN2DE_fastai_model_fastai_pre_proc",
    "EN2DE_fastai_model_fastai_pre_proc_attention",
    "EN2DE_fastai_model_fastai_pre_proc_hidden_dim_1024",
    "EN2DE_our_model_fastai_pre_proc",
    "EN2DE_our_model_fastai_pre_proc_hidden_dim_1024",
    # WMT16 1mio
    "EN2DE_fastai_model_fastai_pre_proc_wmt16_1000000",
    "EN2DE_fastai_model_fastai_pre_proc_attention_wmt16_1000000",
    "EN2DE_fastai_model_fastai_pre_proc_hidden_dim_1024_wmt16_1000000",
    "EN2DE_our_model_fastai_pre_proc_wmt16_1000000",
    # "EN2DE_our_model_fastai_pre_proc_hidden_dim_1024_wmt16_1000000",
    # WMT16 big
    # "EN2DE_fastai_model_fastai_pre_proc_wmt16",
    "EN2DE_fastai_model_fastai_pre_proc_attention_wmt16",
    "EN2DE_fastai_model_fastai_pre_proc_hidden_dim_1024_wmt16",
    "EN2DE_our_model_fastai_pre_proc_wmt16",
    "EN2DE_our_model_fastai_pre_proc_hidden_dim_1024_wmt16",
    # two more models with bpe:
    "EN2DE_fastai_model_fastai_pre_proc_wmt16_bpe30000"
]
for model_name in models:
    print("\n" + model_name)
    basepath = "C:/Users/Nicolas/Dropbox/Evaluations/"
    score_file_path = basepath + model_name + "/bleu_scores/"
    stat_file = basepath + model_name + "/statistics.csv"
    score_files = os.listdir(score_file_path)
    with(open(stat_file, "w")) as out_csv_file:
        out_csv_file.write(
            "model_name;epoch;our_bleu;google_bleu;smoothed_bleu;epoch_of_our_bleu;epoch_of_google_bleu;epoch_of_smoothed_bleu\n")
    for score_file in score_files:
        epoch, our_bleu, google_bleu, smoothed_bleu, epoch_of_our_bleu, epoch_of_google_bleu, epoch_of_smoothed_bleu = get_max_score_for_epoch(
            score_file_path + score_file)
        with(open(stat_file, "a")) as out_csv_file:
            out_csv_file.write(
                model_name + ";" + str(epoch) + ";" + str(our_bleu) + ";" + str(google_bleu) + ";" + str(
                    smoothed_bleu) + ";" + str(epoch_of_our_bleu) + ";" + str(epoch_of_google_bleu) + ";" + str(
                    epoch_of_smoothed_bleu) + "\n")
            # we do not need to save the weight file, because we take only the highest bleu of each epoch, so we implicit skip bad weight files of the same epoch
