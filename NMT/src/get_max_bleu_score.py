import os


def get_max_score(path):
    scores = []
    our_max_bleu = 0.0
    google_max_bleu = 0.0
    smoothed_max_bleu = 0.0
    epoch_of_our_max_bleu = -1.0
    epoch_of_google_max_bleu = -1.0
    epoch_of_smoothed_max_bleu = -1.0
    max_epoch = -1
    score_files = os.listdir(path)
    for score_file in score_files:
        with(open(path + score_file)) as in_file:
            lines = in_file.readlines()
            for line in lines:
                if line.startswith("ini"):
                    continue
                our_bleu = float(line.split("\t")[1].split(" ")[1])
                google_bleu = float(line.split("\t")[5].split(" ")[1])
                smoothed_bleu = float(line.split("\t")[6].split(" ")[1])
                epoch = int(line.split("\t")[2].split(" ")[1])
                if epoch > max_epoch:
                    max_epoch = epoch
                if our_bleu > our_max_bleu:
                    our_max_bleu = our_bleu
                    epoch_of_our_max_bleu = epoch
                if google_bleu > google_max_bleu:
                    google_max_bleu = google_bleu
                    epoch_of_google_max_bleu = epoch
                if smoothed_bleu > smoothed_max_bleu:
                    smoothed_max_bleu = smoothed_bleu
                    epoch_of_smoothed_max_bleu = epoch

    print(str(our_max_bleu).replace(".", ",") + "\t" + str(google_max_bleu).replace(".", ",") + "\t" + str(
        smoothed_max_bleu).replace(".", ",") + "\t" + str(epoch_of_smoothed_max_bleu) + "\t" + str(
        epoch_of_our_max_bleu) + "\t" + str(epoch_of_google_max_bleu) + "\tmax_epoch: " + str(max_epoch))


models = [
    # Question dataset:
    # fin# "EN2FR_fastai_model_fastai_pre_proc",
    # fin# "EN2FR_fastai_model_fastai_pre_proc_attention",
    # fin# "EN2FR_fastai_model_fastai_pre_proc_hidden_dim_1024",
    # fin# "EN2FR_our_model_fastai_pre_proc",
    # fin# "EN2FR_our_model_fastai_pre_proc_hidden_dim_1024",
    # Tatoeba dataset:
    # fin# "EN2DE_fastai_model_fastai_pre_proc",
    # fin# "EN2DE_fastai_model_fastai_pre_proc_attention",
    # fin# "EN2DE_fastai_model_fastai_pre_proc_hidden_dim_1024",
    # fin# "EN2DE_our_model_fastai_pre_proc",
    # fin# "EN2DE_our_model_fastai_pre_proc_hidden_dim_1024",
    # WMT16 1mio
    # fin# "EN2DE_fastai_model_fastai_pre_proc_wmt16_1000000",
    # fin# "EN2DE_fastai_model_fastai_pre_proc_attention_wmt16_1000000",
    # fin# "EN2DE_fastai_model_fastai_pre_proc_hidden_dim_1024_wmt16_1000000",
    # fin# "EN2DE_our_model_fastai_pre_proc_wmt16_1000000",
    "EN2DE_our_model_fastai_pre_proc_hidden_dim_1024_wmt16_1000000",
    # WMT16 big
    # fin# "EN2DE_fastai_model_fastai_pre_proc_wmt16",
    "EN2DE_fastai_model_fastai_pre_proc_attention_wmt16",
    "EN2DE_fastai_model_fastai_pre_proc_hidden_dim_1024_wmt16",
    # fin# "EN2DE_our_model_fastai_pre_proc_wmt16",
    # fin# "EN2DE_our_model_fastai_pre_proc_hidden_dim_1024_wmt16",
    # two more models with bpe:
    "EN2DE_fastai_model_fastai_pre_proc_wmt16_bpe30000"
]

for model_name in models:
    print("\n" + model_name)
    basepath = "../../Evaluations/"
    path = basepath + model_name + "/bleu_scores/"
    get_max_score(path)
