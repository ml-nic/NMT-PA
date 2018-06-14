import os

from metrics.Bleu import Bleu

a = "../../Evaluations/"


def evaluate_each_pred_file():
    with(open("../../output.csv", "w")) as out_csv_file:
        out_csv_file.write(
            "model;weight;epoch;our_bleu;google_bleu;smoothed_bleu;max_our_epoch;max_google_epoch;max_smooted_epoch\n")
    for element in os.listdir(a):
        sub_dir = os.path.join(a, element)
        if os.path.isdir(sub_dir):
            model_name = sub_dir.split("/")[-1]
            print("Model name:", model_name)
            if "en2fr" in model_name.lower() or "e2f" in model_name.lower():
                print("jjj")
                reference_file = "../../DataSets/fast_ai_tftalk/en_fr_base_tftalk/validation_questions.tok.fr.lower"
                reference_file_capitalized = "../../DataSets/fast_ai_tftalk/en_fr_base_tftalk/validation_questions.tok.fr"
            elif "en2de" in model_name.lower() and "bpe" not in model_name.lower():
                print("fff")
                reference_file = "../../hparams/generate_vocabs_with_different_size_with_bpe/wmt16_de_en/newstest2013.tok.de.lower"
                reference_file_capitalized = "../../hparams/generate_vocabs_with_different_size_with_bpe/wmt16_de_en/newstest2013.tok.de"
            elif "en2de" in model_name.lower() and "bpe" in model_name.lower():
                print("kk")
                reference_file = "../../hparams/generate_vocabs_with_different_size_with_bpe/wmt16_de_en/newstest2013.tok.bpe.30000.de"
                reference_file_capitalized = None
            else:
                print("error at model:", model_name)
                exit(-1)
            for sub_element in os.listdir(sub_dir):
                sub_sub_dir = os.path.join(a, element, sub_element)
                if sub_sub_dir.endswith("predictions"):
                    max_our_bleu = 0.0
                    max_google_bleu = 0.0
                    max_smoothed_bleu = 0.0
                    epoch_of_max_our_bleu = -1
                    epoch_of_max_google_bleu = -1
                    epoch_of_max_smoothed_bleu = -1
                    for pred_file in os.listdir(sub_sub_dir):
                        if pred_file.endswith(".pred"):
                            weight_file = pred_file.split(".pred")[0]
                            print("Weight file:", weight_file)
                            if len(weight_file.split(".")) == 1:
                                continue
                            translation_file = os.path.join(sub_sub_dir, pred_file)

                            epoch = int(weight_file.split(".")[1].split("-")[0])
                            our_bleu, google_bleu, smoothed_bleu = Bleu(model_name, "BLEU", epoch,
                                                                        timestamp=False).evaluate_hypothesis_corpus(
                                translation_file, reference_file)
                            cap_our_bleu, cap_google_bleu, cap_smoothed_bleu = Bleu(model_name, "BLEU", epoch,
                                                                                    timestamp=False).evaluate_hypothesis_corpus(
                                translation_file, reference_file_capitalized)
                            if cap_our_bleu > our_bleu:
                                print("cap_bleu is higher than our_bleu", cap_our_bleu)
                            if cap_google_bleu > google_bleu:
                                print("cap_bleu is higher than google_bleu", cap_google_bleu)
                            if cap_smoothed_bleu > smoothed_bleu:
                                print("cap_bleu is higher than smoothed_bleu", cap_smoothed_bleu)

                            if our_bleu > max_our_bleu:
                                max_our_bleu = our_bleu
                                epoch_of_max_our_bleu = epoch
                            if google_bleu > max_google_bleu:
                                max_google_bleu = google_bleu
                                epoch_of_max_google_bleu = epoch
                            if smoothed_bleu > max_smoothed_bleu:
                                max_smoothed_bleu = smoothed_bleu
                                epoch_of_max_smoothed_bleu = epoch

                            with(open("../../output.csv", "a")) as out_csv_file:
                                out_csv_file.write(
                                    model_name + ";" + weight_file + ";" + str(epoch) + ";" + str(our_bleu) + ";" + str(
                                        google_bleu) + ";" + str(smoothed_bleu) + "\n")

                    print("Max our bleu is", str(max_our_bleu), "at epoch", epoch_of_max_our_bleu)
                    print("Max google bleu is", str(max_google_bleu), "at epoch", epoch_of_max_google_bleu)
                    print("Max smoothed bleu is", str(max_smoothed_bleu), "at epoch", epoch_of_max_smoothed_bleu)
                    with(open("../../output.csv", "a")) as out_csv_file:
                        out_csv_file.write(
                            model_name + ";" + "finished" + ";" + "" + ";" + str(max_our_bleu) + ";" + str(
                                max_google_bleu) + ";" + str(max_smoothed_bleu) + ";" + str(
                                epoch_of_max_our_bleu) + ";" + str(epoch_of_max_google_bleu) + ";" + str(
                                epoch_of_max_smoothed_bleu) + "\n")


evaluate_each_pred_file()
