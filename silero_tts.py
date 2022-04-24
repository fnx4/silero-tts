#!/usr/bin/env python

import os
import re

import torch
import nltk
import transliterate
import num2words

###################################################
# repo: https://github.com/snakers4/silero-models #
# commit 941f911858f51c0cee6cad09d862c00bd8221204 #
###################################################

if __name__ == "__main__":
    # nltk.download("punkt") # if needed

    device = torch.device("cpu")
    torch.set_num_threads(16)
    sample_rate = 48000  # 8000, 24000, 48000

    local_file = "ru_v3.pt"
    speaker = "xenia"  # "aidar", "baya", "kseniya", "xenia"
    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/ru_v3.pt", local_file)

    # local_file_multi = "v2_multi.pt"
    # multi_speakers = ["baya", "kseniya", "irina", "ruslan", "natasha", "thorsten", "tux", "gilles", "lj", "dilyara"]
    # if not os.path.isfile(local_file_multi):
    #     torch.hub.download_url_to_file("https://models.silero.ai/models/tts/multi/v2_multi.pt", local_file_multi)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    # model_multi = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    # model_multi.to(device)

    out_folder = "result/" # TODO dynamic
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    source_file = open("text.txt", "r", encoding="utf-8") # TODO dynamic
    source_lines = source_file.read().splitlines()
    size = str(len(source_lines))
    source_file.close()

    wrn = []

    for line_num, text in enumerate(source_lines):
        text = re.sub("[^A-Za-z0-9А-Яа-яЁё_\s .,;!№$%&?+–-]+", "", text.strip())
        sentences = nltk.sent_tokenize(text) # TODO accent
        for sentence_num, sentence in enumerate(sentences):
            file_name = out_folder + "tts_" + str(line_num).zfill(len(size)) + "_" + str(sentence_num).zfill(3) + ".wav"
            print(file_name + " [" + str(line_num).zfill(len(size)) + "/" + size + "]: " + sentence)
            sentence = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0)), lang="ru"), sentence)
            # print(sentence)
            if text.strip() != "" and not os.path.exists(file_name):
                try:
                    audio_paths = model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)
                except ValueError:
                    # audio_paths = model_multi.save_wav(text=sentence, speakers=multi_speakers, sample_rate=sample_rate) # TODO langs
                    print("ValueError EX!")
                    wrn_text = str(line_num) + ": " + text + " -> " + sentence
                    print(wrn_text)
                    wrn.append(wrn_text)
                    sentence = transliterate.translit(sentence, language_code="ru")
                    audio_paths = model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)

    out_dir_file = open("out_files.txt", "w+", encoding="utf-8")
    for path, dirs, files in os.walk(out_folder):
        for file in files:
            out_dir_file.write("file " + path + file + "\n")
    out_dir_file.close()
    # os.system("ffmpeg -f concat -safe 0 -i out_files.txt -c copy -y final_output.wav") # TODO aevalsrc, w/o external

    print("\nFinished!")
    print("Warnings: ")
    print(wrn)
    exit(0)

