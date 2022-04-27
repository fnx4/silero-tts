#!/usr/bin/env python

import os
import re
import argparse

import torch
import nltk
import transliterate
import num2words


###################################################
# repo: https://github.com/snakers4/silero-models #
# commit f8a0190b29ca20c139b725e9ffb95a08633da0a0 #
###################################################

wrn = [] # TODO exact file ptr
parser = argparse.ArgumentParser(description="tts", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 epilog="Example: ./silero_tts.py -i chapter.txt -o relative/path/to/result -t 16 -s xenia -d cuda -r 48000")


def open_file(source, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file = open(source, "r", encoding="utf-8")
    lines = file.read().splitlines()
    file.close()
    size = str(len(lines))
    tts(lines, size, out_folder)


def tts(lines, size, out_folder):
    for line_num, text in enumerate(lines):
        text = str(text).replace("…", ",")
        text = re.sub("[^A-Za-z0-9А-Яа-яЁё_\s .,;!№$%&?+–-]+", "", text.strip())
        sentences = nltk.sent_tokenize(text) # TODO accent
        print(sentences)
        for sentence_num, sentence in enumerate(sentences):
            file_name = out_folder + "/tts_" + str(line_num).zfill(len(size)) + "_" + str(sentence_num).zfill(3) + ".wav"
            print(file_name + " [" + str(line_num).zfill(len(size)) + "/" + size + "]: " + sentence)
            sentence = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0)), lang="ru"), sentence)
            # print(sentence)
            if text.strip() != "" and not os.path.exists(file_name):
                if (re.search('[A-Za-z0-9А-Яа-яЁё]', sentence)) is not None:
                    try:
                        model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)
                    except ValueError:
                        # audio_paths = model_multi.save_wav(text=sentence, speakers=multi_speakers, sample_rate=sample_rate) # TODO langs
                        print("ValueError EX!")
                        wrn_text = str(line_num) + ": " + text + " -> " + sentence
                        print(wrn_text)
                        wrn.append(wrn_text)
                        sentence = transliterate.translit(sentence, language_code="ru")
                        model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)
                    except Exception as e:
                        if str(e) == "Model couldn't generate your text, probably it's too long":
                            sentence = ''.join(filter(str.isalpha, sentence))
                            model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)

    out_dir_file = open(os.path.join(out_folder, "out_files.txt"), "w+", encoding="utf-8")
    for path, dirs, files in os.walk(out_folder):
        for file in files:
            if str(file).lower().endswith(".wav"):
                out_dir_file.write("file " + file + "\n")
                # TMP: 150ms delay (wav/32bit/float) TODO SSML break:
                out_dir_file.write("file " + os.path.join(root, "_silence_150.wav").replace("\\", "/") + "\n")
    out_dir_file.close()
    concat_cmd = "ffmpeg -f concat -safe 0 -i " + os.path.join(out_folder, "out_files.txt") + \
                 " -c copy -hide_banner -y " + os.path.join(out_folder, "final_output.wav")
    os.system(concat_cmd) # TODO w/o external ffmpeg


if __name__ == "__main__":
    parser.add_argument("-i", "--input", action="store", help="input file or folder (all txt files or chapters)", required=True)
    parser.add_argument("-o", "--output", action="store", help="relative output folder", default="result")
    parser.add_argument("-t", "--threads", action="store", help="thread count (torch.set_num_threads value)", default="4")
    parser.add_argument("-s", "--speaker", action="store", help="model speaker", default="xenia",
                        choices=["aidar", "baya", "kseniya", "xenia"])
    parser.add_argument("-d", "--device", action="store", help="torch.device value", default="cpu",
                        choices=["cpu", "cuda", "xpu", "opengl", "opencl", "ideep", "vulkan", "hpu"])
    parser.add_argument("-r", "--rate", action="store", help="sample rate", default="48000")
    args = parser.parse_args()
    cfg = vars(args)
    root = os.getcwd()
    print(cfg)

    # nltk.download("punkt") # if needed

    device = torch.device(cfg["device"])
    torch.set_num_threads(int(cfg["threads"]))
    sample_rate = int(cfg["rate"])

    local_file = "ru_v3.pt"
    speaker = cfg["speaker"]
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

    root_out_folder = cfg["output"]
    source = cfg["input"]

    if os.path.isdir(source):
        for path, dirs, files in os.walk(source):
            for file_name in files:
                out_file = str(file_name).replace(" ", "").replace(".txt", "")
                out_folder = root_out_folder + "/" + out_file
                open_file(os.path.join(path, file_name), out_folder)
                try:
                    os.replace(os.path.join(out_folder, "final_output.wav"), os.path.join(root_out_folder, out_file + ".wav"))
                except FileNotFoundError:
                    print(out_folder + ": file not merged or already moved")

    elif os.path.isfile(source):
        print(source)
        print(root_out_folder)
        open_file(source, root_out_folder)

    else:
        exit(1)

    print("\nWarnings: ")
    print(wrn)
    print("\nFinished!")
    exit(0)

