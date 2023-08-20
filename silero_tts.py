#!/usr/bin/env python

import os
import re
import argparse
import concurrent
import subprocess
import uuid

import torch # cuda: 1.13.1+cu117
import nltk
import transliterate
import num2words
import tqdm
import pypandoc
import ffmpeg # ffmpeg-python


###################################################
# repo: https://github.com/snakers4/silero-models #
# commit ce0756babc77ff3e4cd9aab1b871699e362325fc #
###################################################

wrn = []
parser = argparse.ArgumentParser(description="tts", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 epilog="Example: ./silero_tts.py -i folder/or/file.txt -o path/to/result -t 16 -s xenia -d cuda -r 48000 --merge")


def open_file(source, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file = open(source, "r", encoding="utf-8")
    lines = file.read().splitlines()
    file.close()
    print()
    print(out_folder)
    tts(lines, out_folder)


def experimental_svc(stream, export_folder): # UNSTABLE

    svc_path = os.path.join(os.path.abspath(os.getcwd()), "so-vits-svc-fork")
    svcg_path = os.path.join(svc_path, "venv", "Scripts", "")
    in_path = os.path.join(svc_path, "tmp", "in")
    out_path = os.path.join(svc_path, "tmp", "out")

    model_name = "ru-saya" # https://huggingface.co/fnx/so-vits-svc-4.0-ru-saya/tree/main
    checkpoint_name = "G_10000.pth"

    threshold = "-25" if speaker == "xenia" else "-20"
    svc_device = "cuda" if cfg["device"] == "cuda" else "cpu"

    os.makedirs(in_path,  exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    file_name = str(uuid.uuid4()) + ".wav"
    in_file_path = os.path.join(in_path, file_name)
    out_file_path = os.path.join(out_path, file_name)

    stream = ffmpeg.output(stream, in_file_path, c="copy", rf64="auto", loglevel="error") # pcm_s16le/768/48k/RF64
    ffmpeg.run(stream, overwrite_output=True)

    if os.path.getsize(in_file_path) > (4 * 1024 * 1024 * 1024): # >4GB
        # out_file_path = out_file_path.replace(".wav", ".flac")
        print("File is too large: " + in_file_path)
        exit(1)

    svcg_params = " infer " \
                  "-o " + out_file_path + " " \
                  "-s " + model_name + " " \
                  "-m " + os.path.join(svc_path, "models", model_name, checkpoint_name) + " " \
                  "-c " + os.path.join(svc_path, "models", model_name, "config.json") + " " \
                  "-t 3 " \
                  "-db " + threshold + " " \
                  "--no-auto-predict-f0 " \
                  "-d " + svc_device + " " \
                  "" + in_file_path

    subprocess.run(svcg_path + "svc " + svcg_params, stdout=subprocess.PIPE, shell=True)

    opus_stream = ffmpeg.input(out_file_path)
    opus_stream = ffmpeg.output(opus_stream, os.path.join(export_folder, "compressed_output.opus"), acodec="libopus", audio_bitrate="32k", loglevel="error")
    ffmpeg.run(opus_stream, overwrite_output=True)

    os.remove(in_file_path)
    os.remove(out_file_path)


def enc_merge(merge_object):
    for path, dirs, files in os.walk(merge_object["out_folder"]):
        opus_file_path = os.path.join(path, "compressed_output.opus")

        concat_file = open(os.path.join(path, "concat.txt"), "w+", encoding="utf-8")
        for file in sorted(files):
            if str(file).endswith(".wav"):
                # segments.append(ffmpeg.input(os.path.join(path, file))) # ERR 206
                concat_file.write("file " + file + "\n")
                concat_file.write("file " + os.path.join(root, "silence150.wav").replace("\\", "/") + "\n")
        concat_file.close()
        # stream = ffmpeg.concat(*segments, v=0, a=1)
        stream = ffmpeg.input(os.path.join(path, "concat.txt"), f="concat", safe=0)

        if use_svc:
            experimental_svc(stream, path)
        else:
            stream = ffmpeg.output(stream, opus_file_path, acodec="libopus", audio_bitrate="32k", loglevel="error")
            ffmpeg.run(stream, overwrite_output=True)
        os.replace(opus_file_path, os.path.join(root_out_folder, merge_object["out_file"] + ".opus"))


def enc_merge_exec(merge_objects):
    print()
    merge_threads = int(cfg["threads"])

    if not os.path.exists("silence150.wav"):
        silence_stream = ffmpeg.input("anullsrc=r=" + cfg["rate"] + ":cl=mono", t="0.15", f="lavfi")
        silence_stream = ffmpeg.output(silence_stream, "silence150.wav", loglevel="error")
        ffmpeg.run(silence_stream, overwrite_output=True)

    if use_svc:
        merge_threads = 1 if merge_threads < 4 else merge_threads // 4
    print("Merging and encoding, please wait... (" + str(merge_threads) + " threads)")
    pbar = tqdm.tqdm(total=len(merge_objects), desc="MERGING", unit="chapter")
    with concurrent.futures.ThreadPoolExecutor(max_workers=merge_threads) as executor:
        futures = [executor.submit(enc_merge, merge_object) for merge_object in merge_objects]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                pbar.update(1)
            except Exception as e:
                print("\n" + str(e))
                pbar.close()
                exit(1)


def open_fb2(source):
    print("converting file " + source + "...")
    md = pypandoc.convert_file(source, "md")

    print("extracting chapters...")
    chapters = []
    chapter_text = source
    for line in md.splitlines():
        if line.startswith("::: ") and "section" in line:
            chapters.append(chapter_text)
            chapter_text = ""
        else:
            # chapter_text += pypandoc.convert_text(line, "plain", format="md") # too slow
            chapter_text += line + "\n"
            if line == "":
                chapter_text += "\r\n"
    chapters.append(chapter_text)

    print()
    merge_objects = []
    for line_num, chapter in enumerate(chapters):
        chapter_name = str(chapter).partition("\n")[0]
        chapter_name = pypandoc.convert_text(chapter_name, "plain", format="md")
        chapter_name = chapter_name.replace(" ", "_").replace("\r", "").replace("\n", "")
        chapter_name = re.sub("[^A-Za-z0-9А-Яа-яЁё_!#%№-]+", "", chapter_name.strip())

        out_file = (str(line_num).zfill(6) + "_" + chapter_name).replace(".fb2", "")
        out_folder = os.path.join(root_out_folder, out_file).replace(" ",  "_")
        merge_object = {
            "out_file": out_file,
            "out_folder": out_folder,
        }
        merge_objects.append(merge_object)

        chapter = str(chapter).replace("\n", " ")
        chapter = re.sub("[^A-Za-z0-9А-Яа-яЁё_\s .,;!№$&?–—-]+", "", chapter.strip())

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        print()
        print(out_folder)
        tts(chapter.splitlines(), out_folder)
    if merge:
        enc_merge_exec(merge_objects)


def tts(lines, out_folder):
    for line_num, text in enumerate(tqdm.tqdm(lines, desc="    TTS")):
        text = str(text).replace("…", ",")
        text = str(text).replace("+", " плюс ")
        text = str(text).replace("%", " процент ")
        text = re.sub("[^A-Za-z0-9А-Яа-яЁё_\s .,;!№$&?–—-]+", "", text.strip())
        t_sentences = nltk.sent_tokenize(text)
        sentences = []
        for sentence in t_sentences:
            sentences += re.findall('.{1,800}', sentence)
        # print(sentences)
        for sentence_num, sentence in enumerate(sentences):
            file_name = os.path.join(out_folder,  "tts_" + str(line_num).zfill(6) + "_" + str(sentence_num).zfill(3) + ".wav")
            sentence = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0)), lang="ru"), sentence)
            # print(sentence)
            if text.strip() != "" and not os.path.exists(file_name):
                if (re.search('[A-Za-z0-9А-Яа-яЁё]', sentence)) is not None:
                    sentence = transliterate.translit(sentence, language_code="ru") # TODO langs
                    # ssml_text = "<speak><s>" + sentence + "</s><break time='150ms'/></speak>"
                    try:
                        model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)
                    except Exception as e:
                        if str(e) == "Model couldn't generate your text, probably it's too long":
                            print("Broken tokenization EX!")
                            wrn_text = out_folder + " (" + str(line_num) + ") " + text + " -> " + sentence
                            print(wrn_text)
                            wrn.append(wrn_text)
                            sentence = ''.join(filter(str.isalpha, sentence))
                            model.save_wav(text=sentence, speaker=speaker, sample_rate=sample_rate, audio_path=file_name)
                        else:
                            print(str(e))
                            exit(1)


if __name__ == "__main__":
    parser.add_argument("-i", "--input", action="store", help="input fb2 file or txt file(s) or folder with txt files (chapters)", required=True)
    parser.add_argument("-o", "--output", action="store", help="relative output folder", default="result")
    parser.add_argument("-t", "--threads", action="store", help="thread count (torch.set_num_threads value)", default="4")
    parser.add_argument("-s", "--speaker", action="store", help="model speaker", default="xenia",
                        choices=["aidar", "baya", "kseniya", "xenia", "eugene"])
    parser.add_argument("-d", "--device", action="store", help="torch.device value", default="cpu",
                        choices=["cpu", "cuda", "xpu", "opengl", "opencl", "ideep", "vulkan", "hpu"])
    parser.add_argument("-r", "--rate", action="store", help="sample rate", default="48000")
    parser.add_argument("-m", "--merge", action="store_true", help="[FFmpeg required] merge wav files and save as opus")
    parser.add_argument("-c", "--svc", action="store_true", help="[FFmpeg required] experimental, use voice conversion (so-vits-svc)")
    args = parser.parse_args()
    cfg = vars(args)
    root = os.getcwd()
    print(cfg)

    nltk.download("punkt") # if needed

    torch._C._jit_set_profiling_mode(False)

    device = torch.device(cfg["device"])
    torch.set_num_threads(int(cfg["threads"]))
    sample_rate = int(cfg["rate"])
    merge = cfg["merge"]
    use_svc = cfg["svc"]

    # v4 (v4_ru.pt) is trash: robotic voice, poor gpu(cuda) performance on 2.0.1+cu118, not working at all on 1.13.1
    local_file = "v3_1_ru.pt" # ru_v3.pt, v3_1_ru.pt
    speaker = cfg["speaker"]
    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/" + local_file, local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    root_out_folder = cfg["output"]
    source = cfg["input"]

    if os.path.isdir(source):
        merge_objects = []
        for path, dirs, files in os.walk(source):
            for file_name in sorted(files):
                out_file = str(file_name).replace(" ", "_").replace(".txt", "")
                out_file = re.sub("[^A-Za-z0-9А-Яа-яЁё_!#%№-]+", "", out_file.strip())
                out_folder = os.path.join(root_out_folder, out_file)
                open_file(os.path.join(path, file_name), out_folder)
                merge_object = {
                    "out_file": out_file,
                    "out_folder": out_folder,
                }
                merge_objects.append(merge_object)
        if merge:
            enc_merge_exec(merge_objects)

    elif os.path.isfile(source):
        if str(source).lower().endswith(".fb2"):
            open_fb2(source)
        else:
            print(source)
            print(root_out_folder)
            open_file(source, root_out_folder)
            if merge:
                merge_object = {
                    "out_file": "compressed_output.opus",
                    "out_folder": root_out_folder,
                }
                print("Merging and encoding, please wait...")
                enc_merge(merge_object)

    else:
        print("Error: input file or folder not found")
        exit(1)

    if len(wrn) > 0:
        print("\nWarnings: ")
        print(wrn)
    print("\nFinished!")
    exit(0)

