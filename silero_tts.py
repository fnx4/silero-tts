#!/usr/bin/env python3.12

import os
import re
import argparse
import subprocess
from sys import exit
import concurrent

import torch # cuda: 2.8.0+cu129
import nltk
import transliterate
import num2words
from tqdm.auto import tqdm
import pypandoc
import ffmpeg # ffmpeg-python


###################################################
# repo: https://github.com/snakers4/silero-models #
# commit ce0756babc77ff3e4cd9aab1b871699e362325fc #
###################################################

REGEXP_NAME = r"[^A-Za-z0-9А-Яа-яЁё_-]+"
REGEXP_TEXT = r"[^A-Za-z0-9А-Яа-яЁё_/\s .,;!№$%&?+–—-]+"

RVC_VRAM_LIMIT = 12 # RMVPE: >12gb (~40+ min. chunks) may cause unexpected behavior (RVCv2 large input file bug: incorrect sequence of segments)

wrn = []
parser = argparse.ArgumentParser(description="tts", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 epilog="Example: ./silero_tts.py -i folder/or/file.txt -o path/to/result -t 16 -s xenia -d cuda -r 48000 --merge")


class Cfg:
    def __init__(self, param=None):
        self.input = param["input"]
        self.output = param["output"]
        self.threads = param["threads"]
        self.speaker = param["speaker"]
        self.device = param["device"]
        self.rate = param["rate"]
        self.merge = param["merge"]
        self.rvc = param["rvc"]
        self.rvc_model_pth = param["rvc_model_pth"]
        self.rvc_model_index = param["rvc_model_index"]
        self.rvc_transpose = param["rvc_transpose"]


class MergeParameters:
    volume_name = None
    out_file_name = None
    out_file_path = None

    def __init__(self, volume_name, out_file_name, out_file_path):
        self.volume_name = volume_name
        self.out_file_name = out_file_name
        self.out_file_path = out_file_path

    def __repr__(self):
        return "\n\t" + str(self.volume_name) + " -> " + str(self.out_file_name) + ": " + str(self.out_file_path)


class ConcatenatedFilesParameters:
    root_path: None
    wav_dir_path: None
    file_base_name: None
    wav_file_path: None
    volume: None

    def __init__(self, root_path, wav_dir_path, file_base_name, wav_file_path, volume):
        self.root_path = root_path
        self.wav_dir_path = wav_dir_path
        self.file_base_name = file_base_name
        self.wav_file_path = wav_file_path
        self.volume = volume

    def __repr__(self):
        return "\n\t" + str(self.root_path) + "; " + str(self.wav_dir_path) + "; " + str(self.file_base_name) + \
               "; " + str(self.wav_file_path) + "; " + str(self.volume)


def main(args):
    print()
    print(args)
    cfg = Cfg(vars(args))

    nltk.download("punkt")

    torch._C._jit_set_profiling_mode(False)

    device = torch.device(cfg.device)
    torch.set_num_threads(int(cfg.threads))

    # v4 (v4_ru.pt) is trash: robotic voice, poor gpu(cuda) performance on 2.0.1+cu118, not working at all on 1.13.1
    local_file = "v3_1_ru.pt" # ru_v3.pt, v3_1_ru.pt
    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/" + local_file, local_file)
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    merge_objects = []
    if os.path.exists(cfg.input):
        if os.path.isdir(cfg.input):
            input_directory_handle(cfg, model, merge_objects)
        elif os.path.isfile(cfg.input):
            read_file(cfg, model, merge_objects, cfg.input, cfg.output)
        if cfg.merge:
            encode(cfg, merge_objects)
    else:
        error("Error: input file or folder not found")
        exit(1)
    print("\nFinished!")
    exit(0)


def input_directory_handle(cfg: Cfg, model, merge_objects):
    for path, dirs, files in os.walk(cfg.input):
        for file_name in sorted(files):
            out_file_name = re.sub(REGEXP_NAME, "", str(file_name).replace(" ", "_")
                                   .replace(".txt", "").replace(".fb2", "").replace(".epub", "").strip())
            in_file_path = os.path.join(path, file_name)
            out_file_path = os.path.join(cfg.output, out_file_name)

            read_file(cfg, model, merge_objects, in_file_path, out_file_path)


def read_file(cfg: Cfg, model, merge_objects, in_file_path, out_file_path):
    print()
    print("file: " + in_file_path + " ...")

    md = None
    is_ebook = True
    if in_file_path.lower().endswith(".txt"):
        file = open(in_file_path, "r", encoding="utf-8")
        md = file.read()
        file.close()
        is_ebook = False
    elif in_file_path.lower().endswith(".epub"):
        fb2 = pypandoc.convert_file(in_file_path, "fb2")
        md = pypandoc.convert_text(fb2, "md", format="fb2")
    elif in_file_path.lower().endswith(".fb2"):
        md = pypandoc.convert_file(in_file_path, "md")
    else:
        error("Error: unknown file extension")
        exit(1)

    chapters = []
    chapter_text = in_file_path
    for line in md.splitlines():
        if line.startswith("::: ") and "section" in line:
            chapters.append(chapter_text)
            chapter_text = ""
        else:
            # chapter_text += pypandoc.convert_text(line, "plain", format="md") # too slow
            chapter_text += line + "\n"
            if line == "":
                chapter_text += "\r\n"
            if cfg.rvc and len(chapter_text.encode("utf-8")) > RVC_VRAM_LIMIT * 6 * 1000:
                chapters.append(chapter_text)
                chapter_text = " \r\n"
    chapters.append(chapter_text)

    for line_num, chapter in enumerate(tqdm(chapters, position=0, desc=" CHAPTER")):
        chapter_name = str(chapter).partition("\n")[0] if is_ebook else os.path.basename(in_file_path)
        chapter_name = pypandoc.convert_text(chapter_name, "plain", format="md")
        chapter_name = chapter_name.replace(" ", "_").replace("\r", "").replace("\n", "")
        chapter_name = re.sub(REGEXP_NAME, "", chapter_name.strip())
        chapter_name = re.findall('.{1,50}', chapter_name)[0] if chapter_name != "" else "_"

        line_num_str = str(line_num).zfill(6)
        out_file_name = (line_num_str + "_" + chapter_name) if is_ebook else (chapter_name + "_" + line_num_str)
        out_file_name = out_file_name.replace(".fb2", "").replace(".epub", "").replace("txt", "_")
        out_chapter_path = os.path.join(out_file_path, out_file_name).replace(" ", "_")

        chapter = str(chapter).replace("\n", " ")
        chapter = re.sub(REGEXP_TEXT, "", chapter.strip())

        if not os.path.exists(out_chapter_path):
            os.makedirs(out_chapter_path)
        if is_ebook:
            volume_name = os.path.basename(out_file_path)
        else:
            volume_name = ""
        tts(cfg, model, chapter.splitlines(), out_chapter_path)

        merge_objects.append(MergeParameters(
            volume_name=volume_name,
            out_file_name=out_file_name,
            out_file_path=out_chapter_path)
        )


def tts(cfg: Cfg, model, lines, out_file_path):
    for line_num, text in enumerate(tqdm(lines, leave=False, position=1, desc="    TEXT")):
        text = str(text).replace("…", ",")
        text = str(text).replace("+", " плюс ")
        text = str(text).replace("%", " процент ")
        text = str(text).replace("/", " из ")
        text = re.sub(r"(?<=\d)[ ,']+(?=\d{3})", "", text.strip())
        text = re.sub(r"(?<=\d)\.(?=\d)", " целых запятая ", text)
        text = re.sub(REGEXP_TEXT, "", text)
        t_sentences = nltk.sent_tokenize(text)
        sentences = []
        for sentence in t_sentences:
            sentences += re.findall('.{1,800}', sentence)
        # print(sentences)
        for sentence_num, sentence in enumerate(tqdm(sentences, leave=False, position=2, desc="SENTENCE")):
            file_name = os.path.join(out_file_path, "tts_" + str(line_num).zfill(6) + "_" + str(sentence_num).zfill(3) + ".wav")
            sentence = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0)), lang="ru"), sentence)
            # print(sentence)
            if text.strip() != "" and not os.path.exists(file_name):
                if (re.search('[A-Za-z0-9А-Яа-яЁё]', sentence)) is not None:
                    sentence = transliterate.translit(sentence, language_code="ru")
                    try:
                        model.save_wav(text=sentence, speaker=cfg.speaker, sample_rate=int(cfg.rate), audio_path=file_name)
                    except Exception as e:
                        if str(e) == "Model couldn't generate your text, probably it's too long":
                            print("Broken tokenization EX!")
                            wrn_text = 'TTS E1: ' + out_file_path + " (" + str(line_num) + ") " + text + " -> " + sentence
                            print(wrn_text)
                            wrn.append(wrn_text)
                            sentence = ''.join(filter(str.isalpha, sentence))
                            model.save_wav(text=sentence, speaker=cfg.speaker, sample_rate=int(cfg.rate), audio_path=file_name)
                        elif type(e).__name__ == "ValueError":
                            print('Unable to process text: ' + sentence)
                            wrn_text = 'TTS E2: ' + out_file_path + " (" + str(line_num) + ") " + text + " -> " + sentence
                            print(wrn_text)
                            wrn.append(wrn_text)
                        else:
                            print('Exception  ' + str(e))
                            exit(1)


def encode(cfg: Cfg, merge_objects):
    print()
    print("Merging and encoding, please wait...")

    if not os.path.exists("silence150.wav"):
        silence_stream = ffmpeg.input("anullsrc=r=" + cfg.rate + ":cl=mono", t="0.15", f="lavfi")
        silence_stream = ffmpeg.output(silence_stream, "silence150.wav", loglevel="error")
        ffmpeg.run(silence_stream, overwrite_output=True)

    concatenated_wav_files = []

    for merge_object in merge_objects:
        for root, dirs, files in os.walk(merge_object.out_file_path):
            # print(root)
            concat_file_path = os.path.join(root, "concat.txt")
            concat_file = open(concat_file_path, "w+", encoding="utf-8")
            for file in sorted(files):
                if str(file).lower().startswith("tts"):
                    # print(file)
                    concat_file.write("file " + os.path.abspath(os.path.join(root, file)).replace("\\", "/") + "\n")
                    concat_file.write("file " + os.path.join(os.getcwd(), "silence150.wav").replace("\\", "/") + "\n")
            concat_file.close()

            wav_volume_folder_path = os.path.join(cfg.output, "_wav", merge_object.volume_name)
            os.makedirs(wav_volume_folder_path, exist_ok=True)
            wav_out_path = os.path.join(wav_volume_folder_path, merge_object.out_file_name + ".wav")
            stream_wav = ffmpeg.input(concat_file_path, f="concat", safe=0)
            stream_wav = ffmpeg.output(stream_wav, wav_out_path, c="copy", rf64="auto", loglevel="error")  # pcm_s16le/768/48k/RF64
            ffmpeg.run(stream_wav, overwrite_output=True)

            concatenated_wav_files.append(ConcatenatedFilesParameters(
                root_path=cfg.output,
                wav_dir_path=wav_volume_folder_path,
                file_base_name=merge_object.out_file_name,
                wav_file_path=wav_out_path,
                volume=merge_object.volume_name)
            )

    # print(concatenated_wav_files)
    streams = []
    if cfg.rvc:
        if cfg.device != "cuda":
            error("Error: device " + cfg.device + " is not supported")
            exit(1)

        rvc_path = os.path.join(os.path.abspath(os.getcwd()), "rvc")
        vc_python_bin_win = os.path.join(rvc_path, "venv", "Scripts")
        vc_python_bin_nix = os.path.join(rvc_path, "venv", "bin")
        if os.path.exists(vc_python_bin_win.strip()):
            vc_python_bin = os.path.join(vc_python_bin_win, "python")
        elif os.path.exists(vc_python_bin_nix.strip()):
            vc_python_bin = os.path.join(vc_python_bin_nix, "python")
        else:
            vc_python_bin = "python"

        model = " --model_name " + cfg.rvc_model_pth # ./<rvc_path>/assets/weights
        if cfg.rvc_model_index is not None:
            model += " --index_path " + cfg.rvc_model_index

        for file in tqdm(concatenated_wav_files, desc=" RVC"):
            if os.path.getsize(file.wav_file_path) > (RVC_VRAM_LIMIT * 20 * 1024 * 1024):  # ~6KB of text (~20MB of wav) for each GB of VRAM
                wrn_text = "WARNING: File is too large, high VRAM usage: " + file.wav_out_path
                wrn.append(wrn_text)
                print(wrn_text)
                # exit(1)

            rvc_out_path = os.path.join(file.root_path, "_wav_rvc", file.volume)
            os.makedirs(rvc_out_path, exist_ok=True)
            rvc_out_file_path = os.path.join(rvc_out_path, file.file_base_name + ".wav")
            rvc_params = model + \
                         " --f0up_key " + cfg.rvc_transpose + \
                         " --input_path " + os.path.join(os.getcwd(), file.wav_file_path) + \
                         " --opt_path " + os.path.join(os.getcwd(), rvc_out_file_path) + \
                         " --device cuda:0" + \
                         " --f0method rmvpe" + \
                         " --filter_radius 0" + \
                         " --index_rate 0" + \
                         " --rms_mix_rate 0.4"
            proc = vc_python_bin + " " + os.path.join(rvc_path, "tools", "infer_cli.py") + " " + rvc_params
            process = subprocess.run(proc, cwd=rvc_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

            if not os.path.exists(rvc_out_file_path) or os.path.getsize(rvc_out_file_path) == 0:
                print()
                print("RVC output file not found: ")
                error(process.stdout.decode("utf-8"))
                exit(1)

            opus_out_path = os.path.join(file.root_path, "_opus", file.volume)
            opus_out_file_path = os.path.join(opus_out_path, file.file_base_name + ".opus")
            os.makedirs(opus_out_path, exist_ok=True)

            stream_opus = ffmpeg.input(rvc_out_file_path)
            stream_opus = ffmpeg.output(stream_opus, opus_out_file_path, acodec="libopus", audio_bitrate="32k", loglevel="error")
            streams.append(stream_opus)
    else:
        for file in tqdm(concatenated_wav_files):
            opus_volume_folder_path = os.path.join(file.root_path, "_opus", file.volume)
            os.makedirs(opus_volume_folder_path, exist_ok=True)
            opus_out_path = os.path.join(opus_volume_folder_path, file.file_base_name + ".opus")
            stream_opus = ffmpeg.input(file.wav_file_path)
            stream_opus = ffmpeg.output(stream_opus, opus_out_path, acodec="libopus", audio_bitrate="32k", loglevel="error")
            streams.append(stream_opus)

    pbar = tqdm(total=len(streams), desc="OPUS")
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cfg.threads)) as executor:
        futures = [executor.submit(ffmpeg_run, stream) for stream in streams]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                pbar.update(1)
            except Exception as e:
                error("\n" + str(e))
                pbar.close()
                exit(1)


def error(err):
    print("\033[91m" + err + "\033[0m")


def ffmpeg_run(stream):
    ffmpeg.run(stream, overwrite_output=True)


if __name__ == "__main__":
    parser.add_argument("-i", "--input", action="store", help="input txt/fb2/epub file or folder with txt/fb2/epub files", required=True)
    parser.add_argument("-o", "--output", action="store", help="relative output folder", default="result")
    parser.add_argument("-t", "--threads", action="store", help="thread count (torch.set_num_threads value)", default="4")
    parser.add_argument("-s", "--speaker", action="store", help="model speaker", default="xenia",
                        choices=["aidar", "baya", "kseniya", "xenia", "eugene"])
    parser.add_argument("-d", "--device", action="store", help="torch.device value", default="cpu",
                        choices=["cpu", "cuda", "xpu", "opengl", "opencl", "ideep", "vulkan", "hpu"])
    parser.add_argument("-r", "--rate", action="store", help="sample rate", default="48000")
    parser.add_argument("--merge", action="store_true", help="[FFmpeg required] merge wav files and save as opus")
    parser.add_argument("--rvc", action="store_true", help="[FFmpeg required] [cuda only] use voice conversion (Retrieval-based-Voice-Conversion v2.2)")
    parser.add_argument("--rvc_model_pth", action="store", help="RVC model: .pth file name (required)", default="ru-saya-1000.pth")
    parser.add_argument("--rvc_model_index", action="store", help="RVC model: .index file name")
    parser.add_argument("--rvc_transpose", action="store", help="RVC model: f0 transpose", default="3")
    # TODO RVC vram target/limit
    # TODO print debug logs

    main(parser.parse_args())
