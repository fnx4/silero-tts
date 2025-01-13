# Fish TTS
## ru txt/fb2/epub file or folder with txt/fb2/epub files to wav(s)/opus

### Usage:

```
usage: fish_tts.py [-h] -i INPUT [-o OUTPUT] [-t THREADS] [-r RATE] [--merge] [--seed SEED]
                   [--reference REFERENCE]

tts

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input txt/fb2/epub file or folder with txt/fb2/epub files (default: None)
  -o OUTPUT, --output OUTPUT
                        relative output folder (default: result)
  -t THREADS, --threads THREADS
                        thread count (torch.set_num_threads value) (default: 4)
  -r RATE, --rate RATE  sample rate (default: 48000)
  --merge               [FFmpeg required] merge wav files and save as opus (default: False)
  --seed SEED           fish seed (default: 205)
  --reference REFERENCE
                        fish reference_id (default: 4_saya)

Example: ./fish_tts.py -i folder/or/file.txt -o path/to/result -r 48000 --merge
```