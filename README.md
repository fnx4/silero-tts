# Silero TTS
## ru txt file(s)/fb2 to wav(s)/opus

### Usage:

```
usage: silero_tts.py [-h] -i INPUT [-o OUTPUT] [-t THREADS] [-s {aidar,baya,kseniya,xenia,eugene}]
                     [-d {cpu,cuda,xpu,opengl,opencl,ideep,vulkan,hpu}] [-r RATE] [--merge] [--rvc]

tts

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input txt/fb2/epub file or folder with txt files (chapters) (default: None)
  -o OUTPUT, --output OUTPUT
                        relative output folder (default: result)
  -t THREADS, --threads THREADS
                        thread count (torch.set_num_threads value) (default: 4)
  -s {aidar,baya,kseniya,xenia,eugene}, --speaker {aidar,baya,kseniya,xenia,eugene}
                        model speaker (default: xenia)
  -d {cpu,cuda,xpu,opengl,opencl,ideep,vulkan,hpu}, --device {cpu,cuda,xpu,opengl,opencl,ideep,vulkan,hpu}
                        torch.device value (default: cpu)
  -r RATE, --rate RATE  sample rate (default: 48000)
  --merge               [FFmpeg required] merge wav files and save as opus (default: False)
  --rvc                 [FFmpeg required] [cuda only] use voice conversion (Retrieval-based-Voice-Conversion) (default: False)

Example: ./silero_tts.py -i folder/or/file.txt -o path/to/result -t 16 -s xenia -d cuda -r 48000 --merge
```