# Silero TTS
## ru txt/fb2/epub file or folder with txt/fb2/epub files to wav(s)/opus

### Usage:

```
usage: silero_tts.py [-h] -i INPUT [-o OUTPUT] [-t THREADS] [-s {aidar,baya,kseniya,xenia,eugene}]
                     [-d {cpu,cuda,xpu,opengl,opencl,ideep,vulkan,hpu}] [-r RATE] [-n] [--merge] [--rvc]
                     [--rvc_model_pth RVC_MODEL_PTH] [--rvc_model_index RVC_MODEL_INDEX]
                     [--rvc_transpose RVC_TRANSPOSE]

tts

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input txt/fb2/epub file or folder with txt/fb2/epub files (default: None)
  -o OUTPUT, --output OUTPUT
                        relative output folder (default: result)
  -t THREADS, --threads THREADS
                        thread count (torch.set_num_threads value) (default: 4)
  -s {aidar,baya,kseniya,xenia,eugene}, --speaker {aidar,baya,kseniya,xenia,eugene}
                        model speaker (default: xenia)
  -d {cpu,cuda,xpu,opengl,opencl,ideep,vulkan,hpu}, --device {cpu,cuda,xpu,opengl,opencl,ideep,vulkan,hpu}
                        torch.device value (default: cpu)
  -r RATE, --rate RATE  sample rate (default: 48000)
  -n, --ignore_newlines
                        use only for low-quality sources, such as books exported from PDF (default: False)
  --merge               [FFmpeg required] merge wav files and save as opus (default: False)
  --rvc                 [FFmpeg required] [cuda only] use voice conversion (Retrieval-based-Voice-Conversion v2.2)
                        (default: False)
  --rvc_model_pth RVC_MODEL_PTH
                        RVC model: .pth file name (required) (default: ru-saya-1000.pth)
  --rvc_model_index RVC_MODEL_INDEX
                        RVC model: .index file name (default: None)
  --rvc_transpose RVC_TRANSPOSE
                        RVC model: f0 transpose (default: 3)

Example: ./silero_tts.py -i folder/or/file.txt -o path/to/result -t 16 -s xenia -d cuda -r 48000 --merge
```