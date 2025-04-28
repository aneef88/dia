<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/yBrqQ9Dd" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Dataset on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- (Update) We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/yBrqQ9Dd) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## ⚡️ Quickstart

### Install via pip

```bash
# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git
```

### Run the Gradio UI

This will open a Gradio UI that you can work on.

```bash
git clone https://github.com/nari-labs/dia.git
cd dia && uv run app.py
```

or if you do not have `uv` pre-installed:

```base
git clone https://github.com/nari-labs/dia.git
cd dia
python -m venv .venv
source .venv/bin/activate
pip install -e .
python app.py
```

Note that the model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
You can keep speaker consistency by either adding an audio prompt (a guide coming VERY soon - try it with the second example on Gradio for now), or fixing the seed.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.

## ⚙️ Usage

### As a Python Library

```python
from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(text, use_torch_compile=True, verbose=True)

model.save_audio("simple.mp3", output)
```

A pypi package and a working CLI tool will be available soon.

## 🚀 FastAPI Server for Local Use

This repository also includes a simple FastAPI server providing a local API endpoint for text-to-speech (TTS) generation using the Dia model. This server is designed primarily for local use, allowing other applications on the same machine or local network to easily generate audio via HTTP requests. It utilizes a server-side prompt management system for voice selection.

**Server Features:**
- Provides a /tts endpoint for audio generation.
- Uses FastAPI and Uvicorn for the web server.
- Loads the specified Nari Dia model (PyTorch).
- Manages voice prompts server-side via a prompts directory.
- Supports overriding default generation parameters (temperature, seed, speed, etc.) via the API request.
- Packaged as an installable Python package (nari-tts) with a console script to run the server.

### Server Prerequisites
- Python 3.10+
- pip or a compatible package manager (like uv)
- PyTorch installed (compatible with your system - CPU or CUDA GPU recommended). See PyTorch installation guide.
- A CUDA-enabled GPU is highly recommended for reasonable generation speed. CPU generation will be very slow.

### Server Installation
**(Requires cloning the repository first if not done already)**
1. Navigate to the repository root directory.
2. Create and activate a virtual environment (recommended):
```
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```
3. Install the package and dependencies:
   This will install the server code and all dependencies listed in `pyproject.toml`, including the `dia` library, FastAPI, and Uvicorn.
```
pip install .
# Or for development (editable install):
# pip install -e .
```
This also installs the `nari-tts` console script.

### Server Configuration
1. Prompt Setup (Server-Side)

- **Create** a `prompts` **Directory:** By default, the server looks for a directory named `prompts` in the location where you run the `nari-tts` command. You can change this using the `--prompts-dir` argument or `NARI_PROMPTS_DIR` environment variable.
- **Add Prompt Fields:** Inside the `prompts` directory, add pairs of files for each desired voice prompt:
  - `<promp_id_>.wav`: The audio file for the prompt. MUST be a MONO WAV file. The model expects mono input.
  - `<prompt_id_>.txt`: A plain text file containing the exact transcript of the corresponding `.wav` file, including any speaker tags (eg. `[S1]`) used by the model.
- `<prompt_id>`: The base filename (without the extension) is used as the `prompt_id` in API requests to select that voice.

2. Server Options (Environment Variables / CLI Arguments)
You can configure the server's behavior using command-line arguments or environment variables. CLI arguments take precedence over environment variables, which take precedence over defaults.

| Feature | CLI Argument | Enviornment Variable | Default Value | Description |
|:-:|:-:|:-:|:-:|:-:|
| Host Address | `--host` | `NARI_HOST` | `0.0.0.0` | Network address to bind the server. |
| Port Number | `--port` | `NARI_PORT` | `8210` | Port number for the server. | 
| Prompts Directory | `--prompts-dir` | `NARI_PROMPTS_DIR` | `prompts` | Path to the directory containing prompt `.wav / .txt` files. | 
| Default Prompt ID | `--default-prompt` | `NARI_DEFAULT_PROMPT` | `default_voice` | `prompt_id` used if none is specified in the request. Must match a file pair. |
| Model Name/Path | `--model-name` | `NARI_MODEL_NAME` | `nari-labs/Dia-1.6B` | Hugging Face model name or local path to load. |
| Enable Torch Compile | `--use-torch-compile / --no-use-torch-comnpile` | `NARI_USE_COMPILE (true/false)` | `true` | Whether to use `torch.compile` for potential speedup. |
| Development Reload | `--reload` | (N/A) | `False` | Enable Uvicorn's audo-reload feature for development |

### Running the Server
Once installed and configured (ensure your prompts directory is set up), run the server using the console script:
```
nari-tts [OPTIONS]
```

**Examples**
- Run with defaults:
```
nari-tts
```
- Run on a different port and specify a custom prompts directory:
```
nari-tts ---port 9000 ---prompts-dir /path/to/my/audio/prompts
```
- Run without Torch Compile:
```
nari-tts --no-use-torch-compile
```
- Run using enviornment variables (Linux/macOS example):
```
export NARI_DEFAULT_PROMPT="example_prompt"
export NARI_PROMPTS_DIR="./my_prompts"
nari-tts --port 8080
```

The server will start, load the model, and listen for requests on the configured host and port. Watch the console logs for status information and potential errors.

### API Endpoint: `/tts`
This is the primary endpoint for generating speech.
- URL: `/tts` (relative to the server's base URL, e.g., http://localhost:8210/tts)
- Method:`POST`
- Request Format: `application/json`

#### Request Body
The request body should be a JSON object containing the following fields:
| `Field` | Type | Required | Default | Description |
|:-:|:-:|:-:|:-:|:-:|
| `text` | `string` | Yes | (N/A) | The text content to be synthesized into speech. |
| `prompt_id` | `string` | No | Server Default* | The identifier (base filename) of the server-side prompt (.wav/.txt) to use for the voice. |
| `max_new_tokens` | `integer` | No | 3072 | Maximum number of new audio tokens to generate. Must be > 0. |
| `cfg_scale` | `number` | No | `3.0` | Classifier-Free Guidance scale. Higher = stricter adherence to text/prompt. Must be >= 1.0. |
| `temperature` | `number` | No | `1.3` | Sampling temperature. Controls randomness (>0.0). Lower values = more deterministic. |
| `top_p` | `number` | No | `0.95` | Nucleus sampling threshold (0.0 < top_p <= 1.0). |
| `cfg_filter_top_k` | `integer` | No | `30` | Filters guidance logits to the top K values during CFG. Must be >= 1. |
| `speed_factor` | `number` | No | `1.0` | Audio speed multiplier (>0.1). <1.0 is slower, >1.0 is faster. Note: Nari default was 0.94 |
| `seed` | `integer` | No | `null` (Random) | Random seed for generating reproducible audio. If null or omitted, output will be random. |

*If `prompt_id` is null or omitted, the server uses the prompt ID configured via `--default-prompt` or `NARI_DEFAULT_PROMPT`. If that is also not configured or the files don't exist, generation proceeds without a voice prompt (which may result in poor quality or silence).

#### Success Response
- Status Code: `200 OK`
- Content-Type: `audio/wav`
- Headers: Includes `Content-Length` indicating the size of the audio data in bytes.
- Body: The raw binary data of the generated WAV audio file.

#### Error Responses
The server returns standard HTTP error codes with a JSON body containing a detail field explaining the error.
- `400 Bad Request:` Invalid input provided in the request body (e.g., `text` is empty, a numeric parameter is out of range). The `detail` message specifies the issue.
- `404 Not Found:` The requested `prompt_id` does not correspond to existing `.wav` and `.txt` files in the server's `prompts` directory.
- `500 Internal Server Error:` An unexpected error occurred during model generation, audio processing, or WAV encoding on the server. Check server logs for detailed tracebacks.
- `503 Service Unavailable:` The TTS model failed to load during server startup or is otherwise unavailable. Check server startup logs.

#### Example API Usage
```
curl -X POST "http://localhost:8210/tts" \
     -H "Content-Type: application/json" \
     -H "Accept: audio/wav" \
     -d '{
           "text": "[S1] This is a test using curl.",
           "prompt_id": "example_prompt",
           "seed": 42,
           "temperature": 0.9
         }' \
     --output curl_output.wav
```

(This saves the generated audio to curl_output.wav)

```
import requests
import json

SERVER_URL = "http://localhost:8210/tts" # Replace with your server's URL if different
OUTPUT_FILE = "python_output.wav"

payload = {
    "text": "[S1] Testing audio generation from Python.",
    "prompt_id": "example_prompt", # Or None to use server default
    "seed": 123,
    "speed_factor": 1.0
    # Add other parameters as needed
}

print(f"Sending request to {SERVER_URL}")
try:
    response = requests.post(
        SERVER_URL,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        json=payload,
        stream=True,
        timeout=180 # Set a reasonable timeout (seconds)
    )

    if response.status_code == 200:
        print(f"Success! Saving audio to {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Error: Server returned status code {response.status_code}")
        try:
            error_details = response.json()
            print(f"Error details: {error_details}")
        except json.JSONDecodeError:
            print(f"Error details (raw): {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

## 💻 Hardware and Inference Speed

Dia has been tested on only GPUs (pytorch 2.0+, CUDA 12.6). CPU support is to be added soon.
The initial run will take longer as the Descript Audio Codec also needs to be downloaded.

These are the speed we benchmarked in RTX 4090.

| precision | realtime factor w/ compile | realtime factor w/o compile | VRAM |
|:-:|:-:|:-:|:-:|
| `bfloat16` | x2.1 | x1.5 | ~10GB |
| `float16` | x2.2 | x1.3 | ~10GB |
| `float32` | x1 | x0.9 | ~13GB |

We will be adding a quantized version in the future.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Docker support for ARM architecture and MacOS.
- Optimize inference speed.
- Add quantization for memory efficiency.

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/yBrqQ9Dd) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>