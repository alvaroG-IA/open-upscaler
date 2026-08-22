# 🚀 Open Upscaler

> **A lightweight, high-performance web application for AI-powered image super-resolution and restoration.**

Open Upscaler provides a simple **Gradio** interface for increasing image resolution using **Real-ESRGAN**. It automatically detects your hardware (**Apple Silicon MPS, NVIDIA CUDA, or CPU**) and uses tiled inference to process large images efficiently.

---

## ✨ Features

* **Intelligent Upscaling:** Support for **x2, x4, and x8** scaling.
* **Restoration Mode (x1):** Enhances images while keeping their original dimensions.
* **Hardware Acceleration:** Automatic support for **Apple MPS, NVIDIA CUDA, and CPU**.
* **Memory Management:** Adjustable tile size to balance performance and memory usage.
* **Model Caching:** Downloaded models are cached locally for faster subsequent inference.
* **Telemetry:** Built-in logging through `app.log`.
* **Remote Deployment:** Optional public access through **ngrok**.

---

## 💡 Motivation

This project was born out of a single afternoon's frustration. I simply needed to upscale a low-resolution image, but every online tool I found was hidden behind forced account registrations, aggressive paywalls, or expensive subscriptions for a basic AI inference task. 

As an AI engineer, I decided to look under the hood to see what kind of neural networks were actually powering these "premium" services—which ultimately led to the creation of this repository. I built **Open Upscaler** to provide a completely free, local, and transparent alternative. I hope this tool helps you bypass those predatory websites as much as it helps me! 🙂

---

## 🛠 Tech Stack

**Core:** Python 3.10+ | PyTorch | Hugging Face Hub

**Processing:** Real-ESRGAN | Pillow | NumPy

**UI & Network:** Gradio | ngrok

---

## 📂 Project Structure

```text
open-upscaler/
│
├── app.py                 # Gradio UI entry point
├── backend.py             # Inference logic, hardware detection and telemetry
├── requirements.txt       # Python dependencies
├── example_start.sh       # Example ngrok deployment script
│
├── src/
│   ├── __init__.py
│   ├── model.py           # Real-ESRGAN wrapper and caching logic
│   └── model_arch.py      # RRDBNet architecture definitions
│
├── weights/               # Local cache for downloaded model weights
├── app.log                # Auto-generated application logs
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/alvaroG-IA/open-upscaler.git
cd open-upscaler
```

### 2. Create a virtual environment

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Model weights (`RealESRGAN_x*.pth`) are not included in the repository. The application automatically downloads the required models from Hugging Face on first use and caches them in the `weights/` directory.

---

## 🚀 Usage

### Option A: Local

Start the application:

**macOS / Linux:**

```bash
source .venv/bin/activate
python app.py
```

**Windows:**

```powershell
.venv\Scripts\activate
python app.py
```

Then open:

```text
http://localhost:8888
```

---

### Option B: Remote Access with ngrok

You can expose Open Upscaler to the internet using **ngrok**.

#### 1. Create an ngrok account

If you don't already have one, create a free account at [ngrok](https://ngrok.com/?utm_source=chatgpt.com)

An ngrok account and authentication token are required to connect the local agent to your account.

#### 2. Install ngrok

Download and install the ngrok Agent for your operating system [ngrok downloads](https://ngrok.com/download?utm_source=chatgpt.com)

For Windows, you can also use the official Windows installation options [ngrok for Windows](https://ngrok.com/download/windows?utm_source=chatgpt.com)

Verify the installation:

```bash
ngrok help
```

#### 3. Connect ngrok to your account

Copy your **Authtoken** from the ngrok dashboard and run:

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

This stores the token in your local ngrok configuration and links the agent to your account.

#### 4. Get a public domain

For a simple setup, ngrok provides a random URL each time. However, ngrok now includes **one free static domain** (e.g., `word-word-word.ngrok-free.dev`) on their free tier.

You can find and claim your permanent domain in the **Domains** section of your ngrok dashboard.

#### 5. Start Open Upscaler

First, start the application normally:

```bash
python app.py
```

Then, in another terminal:

```bash
ngrok http 8888
```

ngrok will display a public URL that forwards traffic to your local application.

If you have claimed your free static domain:

```bash
ngrok http --domain=your-domain.ngrok-free.app 8888
```

#### Optional: Use the provided script

You can also use `example_start.sh` to start both the Gradio application and the ngrok tunnel:

```bash
chmod +x example_start.sh
./example_start.sh
```

On Windows, you can run the two commands manually from separate terminals instead.

---

## 🧠 How It Works

When an image is uploaded, Open Upscaler follows this pipeline:

1. **Hardware Detection:** Selects MPS, CUDA, or CPU automatically.
2. **Model Loading:** Downloads and caches the required Real-ESRGAN model.
3. **Tiled Inference:** Splits large images into tiles to reduce memory usage.
4. **Processing:** Performs the selected x2, x4, x8, or restoration operation.
5. **Memory Cleanup:** Releases unused GPU memory after inference.

---

## 🙏 Acknowledgements

Open Upscaler is based on the **Real-ESRGAN** implementation and model resources originally provided by [AI-Forever](https://github.com/ai-forever/Real-ESRGAN).

The project uses model weights and network architecture definitions derived from their repository. Since the original repository has been **deprecated**, its implementation was adapted to work with newer Python and dependency versions while preserving the original model architecture and inference logic.

Many thanks to the **AI-Forever team** and the open-source community for making these resources available.

---

## 🔮 Future Work (Roadmap)

I'm actively exploring new features to evolve Open Upscaler from a pure resolution-enhancer into a comprehensive AI photo restoration suite:

- [ ] **Historical Photo Restoration Pipeline:** Introducing a dedicated flow for vintage photos, chaining scratch/dust removal models (inpainting) and colorization before the final upscale.
- [ ] **Facial Reconstruction (Facial Priors):** Integrating models like **GFPGAN** or **CodeFormer** to accurately rebuild heavily degraded faces and prevent the "uncanny valley" effect common in standard upscalers.
- [ ] **Batch Processing:** Adding UI support for bulk-uploading multiple images and exporting the processed results as a compressed `.zip` archive.
- [ ] **Multi-Tab Interface:** Evolving the Gradio UI to separate "Modern Upscaling" and "Historical Restoration" into distinct, specialized workspaces.

---

## 📬 Contact

For questions, suggestions, or bug reports, feel free to open an **issue** or start a **discussion** in this repository.

You can also find all my contact information on GitHub: [@alvaroG-IA](https://github.com/alvaroG-IA).


---

## 📜 License

See [`LICENSE`](LICENSE) for details.
