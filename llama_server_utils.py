"""
Starts/stops a local llama.cpp OpenAI-compatible server from inside the notebook,
the same way the reference PEARC tutorial's ollama_utils.py drives Ollama --
just with llama-cpp-python instead, since that's a pure-pip CPU install with
no system binary or install script needed (Colab-friendly).
"""
import atexit
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
HOST = "127.0.0.1"
PORT = 8000

SERVER_PROCESS = None


def download_model(local_dir="models"):
    """Downloads the GGUF model (~1GB, Apache-2.0, no login/gating required)."""
    Path(local_dir).mkdir(exist_ok=True)
    print(f"Downloading {MODEL_REPO}/{MODEL_FILE} (only happens once, ~1GB)...")
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=local_dir)
    print(f"Model ready at {path}")
    return path


def gpu_available() -> bool:
    """
    True if `nvidia-smi` runs successfully, i.e. an NVIDIA GPU + driver is present.

    Note this only tells you a GPU *exists* on the machine -- it does NOT mean
    llama-cpp-python can actually use it. That also requires having installed a
    CUDA-enabled wheel (e.g. `--extra-index-url
    https://abetlen.github.io/llama-cpp-python/whl/cu121`, matched to your CUDA
    version) instead of this repo's default CPU-only wheel. Passing
    n_gpu_layers=-1 to start_llama_server() on the CPU-only wheel is harmless
    but does nothing -- there's no GPU code compiled in to use.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def start_llama_server(model_path=None, n_ctx=4096, n_gpu_layers=0):
    """Starts `python -m llama_cpp.server` in the background and waits until it responds.

    n_gpu_layers: how many model layers to offload to GPU. -1 = all layers,
        0 = CPU-only (the default, matching this repo's default CPU-only wheel
        install so it keeps working out of the box on a plain Colab CPU
        runtime). If you've installed a CUDA-enabled llama-cpp-python wheel and
        have a GPU, pass -1 explicitly -- or use the gpu_available() helper
        above to decide at call time, e.g.:
            start_llama_server(model_path, n_gpu_layers=-1 if gpu_available() else 0)
    """
    global SERVER_PROCESS

    if model_path is None:
        model_path = download_model()

    subprocess.run(["pkill", "-f", "llama_cpp.server"], capture_output=True)
    time.sleep(1)

    log_file = "llama_server.log"
    print("Starting llama.cpp server...")
    print(f"n_gpu_layers={n_gpu_layers} ({'GPU offload' if n_gpu_layers != 0 else 'CPU-only'})")

    # llama_cpp.server checks a CONFIG_FILE env var *before* --config_file (see its
    # __main__.py: `os.environ.get("CONFIG_FILE", args.config_file)`) and tries to
    # JSON-parse whatever it points to as its own server config. On JupyterHub/Open
    # OnDemand setups (e.g. Anvil), the notebook launcher itself sets CONFIG_FILE to
    # its own jupyter_notebook_config.py -- unrelated to llama.cpp, but the name
    # collides, and llama.cpp's server crashes trying to parse that Python file as
    # JSON ("Invalid JSON: expected value..."). Strip it from this subprocess's
    # environment so it can't interfere.
    env = os.environ.copy()
    env.pop("CONFIG_FILE", None)

    process = subprocess.Popen(
        [
            "python3", "-m", "llama_cpp.server",
            "--model", model_path,
            "--n_ctx", str(n_ctx),
            "--n_gpu_layers", str(n_gpu_layers),
            "--n_threads", "-1",
            "--chat_format", "chatml-function-calling",
            "--host", HOST,
            "--port", str(PORT),
        ],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,
        env=env,
    )
    SERVER_PROCESS = process
    print(f"Logs: {log_file}")
    print(f"Waiting for the server at http://{HOST}:{PORT} ...")

    for _ in range(60):
        if process.poll() is not None:
            print("Server process exited early -- check llama_server.log")
            return
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=1)
            print(f"Server ready at http://{HOST}:{PORT}/v1")
            break
        except Exception:
            time.sleep(2)
    else:
        print("Server did not become ready in time -- check llama_server.log")

    atexit.register(stop_llama_server)


def stop_llama_server():
    global SERVER_PROCESS
    if SERVER_PROCESS is None:
        return
    try:
        os.killpg(os.getpgid(SERVER_PROCESS.pid), signal.SIGTERM)
        print("llama.cpp server stopped")
    except ProcessLookupError:
        pass
    finally:
        SERVER_PROCESS = None
