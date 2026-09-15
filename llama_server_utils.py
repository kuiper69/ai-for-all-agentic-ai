"""
Starts/stops a local llama.cpp OpenAI-compatible server from inside the notebook,
the same way the reference PEARC tutorial's ollama_utils.py drives Ollama --
just with llama-cpp-python instead, since that's a pure-pip CPU install with
no system binary or install script needed (Colab-friendly).
"""
import atexit
import os
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


def start_llama_server(model_path=None, n_ctx=4096):
    """Starts `python -m llama_cpp.server` in the background and waits until it responds."""
    global SERVER_PROCESS

    if model_path is None:
        model_path = download_model()

    subprocess.run(["pkill", "-f", "llama_cpp.server"], capture_output=True)
    time.sleep(1)

    log_file = "llama_server.log"
    print("Starting llama.cpp server...")
    process = subprocess.Popen(
        [
            "python3", "-m", "llama_cpp.server",
            "--model", model_path,
            "--n_ctx", str(n_ctx),
            "--n_threads", "-1",
            "--chat_format", "chatml-function-calling",
            "--host", HOST,
            "--port", str(PORT),
        ],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,
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
