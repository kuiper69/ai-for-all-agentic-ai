# Local Agentic AI Stack on an Anvil-Style HPC Cluster (Ollama + Hermes Agent)

This is the "Part II" follow-on to the from-scratch notebook: once the raw ReAct loop, tool-calling, RAG, and MCP mechanics are understood, Ollama (inference) + Hermes Agent (orchestration) automate all of it into a usable day-to-day setup. This guide walks through getting both running on a Slurm-managed GPU cluster, using Purdue's **Anvil** as the concrete reference cluster. Every command here is something a regular, unprivileged user can run on their own account — no root, no admin action, no editing shared module trees. If you're on a different Slurm cluster, the shape is identical; only partition names, account flags, and module names will differ (check with `sinfo` and `module spider` on your own system).

**Prerequisites:** SSH access to the cluster and an active Slurm allocation (an `-A <account>` you can charge jobs to). On Anvil, run `mybalance` to see which allocation names you have access to.

---

## 0. Why everything below lives in scratch, not home

HPC home directories are small and backed up; scratch is huge and unbacked-up. On Anvil specifically: `$HOME` is capped at **25 GB**, while `$SCRATCH` gives **100 TB** — but scratch has a **30-day purge policy** on files that haven't been touched. That trade-off shapes this whole guide: everything (the Python environment, Ollama's binaries and model weights, Hermes's state, every package cache) gets installed under `$SCRATCH`, and if you go quiet on this stack for a month, expect to re-pull your models.

---

## 1. Request GPU resources in an interactive shell

```bash
# Quick sanity check (gpu-debug: 1 node, max 2 GPUs, 30-minute limit)
sinteractive -p gpu-debug -A <your-allocation> --gpus-per-node=1

# Real work (gpu: up to 48 hours, up to 12 GPUs/user, 32/allocation)
sinteractive -p gpu -A <your-allocation> --gpus-per-node=1 --time=04:00:00
```

Anvil's GPU nodes are 4x A100 (128 CPU cores, 515 GB RAM) or, on the `ai` partition, 4x H100 (96 cores, 1 TB RAM) — one GPU per job is plenty for a single mid-size model. If your cluster doesn't provide the `sinteractive` wrapper, the equivalent is `salloc -p <gpu-partition> -A <account> --gpus-per-node=1 --time=04:00:00` followed by `srun --pty bash`.

---

## 2. Set up an isolated Python environment in scratch

```bash
mkdir -p $SCRATCH/agentic-ai-stack
cd $SCRATCH/agentic-ai-stack

module spider anaconda          # confirm the current module name/version on your cluster
module load anaconda            # e.g. anaconda/2021.05-py38 on Anvil at time of writing

conda create -p $SCRATCH/agentic-ai-stack/venv python=3.11 -y   # Hermes requires Python 3.11+
conda activate $SCRATCH/agentic-ai-stack/venv
```

(No conda preference either way — a plain `module load python/<version>` followed by `python3 -m venv $SCRATCH/agentic-ai-stack/venv` works the same way if you'd rather skip conda entirely.)

### Redirect every cache that defaults to `$HOME`

pip, conda, and HuggingFace all cache into your home directory by default, which is exactly how a 25 GB quota gets blown by a single afternoon of installs. Point them at scratch instead:

```bash
export XDG_CACHE_HOME=$SCRATCH/agentic-ai-stack/cache
export PIP_CACHE_DIR=$SCRATCH/agentic-ai-stack/cache/pip
export CONDA_PKGS_DIRS=$SCRATCH/agentic-ai-stack/cache/conda-pkgs
export HF_HOME=$SCRATCH/agentic-ai-stack/cache/huggingface
export TMPDIR=$SCRATCH/agentic-ai-stack/tmp
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TMPDIR"
```

Save these exports (plus the `PATH`/`HERMES_HOME`/`OLLAMA_*` variables from the steps below) into one `env.sh` file that you `source` at the start of every session, rather than baking them into `.bashrc` — Anvil's own docs specifically warn that module/env changes in shell profiles can break other tools (e.g. ThinLinc sessions).

---

## 3. Install Ollama (official archive, no root)

Ollama's official release archive moved from `.tgz` to `.tar.zst` (zstandard) — don't reuse an old `.tgz` URL, it's now a stub.

```bash
mkdir -p $SCRATCH/agentic-ai-stack/ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst
tar -xf /tmp/ollama.tar.zst -C $SCRATCH/agentic-ai-stack/ollama
export PATH=$SCRATCH/agentic-ai-stack/ollama/bin:$PATH
```

Modern `tar` (with the `zstd` binary available on `$PATH`) auto-detects `.tar.zst` and extracts it directly. If your cluster's `tar` is too old for that, decompress explicitly first: `zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar && tar -xf /tmp/ollama.tar -C $SCRATCH/agentic-ai-stack/ollama`.

---

## 4. Configure and start the Ollama server

```bash
export OLLAMA_MODELS=$SCRATCH/agentic-ai-stack/ollama/models   # model weights go to scratch, not $HOME/.ollama
export OLLAMA_CONTEXT_LENGTH=65536                              # set BEFORE starting the server -- Hermes needs >=64K context
ollama serve &
```

Then pull whichever model you're targeting — this is the same swappable-placeholder idea as the notebook: pick a model that fits your allocated GPU's memory and actually supports tool calling with a large context window.

```bash
ollama pull <model-name>
```

---

## 5. Install Hermes Agent

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
export HERMES_HOME=$SCRATCH/agentic-ai-stack/hermes   # keep Hermes's own state off $HOME too
```

**Gotcha:** Hermes bundles a native `node-pty` module that has to be compiled, and it needs a C++20-capable compiler. If the installer fails there, get a newer compiler first — either `module load gcc/<newer-version>` (check `module spider gcc` for what's available) or, inside a conda environment, `conda install -c conda-forge gxx` — then re-run the installer.

---

## 6. Point Hermes at your Ollama server

```bash
hermes config set CUSTOM_BASE_URL http://127.0.0.1:11434/v1   # the /v1 suffix is required -- Hermes will not infer it
hermes config set CUSTOM_API_KEY not-needed                    # Ollama ignores this value; any non-empty string works
```

(Equivalently, run `hermes model` for the interactive wizard and choose "Custom Endpoint".)

---

## 7. Verify

```bash
hermes doctor   # sanity-checks the connection and the model's capabilities
```

If Hermes rejects the model at startup, it's almost always one of two things: the context window is under 64K, or the model doesn't actually support tool calling. Both are hard requirements Hermes checks before it'll use a model.

---

## 8. Tear down cleanly

Stop the Ollama server before your interactive session ends (`pkill ollama`, or track the PID from `ollama serve &` and `kill` it) — otherwise it keeps sitting on the GPU allocation until the walltime limit hits. To run this stack unattended instead of in an interactive shell, wrap Steps 3-7 in an `sbatch` script and submit it as a batch job.

---

## Known gotchas, all in one place

- **Ollama's archive format changed**: `.tar.zst`, not `.tgz` — an old `.tgz` URL is a dead stub now.
- **`OLLAMA_CONTEXT_LENGTH` must be set before `ollama serve` starts** — it can't be changed on a running server.
- **`CUSTOM_BASE_URL` needs the explicit `/v1` suffix** — Hermes does not add it for you.
- **`node-pty`'s native build needs a C++20-capable compiler** — fix with a newer `gcc` module or `conda install -c conda-forge gxx`.
- **Hermes hard-requires >=64K context and tool-calling support** — it rejects incompatible models at startup rather than degrading silently.
- **Everything (env, Ollama, Hermes, caches) lives under `$SCRATCH`, never `$HOME`** — home quotas are small, and scratch's 30-day purge on untouched files is the trade-off for the extra room.

---

## Sources (Anvil-specific facts verified against Purdue RCAC docs)

- [Anvil User Guide: Slurm Partitions (Queues)](https://www.rcac.purdue.edu/knowledge/anvil/run/partitions?all=true)
- [Job Submission — RCAC Documentation](https://docs.rcac.purdue.edu/userguides/anvil/jobs/)
- [Anvil User Guide: Module System](https://www.rcac.purdue.edu/knowledge/anvil/software/modules)
- [System Architecture — RCAC Documentation](https://docs.rcac.purdue.edu/userguides/anvil/architecture/)
- [Ollama: Linux manual install](https://docs.ollama.com/linux)
- [Hermes Agent: Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)
- [Hermes Agent: FAQ & Troubleshooting](https://hermes-agent.nousresearch.com/docs/reference/faq)
