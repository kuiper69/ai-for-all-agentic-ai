# AI for All — Mississippi Workshop: Agentic AI

Companion Jupyter notebook for the "Agentic AI" theory segment of the AI for All workshop. Turns each concept — agent loops, schema-enforced control flow, tool calling, RAG, and MCP — into runnable code, using a small (1.5B) CPU-only local model so it runs on a free Google Colab instance with no GPU and no gated model downloads.

> Full write-up and usage instructions coming soon. For now:

## Quickstart

```bash
git clone https://github.com/kuiper69/ai-for-all-mississippi-agentic-ai.git
cd ai-for-all-mississippi-agentic-ai
pip install -r requirements.txt
jupyter notebook agentic_ai_workshop.ipynb
```

The first cell in the notebook downloads a small open-weight model (Qwen2.5-1.5B-Instruct, Apache-2.0, ~1GB) and starts a local OpenAI-compatible server for it — no external API key or account needed.
