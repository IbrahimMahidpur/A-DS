"""
Central configuration for Multimodal Agentic DS Engine.
All settings from environment variables — no hardcoded secrets.

FIX: Default model names corrected to match docker-compose.yml and Ollama Hub:
  qwen3.5:9b  → qwen2.5:7b   (qwen3.5 doesn't exist on Ollama Hub)
  gemma4:latest → qwen2.5:7b  (gemma4 doesn't exist on Ollama Hub)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent.parent.parent
DATA_DIR  = ROOT_DIR / "data"
OUTPUT_DIR = Path(os.getenv("WORKING_DIR",       "./agentic_output"))
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
FAISS_DIR  = Path(os.getenv("FAISS_INDEX_DIR",    "./data/faiss"))

for d in [DATA_DIR, OUTPUT_DIR, CHROMA_DIR, FAISS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Specialized model assignments (override with env vars)
# Falls back to PLANNER_MODEL/CODER_MODEL if specialized model not pulled
STATS_INTERPRETER_MODEL = os.getenv("STATS_MODEL", os.getenv("REVIEWER_MODEL", "ollama/mistral:7b"))
CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", os.getenv("CODER_MODEL", "ollama/deepseek-coder-v2:16b"))
CODE_FIX_MODEL = os.getenv("CODE_FIX_MODEL", os.getenv("CODER_MODEL", "ollama/qwen2.5-coder:7b"))
HYPOTHESIS_MODEL = os.getenv("HYPOTHESIS_MODEL", os.getenv("PLANNER_MODEL", "ollama/deepseek-r1:7b"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "ollama/mxbai-embed-large")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PLANNER_MODEL   = os.getenv("PLANNER_MODEL",  "ollama/qwen2.5:7b")
CODER_MODEL     = os.getenv("CODER_MODEL",    "ollama/qwen2.5:7b")
REVIEWER_MODEL  = os.getenv("REVIEWER_MODEL", "ollama/qwen2.5:7b")
VISION_MODEL    = os.getenv("VISION_MODEL",   "ollama/llava:7b")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
LLM_TIMEOUT    = int(os.getenv("LLM_TIMEOUT",    "600"))
LLM_RETRIES    = int(os.getenv("LLM_RETRIES",    "1"))
ENABLE_PII     = os.getenv("ENABLE_PII_DETECTION", "true").lower() == "true"

# ── API ────────────────────────────────────────────────────
API_HOST  = os.getenv("API_HOST",  "0.0.0.0")
API_PORT  = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
