import os
from pathlib import Path
from llama_cpp import Llama

# MODEL CONFIGURATION
DEFAULT_MODEL_PATH = Path("./models/llama-3-8b-instruct.Q4_K_M.gguf")

# Context window = 4096 (safe maximum for Q4_K_M on CPU)
N_CTX = 4096

# Number of CPU threads to use for inference
N_THREADS = max(1, (os.cpu_count() or 4) // 2)

# Number of layers to offload to GPU
# 0 = pure CPU inference, and set to positive int if machine has CUDA or GPU
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", 0))

# Singleton
_llm_instance: Llama | None = None


def get_llm() -> Llama:
    """
    Returns the shared Llama instance, loading it on first call

    We use singleton because:
      1. Q4_K_M model takes 10-30 secodns to load. We only load it once at startup
      and reuse it across all three agents
      2. llama-cpp-python is not thread-safe by default. A singleton ensures there
      is exactly one instance, and concurrent calls will queue rather than corrupt
      shared state
    """
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = _load_model()
    
    return _llm_instance

def reset_llm() -> None:
    """ Clears the singleton instance """
    global _llm_instance
    _llm_instance = None


# Private loader
def _load_model() -> Llama:
    """
    Loads GGUF model from disk
    """
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel file not found at: {model_path}\n"
            f"Run the following command to download it:\n"
            f"  bash scripts/download_model.sh\n"
            f"Or set the MODEL_PATH environment variable to point "
            f"to an existing GGUF file."
        )
    
    print(f"Loading model from {model_path} ...")
    print(f"Threads: {N_THREADS} | Context: {N_CTX} | GPU layers: {N_GPU_LAYERS}")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
        use_mmap=True
    )

    print("Model loaded\n")
    return llm

