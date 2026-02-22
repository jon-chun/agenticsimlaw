# These ensembles are synched with /src/config_ollama_models_[all||oss|reasoning|size].yaml files
OLLAMA_ENSEMBLE_SIZE_LS = [
    "llama3.2:1b-instruct-q4_K_M",  # 1400
    "llama3.2:1b-instruct-fp16",    # 538
    "llama3.2:3b-instruct-q4_K_M",  # 300
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-fp16",
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",   
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M", 
]

OLLAMA_ENSEMBLE_OSS_LS = [
    "command-r:35b-08-2024-q4_K_M",  # 300
    "gemma2:9b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-moe:3b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "phi4:14b-q4_K_M",
]
# Add these from previous SIZE ensemble
#     "qwen2.5:7b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",

OLLAMA_ENSEMBLE_REASONING_LS = [
    "dolphin3:8b-llama3.1-q4_K_M",     # 300
    "exaone3.5:7.8b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "marco-o1:7b-q4_K_M",
    "olmo2:7b-1124-instruct-q4_K_M",
    "falcon3:7b-instruct-q4_K_M",      # 36
    "hermes3:8b-llama3.1-q4_K_M",      # 76 (runpod 0255)
    "internlm2:7b-chat-1m-v2.5-q4_K_M", # 226
    "nemotron-mini:4b-instruct-q4_K_M", # 0 
    "smallthinker:3b-preview-q4_K_M",   # 104
    "smollm2:1.7b-instruct-q4_K_M",     # 0
    "tulu3:8b-q4_K_M",
    "opencoder:8b-instruct-q4_K_M",     # 8
    "qwen2.5:32b-instruct-q4_K_M",      # 300
    "yi:9b-v1.5-q4_K_M",                # 300
]
# Add these from previous SIZE ensemble
#     "qwen2.5:7b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",
# Add these from previous REASONING ensemble
#     "command-r:35b-08-2024-q4_K_M",
#     "falcon3:7b-instruct-q4_K_M",
#     "gemma2:9b-instruct-q4_K_M",
#     "granite3.1-dense:8b-instruct-q4_K_M",
#     "llama3.1:8b-instruct-q4_K_M",
#     "marco-o1:7b-q4_K_M",
#     "phi4:14b-q4_K_M",
#     "qwen2.5:7b-instruct-q4_K_M",
#     "tulu3:8b-q4_K_M",
# Add the LARGE REASONING models
#     "tulu3:8b-q4_K_M",
#     "qwq:32b-preview-q4_K_M",
#     "qwen2.5:72b-instruct-q4_K_M",
#     "reflection:70b-q4_K_M",
#     "athene-v2:72b-q4_K_M",

OLLAMA_ENSEMBLE_ALL_LS = [
    "aya-expanse:8b-q4_K_M",
    "aya-expanse:8b-fp16",
    "aya-expanse:32b-q4_K_M",
    "command-r:35b-08-2024-q4_K_M",
    "dolphin3:8b-llama3.1-q4_K_M",
    "dolphin3:8b-llama3.1-fp16",
    "dolphin-llama3.1:8b-v2.9.4-Q4_K_M",
    "dolphin-llama3.1:8b-v2.9.4-Q8_0",
    "dolphin-mistral-nemo:12b-v2.9.3-Q4_K_M",
    "dolphin-mistral-nemo:12b-v2.9.3-F16",
    "dolphin-2.9.2-qwen2-7b:Q4_K_M",
    "dolphin-2.9.2-qwen2-7b:f16",
    "exaone3.5:32b-instruct-q4_K_M",
    "exaone3.5:7.8b-instruct-q4_K_M",
    "exaone3.5:7.8b-instruct-fp16",
    "exaone3.5:2.4b-instruct-q4_K_M",
    "exaone3.5:2.4b-instruct-fp16",
    "falcon3:3b-instruct-fp16",
    "falcon3:7b-instruct-q4_K_M",
    "falcon3:7b-instruct-fp16",
    "falcon3:10b-instruct-q4_K_M",
    "falcon3:10b-instruct-fp16",
    "gemma2:2b-instruct-fp16",
    "gemma2:9b-instruct-q4_K_M",
    "gemma2:9b-instruct-fp16",
    "gemma2:27b-instruct-q4_K_M",
    "glm4:9b-chat-q4_K_M",
    "granite3.1-dense:2b-instruct-fp16",
    "granite3.1-dense:8b-instruct-q4_K_M",
    "granite3.1-dense:8b-instruct-fp16",
    "hermes3:8b-llama3.1-q4_0",
    "hermes3:8b-llama3.1-fp16",
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:1b-instruct-fp16",
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-fp16",
    "llama3.1:70b-instruct-q4_K_M",
    "llama3.3:70b-instruct-q4_K_M",
    "marco-o1:7b-q4_K_M",
    "marco-o1:7b-fp16",
    "mistral:7b-instruct-q4_K_M",
    "mistral:7b-instruct-fp16",
    "mistral-nemo:12b-instruct-2407-q4_K_M",
    "mistral-small:22b-instruct-2409-q4_K_M",
    "mixtral:8x7b-instruct-v0.1-q4_K_M",
    "nemotron-mini:4b-instruct-q4_K_M",
    "nemotron-mini:4b-instruct-fp16",
    "olmo2:7b-1124-instruct-q4_K_M",
    "olmo2:7b-1124-instruct-fp16",
    "olmo2:13b-1124-instruct-q4_K_M",
    "olmo2:13b-1124-instruct-fp16",
    "phi4:14b-q4_K_M",
    "phi4:14b-fp16",
    "qwen2.5:0.5b-instruct-q4_K_M",
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-fp16",
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen2.5:32b-instruct-q4_K_M",
    "sailor2:8b-chat-q4_K_M",
    "sailor2:20b-chat-q4_K_M",
    "smollm2:1.7b-instruct-q4_K_M",
    "smollm2:1.7b-instruct-fp16",
    "solar-pro:22b-preview-instruct-q4_K_M",
    "tulu3:8b-q4_K_M",
    "tulu3:8b-fp16",
    "tulu3:70b-q4_K_M",
    "athene-v2:72b-q4_K_M",
    "qwen2.5:72b-instruct-q4_K_M",
    "reflection:70b-q4_K_M",
    "smallthinker:3b-preview-q4_K_M",
    "smallthinker:3b-preview-fp16",
    "qwq:32b-preview-q4_K_M",
    "yi:9b-v1.5-q4_K_M",
]