@AGENTS.md

# Codebase Guide for AI Assistants

## Project Overview

vLLM is a high-throughput, memory-efficient inference and serving engine for large language models (LLMs). It supports 200+ model architectures, multi-modal inputs, distributed execution, and an OpenAI-compatible API.

- **Language**: Python (with C++/CUDA kernels in `csrc/`)
- **Python**: >=3.10, <3.14
- **Build**: setuptools with cmake/ninja for native extensions
- **Package manager**: `uv` (preferred)
- **License**: Apache-2.0

## Repository Structure

```
vllm/                    # Main package
├── v1/                  # V1 engine (default, production-ready)
│   ├── engine/          # LLMEngine, AsyncLLMEngine
│   ├── executor/        # Execution backends (uniproc, multiproc, ray)
│   ├── worker/          # Hardware workers (GPU, CPU, XPU)
│   ├── core/            # KV cache management, block pools
│   ├── attention/       # Attention backend implementations
│   ├── sample/          # Sampling and logits processing
│   ├── spec_decode/     # Speculative decoding
│   └── structured_output/ # Constrained generation
├── engine/              # Legacy engine (now aliases to v1)
├── entrypoints/         # User-facing interfaces
│   ├── llm.py           # Python API (LLM class for offline inference)
│   ├── openai/          # OpenAI-compatible HTTP API server
│   ├── cli/             # CLI commands (vllm serve, vllm launch, etc.)
│   └── grpc_server.py   # gRPC API
├── config/              # Configuration system (~25 config classes)
│   ├── vllm.py          # VllmConfig (aggregates all sub-configs)
│   ├── model.py         # ModelConfig
│   ├── parallel.py      # ParallelConfig (TP/PP/DP/EP)
│   ├── cache.py         # CacheConfig (KV cache)
│   ├── scheduler.py     # SchedulerConfig
│   ├── compilation.py   # CompilationConfig (CUDA graphs, torch.compile)
│   └── ...              # attention, device, kernel, lora, multimodal, etc.
├── model_executor/      # Model execution and loading
│   ├── models/          # Individual model implementations (200+)
│   │   └── registry.py  # ModelRegistry (HF class name -> vLLM impl)
│   ├── layers/          # Reusable NN components (attention, MLP, norms)
│   └── model_loader/    # Weight loading (safetensors, bin, gguf)
├── distributed/         # Multi-GPU/multi-node infrastructure
│   ├── parallel_state.py # Process group management
│   └── device_communicators/ # NCCL, MPI, GLOO backends
├── multimodal/          # Multi-modal input processing (image, audio, video)
├── lora/                # LoRA adapter support
├── reasoning/           # Reasoning token parsers (DeepSeek R1, Qwen3, etc.)
├── compilation/         # Graph compilation (CUDA graphs, torch.compile)
├── renderers/           # Chat template rendering per model
├── inputs/              # Input data types and preprocessing
├── platforms/           # Hardware platform detection (CUDA, CPU, XPU, TPU)
├── plugins/             # Extension points (I/O processors, LoRA resolvers)
├── tool_parsers/        # Function/tool call parsing
├── tokenizers/          # Tokenizer management
├── kernels/             # Custom kernel wrappers
├── tracing/             # Distributed tracing (OTLP)
├── sampling_params.py   # SamplingParams dataclass
├── outputs.py           # Output types (CompletionOutput, RequestOutput)
├── envs.py              # Environment variable configuration
└── version.py           # Version info

csrc/                    # C++/CUDA kernel implementations
tests/                   # Test suite (mirrors vllm/ structure)
benchmarks/              # Performance benchmarks
docs/                    # Documentation (MkDocs)
examples/                # Usage examples
requirements/            # Dependency files by target
tools/                   # Developer tools and pre-commit scripts
docker/                  # Docker build files
```

## Quick Reference Commands

```bash
# Environment setup
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install

# Install (Python-only changes)
VLLM_USE_PRECOMPILED=1 uv pip install -e .

# Install (with C++/CUDA changes)
uv pip install -e .

# Run a specific test
pytest tests/path/to/test.py -v -s -k test_name

# Run linters
pre-commit run --all-files

# Run specific linter
pre-commit run ruff-check --all-files

# Run mypy (as CI does it)
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

## Architecture Key Concepts

### V1 Engine (Default)

The V1 engine (`vllm/v1/`) is the production-ready inference engine:

- **Engine** (`v1/engine/`): Request scheduling, continuous batching, output processing
- **Executor** (`v1/executor/`): Manages worker processes. Backends: `uniproc` (single GPU), `multiproc` (multi-GPU same node), `ray` (multi-node)
- **Worker** (`v1/worker/`): Runs on each device, executes model forward passes
- **Core** (`v1/core/`): KV cache block management and scheduling decisions

The legacy `vllm/engine/LLMEngine` is now an alias to `vllm.v1.engine.llm_engine.LLMEngine`.

### Configuration System

Configuration uses dataclasses/Pydantic in `vllm/config/`. The top-level `VllmConfig` aggregates all sub-configs. Key configs:
- `ModelConfig` - model identity, dtype, quantization
- `ParallelConfig` - tensor/pipeline/data/expert parallelism
- `SchedulerConfig` - batching and scheduling
- `CacheConfig` - KV cache type and strategy
- `CompilationConfig` - optimization levels O0-O3, CUDA graphs

### Model Registration

Models are registered in `vllm/model_executor/models/registry.py`, mapping HuggingFace architecture names to vLLM implementations. Models implement interfaces like `SupportsLoRA`, `SupportsMultiModal`, `SupportsPP`.

### Entry Points

1. **Python API**: `from vllm import LLM, SamplingParams` (offline inference)
2. **HTTP Server**: `vllm serve <model>` (OpenAI-compatible API)
3. **CLI**: `vllm` command with subcommands (serve, launch, run-batch, collect-env)

## Coding Conventions

### Linting and Formatting

Pre-commit hooks enforce (see `.pre-commit-config.yaml`):
- **ruff**: Linting (E, F, UP, B, ISC, SIM, I, G rules) and formatting
- **mypy**: Type checking with pydantic plugin (`check_untyped_defs: true`)
- **clang-format**: C++/CUDA formatting
- **typos**: Spell checking
- **markdownlint**: Markdown formatting
- **shellcheck**: Shell script linting
- **SPDX headers**: Required on Python files
- **No `torch.cuda` calls**: Use platform abstraction instead
- **No forbidden imports**: Enforced by `check_forbidden_imports.py`
- **Config validation**: All config fields must have defaults and docstrings
- **Signed-off-by**: Automatically added via commit-msg hook

### Important Rules

- **Do NOT use `torch.cuda` directly** - use the platform abstraction in `vllm/platforms/`
- **Config fields require docstrings** - enforced by `validate_config.py`
- **SPDX license headers** required on Python files
- **Lazy imports** in `vllm/__init__.py` - checked by `check_init_lazy_imports.py`
- **No spaces in filenames** - enforced by pre-commit

### Test Organization

Tests mirror the `vllm/` structure under `tests/`. Key directories:
- `tests/v1/` - V1 engine tests (e2e, engine, executor, core)
- `tests/models/` - Model correctness tests
- `tests/distributed/` - Multi-GPU tests
- `tests/entrypoints/` - API endpoint tests
- `tests/kernels/` - Kernel tests
- `tests/lora/` - LoRA tests

Pytest markers: `slow_test`, `core_model`, `hybrid_model`, `distributed`, `optional`

### Dependencies

Requirements files in `requirements/`:
- `test.txt` / `test.in` - Test dependencies (compiled via uv pip-compile)
- `lint.txt` - Linting tools (pre-commit, etc.)
- `build.txt` - Build dependencies
- `cuda.txt`, `cpu.txt`, `rocm.txt`, `tpu.txt`, `xpu.txt` - Platform-specific
- `dev.txt` - Development extras
- `docs.txt` - Documentation build

## Common Patterns

### Adding a New Model

1. Create implementation in `vllm/model_executor/models/<model_name>.py`
2. Register in `vllm/model_executor/models/registry.py`
3. Add tests in `tests/models/`

### Adding Configuration

1. Add field to the appropriate config class in `vllm/config/`
2. Include a default value and docstring (enforced by pre-commit)
3. Wire through `VllmConfig` if needed

### Adding a Reasoning Parser

1. Create parser in `vllm/reasoning/<model>_reasoning_parser.py`
2. Extend `ReasoningParser` base class from `abs_reasoning_parsers.py`

### Adding a Renderer

1. Create renderer in `vllm/renderers/<model>.py`
2. Register in `vllm/renderers/registry.py`
