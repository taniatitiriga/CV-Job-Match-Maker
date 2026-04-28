# CV-Job-Match Agent Guidelines

## Development Setup
- Install dependencies: `uv sync` (uses uv.lock for reproducible installs)
- Python version: 3.12+ (see pyproject.toml)
- Local LLM servers: Run `./run-llama-servers.sh` to start embedding and inference servers
- Qdrant: Uses local storage at ./qdrant_storage

## Key Directories
- `baml_src/`: BAML schema files defining data structures for LLM extraction
- `baml_client/`: Generated Python client for BAML (do not edit manually)
- `notebooks/`: Jupyter notebooks for data processing, testing, and exploration
- `data/`: Dataset files
  - `data/cv-dataset/`: CV/resume documents
  - `data/job-postings/` or `data/job_postings/`: Job postings

## BAML Workflow
1. Edit `.baml` files in `baml_src/` to modify LLM prompts/extraction schemas
2. Regenerate client: `uv run baml-cli generate` (or let editor handle it)
3. Import generated client from `baml_client` in Python code

## Testing & Execution
- Run notebooks: `uv run jupyter lab` or `uv run jupyter notebook`
- Test LLM connections: See `notebooks/ollama_connection.ipynb`
- Vector operations: See `notebooks/qdrant_quickstart.ipynb`

## Environment
- Ollama models: mxbai-embed-large (embeddings) and phi-4-mini (inference)
- Models stored in `../models/` relative to llama-server execution
- Servers run on ports 8080 (LLM) and 8081 (embeddings)