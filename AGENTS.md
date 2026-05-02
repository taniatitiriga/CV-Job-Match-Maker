# CV-Job-Match Agent Guidelines

## Development Setup
- Install dependencies: `uv sync` (uses uv.lock for reproducible installs)
- Python version: 3.12+ (see pyproject.toml)
- Local LLM servers: Run `./run-llama-servers.sh` to start embedding and inference servers
- Qdrant: Uses local storage at ./qdrant_storage
- Opencode: Configuration in .opencode/package.json with available skills

## Key Directories
- `baml_src/`: BAML schema files defining data structures for LLM extraction
- `baml_client/`: Generated Python client for BAML (do not edit manually)
- `notebooks/`: Jupyter notebooks for data processing, testing, and exploration
  - `preprocess_datasets.ipynb`: Main notebook for preprocessing CV and job posting datasets
  - `baml_test_retrieval.ipynb`: Tests BAML extraction and retrieval functionality
  - `embed_processed_json_to_qdrant.ipynb`: Processes JSON data and stores embeddings in Qdrant
  - `ollama_connection.ipynb`: Tests connection to Ollama LLM servers
  - `qdrant_quickstart.ipynb`: Quickstart guide for Qdrant vector operations
- `src/`: Source code for the application
  - `app.py`: Entry point for the Dash application
  - `config.py`: Configuration settings
  - `embeddings.py`: Embedding generation utilities
  - `extraction.py`: Data extraction logic using BAML
  - `pages/`: Dash page components (upload.py, document_detail.py)
- `data/`: Dataset files
  - `data/cv-dataset/`: Original CV/resume documents (organized by domain)
  - `data/job-postings/`: Original job postings CSV file
  - `data/cv-dataset-processed/`: Processed CV/resume JSON files (organized by domain)
  - `data/job-postings-processed/`: Processed job posting JSON files (organized by domain)

## BAML Workflow
1. Edit `.baml` files in `baml_src/` to modify LLM prompts/extraction schemas
2. Regenerate client: Run `uv run baml-cli generate` in baml_src/
3. Import generated client from `baml_client` in Python code

## Testing & Execution
- Run notebooks: `uv run jupyter lab` or `uv run jupyter notebook`
- Test LLM connections: See `notebooks/ollama_connection.ipynb`
- Vector operations: See `notebooks/qdrant_quickstart.ipynb`
- Preprocess datasets: See `notebooks/preprocess_datasets.ipynb`
- Embed processed data: See `notebooks/embed_processed_json_to_qdrant.ipynb`

## Environment
- Ollama models: mxbai-embed-large (embeddings) and phi-4-mini (inference)
- Models stored in `../models/` relative to llama-server execution
- Servers run on ports 8080 (LLM) and 8081 (embeddings)

## Opencode Skills
- `ask-questions-if-underspecified`: Ask the minimum set of clarifying questions needed to avoid wrong work.
- `code-review-assistant`: Review code for best practices, potential bugs, and improvements
- `documentation-generator`: Generate comprehensive documentation for code, APIs, and projects
- `testing-assistant`: Assist with writing, running, and maintaining tests for code