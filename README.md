# Conversation Fact Benchmark

## Setup

### Prerequisites

1. **Install uv** (Python package manager):

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or with pip
   pip install uv
   ```

2. **Install Git LFS** (for dataset management):

   ```bash
   # On macOS
   brew install git-lfs

   # On Ubuntu/Debian
   sudo apt install git-lfs

   # On other systems, visit: https://git-lfs.github.io/

   # Initialize Git LFS
   git lfs install
   ```

### Project Setup

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd Conversation-Fact-Benchmark
   ```

2. **Install Python dependencies**:

   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   # uv automatically creates a virtual environment
   # To activate it manually if needed:
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

### Dataset Setup

This project supports multiple conversation and fact-checking datasets:

#### Supported Datasets

1. **DREAM** - Dialog-based Reading comprehension ExAmination through understanding and reasoning
2. **TruthfulQA** - Questions that test whether a language model is truthful
3. **MSC-MemFuse-MC10** - Multi-Session Chat memory questions (10-way multiple choice)

#### Setting up MSC Dataset

The MSC (Multi-Session Chat) dataset tests conversational memory across multiple conversation sessions. To set it up:

```bash
# Run the MSC dataset setup script
python setup_msc_dataset.py
```

This will:

- Download the `Percena/msc-memfuse-mc10` dataset from HuggingFace
- Transform it to the benchmark format
- Save it to `datasets/msc/processed/msc_memfuse_mc10_transformed.json`

#### Setting up Other Datasets

For DREAM and TruthfulQA datasets:

1. **Clone your HuggingFace dataset repositories**:

   ```bash
   mkdir -p datasets
   cd datasets
   # Replace with your actual HuggingFace dataset URLs
   git clone https://huggingface.co/datasets/onionmonster/dream
   git clone https://huggingface.co/datasets/onionmonster/truthful_qa
   ```

2. **Or run the setup script**:
   ```bash
   chmod +x setup_datasets.sh
   ./setup_datasets.sh
   ```

### Running the Benchmark

#### Basic Usage

```bash
# Make sure you're in the project directory and virtual environment is active
uv run python main.py

# Or if you've activated the virtual environment manually:
python main.py
```

#### Dataset-Specific Examples

**MSC (Multi-Session Chat) Dataset:**

```bash
# Run evaluation on MSC dataset with Ollama
python main.py -d datasets/msc/processed/msc_memfuse_mc10_transformed.json -p ollama -m llama3.2:latest -n 20

# Run evaluation on MSC dataset with OpenRouter
export OPENROUTER_API_KEY="your_api_key_here"
python main.py -d datasets/msc/processed/msc_memfuse_mc10_transformed.json -p openrouter -m anthropic/claude-3-sonnet -n 10
```

**DREAM Dataset:**

```bash
python main.py -d datasets/dream/processed/full_transformed.json -p ollama -m llama3.2:latest -n 20
```

**TruthfulQA Dataset:**

```bash
python main.py -d datasets/truthful_qa/processed/truthful_qa_transformed.json -p ollama -m llama3.2:latest -n 20
```

#### Command Line Arguments

- `-d, --dataset`: Path to dataset file (automatically detects DREAM, TruthfulQA, or MSC format)
- `-p, --provider`: LLM provider (`ollama` or `openrouter`)
- `-m, --model`: Model name to evaluate
- `-n, --max-samples`: Number of questions to evaluate (0 = full dataset)
