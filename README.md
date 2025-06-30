# Conversation-Fact-Benchmark

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

This project uses datasets managed through HuggingFace. The `datasets/` folder is excluded from this Git repository.

To set up the datasets:

1. **Clone your HuggingFace dataset repositories**:

   ```bash
   mkdir -p datasets
   cd datasets
   # Replace with your actual HuggingFace dataset URLs
   git clone https://huggingface.co/datasets/YOUR_USERNAME/dream
   git clone https://huggingface.co/datasets/YOUR_USERNAME/truthful_qa
   ```

2. **Or run the setup script**:
   ```bash
   chmod +x setup_datasets.sh
   ./setup_datasets.sh
   ```

### Running the Benchmark

```bash
# Make sure you're in the project directory and virtual environment is active
uv run python main.py

# Or if you've activated the virtual environment manually:
python main.py
```
