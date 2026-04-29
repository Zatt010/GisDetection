# UV Project Management Setup Guide

This guide explains how to use UV for Python project management in the GIS Detection project.

## What is UV?

UV is an extremely fast Python package installer and resolver, written in Rust. It's designed to be a drop-in replacement for pip and pip-tools workflows, with significant performance improvements.

## Installation

```bash
# Install UV using pip
pip install uv

# Or install using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Project Structure

The project is organized with UV in mind:

```
Scripts/
├── Backend/
│   ├── pyproject.toml          # UV configuration for backend
│   └── ... (backend files)
├── IA/
│   ├── pyproject.toml          # UV configuration for ML/AI components
│   └── ... (ML files)
└── UV_SETUP.md                # This guide
```

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to the Backend directory
cd Backend

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or install dependencies directly
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv sync --dev
```

### 2. IA (Machine Learning) Setup

```bash
# Navigate to the IA directory
cd IA

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or install dependencies directly
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv sync --dev
```

## Common UV Commands

### Dependency Management

```bash
# Add a new dependency
uv add package_name

# Add a development dependency
uv add --dev package_name

# Add a specific version
uv add package_name==1.2.3

# Remove a dependency
uv remove package_name

# Update dependencies
uv sync --upgrade
```

### Running Applications

```bash
# Run a command in the project environment
uv run python script.py

# Run FastAPI backend
uv run uvicorn gateway_main:app --reload

# Run with specific Python version
uv run --python 3.9 python script.py
```

### Virtual Environment Management

```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.9

# Remove virtual environment
rm -rf .venv
```

## Migration from Conda

The dependencies have been extracted from your `geospatial_project` conda environment and converted to UV-compatible format in the `pyproject.toml` files.

### Key Differences:

1. **Speed**: UV is 10-100x faster than pip/pip-tools
2. **Deterministic**: Lock files ensure reproducible builds
3. **Simplified**: Single `pyproject.toml` instead of multiple files
4. **Universal**: Works across different Python installations

### Benefits Over Conda:

- Faster dependency resolution
- Better compatibility with standard Python tools
- More reliable dependency locking
- Easier CI/CD integration
- Smaller environment sizes

## Development Workflow

### Backend Development

```bash
cd Backend

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .

# Run development server
uv run uvicorn gateway_main:app --reload
```

### IA/ML Development

```bash
cd IA

# Install dependencies
uv sync --dev

# Run training script
uv run python train_cnn_landcover.py

# Run inference
uv run python inferencia_final_tesis.py

# Start Jupyter notebook
uv run jupyter notebook
```

## Dependency Lock Files

UV automatically generates `uv.lock` files that ensure:

- **Reproducible builds** across machines
- **Exact versions** of all dependencies
- **Fast installations** with pre-resolved dependencies

Commit the `uv.lock` files to version control for team collaboration.

## Troubleshooting

### Common Issues

1. **Python Version Mismatch**:
   ```bash
   uv venv --python 3.9
   ```

2. **Dependency Conflicts**:
   ```bash
   uv sync --resolution=lowest-direct
   ```

3. **Cache Issues**:
   ```bash
   uv cache clean
   ```

### Getting Help

```bash
# UV help
uv --help

# Specific command help
uv add --help

# Check UV version
uv --version
```

## Best Practices

1. **Always use `uv sync`** to ensure consistent environments
2. **Commit `uv.lock` files** to version control
3. **Use `uv add`** instead of manual `pyproject.toml` editing
4. **Keep development dependencies** separate with `--dev`
5. **Use `uv run`** for running commands in project context
6. **Regular updates**: `uv sync --upgrade` monthly

## Performance Tips

- UV caches downloads automatically
- Use `uv pip install` for faster single-package installs
- Enable parallel resolution with `uv sync --resolution=highest`
- Use `uv cache clean` if you encounter cache corruption

## Integration with IDEs

Most IDEs (VS Code, PyCharm) automatically detect UV virtual environments. Ensure you select the `.venv` directory as your Python interpreter.

## Next Steps

1. Replace any remaining `pip install` commands with `uv add`
2. Update CI/CD pipelines to use UV
3. Train team members on UV workflow
4. Migrate any remaining conda environments to UV

For more information, see the [official UV documentation](https://docs.astral.sh/uv/).
