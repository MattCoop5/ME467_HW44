# Engineering Python Project Template

This is a GitHub template repository for student projects that use a standard Python development environment. Use this template to create a new repository for your assignments and course projects.

## Getting Started

### Prerequisites

Before using this template, ensure you have completed the one-time setup:

1. Install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Install the required VS Code extensions:
   - Python (Microsoft)
   - GitHub Copilot (if you have access)
3. Install `uv` from [https://astral.sh/uv/](https://astral.sh/uv/)

See the "Development Environments" guide for detailed setup instructions.

### Creating Your Project Repository

1. Visit this template repository on GitHub
2. Click "Use this template" → "Create a new repository"
3. Name your repository (e.g., `assignment-1`)
4. Choose Public or Private
5. Click "Create repository from template"
6. Clone your new repository locally

### Setting Up Your Project

Once you've cloned your repository, run the bootstrap script for your operating system:

**macOS:**
```bash
bash scripts/bootstrap-macos.sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap-windows.ps1
```

**Linux / WSL:**
```bash
bash scripts/bootstrap-linux.sh
```

The script will:
- Install Python 3.13 (if not already installed)
- Create a virtual environment (`.venv`)
- Install all dependencies specified in `pyproject.toml`
- Configure VS Code to use the virtual environment automatically

### Opening Your Project

After running the bootstrap script, open the project in VS Code:

```bash
code .
```

VS Code should automatically detect and use the Python interpreter from `.venv`. If not, see the troubleshooting section of the course guide.

## Project Structure

```
your-project/
├── README.md                 # Project documentation (update this!)
├── pyproject.toml           # Python project configuration and dependencies
├── uv.lock                  # Locked dependency versions (auto-generated)
├── .python-version          # Python version specification
├── .vscode/
│   └── settings.json        # VS Code workspace settings
├── scripts/
│   ├── bootstrap-macos.sh   # macOS setup script
│   ├── bootstrap-linux.sh   # Linux/WSL setup script
│   └── bootstrap-windows.ps1 # Windows setup script
├── src/                     # Your Python source code
└── tests/                   # Test files
```

## Managing Dependencies

### Adding a New Package

Edit `pyproject.toml` and add the package to the `dependencies` list:

```toml
[project]
dependencies = [
  "existing-package",
  "new-package",  # Add here
]
```

Then synchronize your environment:

```bash
uv sync
```

### Removing a Package

Remove it from `pyproject.toml` and run:

```bash
uv sync
```

### Locking Dependencies

After modifying dependencies, commit both `pyproject.toml` and `uv.lock` to version control:

```bash
git add pyproject.toml uv.lock
git commit -m "Update dependencies"
```

## Resetting Your Environment

If you encounter issues with your Python environment, reset it completely:

**macOS:**
```bash
rm -rf .venv
uv sync
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

Then restart VS Code.

## Running Your Code

### Python Scripts

```bash
python src/my_script.py
```

### Tests

```bash
python -m pytest tests/
```

## Using AI Assistance with Copilot

With GitHub Copilot installed, you can:

- Press `Ctrl+K` (or `Cmd+K` on macOS) to start an inline chat
- Use suggestions as you type code
- Ask questions about your code in the Copilot Chat panel

