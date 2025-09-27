# Repo2View

A lightweight, browser-based code explorer for analyzing Python codebases. This tool provides a clean, interactive interface to navigate and understand code structure without heavy dependencies.

## Features

- 🚀 **Lightweight** - No heavy dependencies, runs in your browser
- 🔍 **Code Navigation** - Quickly jump between classes and functions
- 📊 **Code Analysis** - View file structure, imports, and docstrings
- 🔎 **Search** - Find text within files
- 🎨 **Dark Theme** - Easy on the eyes for long coding sessions

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to `http://localhost:8000`
3. Enter the path to your Python project directory
4. Click "Analyze" to explore your codebase

## Keyboard Shortcuts

- `Ctrl+F` - Focus the search box
- `Escape` - Clear search
- `Up/Down` - Navigate through search results

## How It Works

1. The backend uses Python's built-in `ast` module to parse Python files and extract:
   - Classes and their methods
   - Functions
   - Imports
   - Docstrings

2. The frontend provides an interactive interface to:
   - Browse files in the project
   - View file contents with syntax highlighting
   - Navigate using the code outline
   - Search within files

## License

MIT
