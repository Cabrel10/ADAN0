import os
import ast
from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles

# Configuration
DEFAULT_EXTENSIONS = "py"

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI app
app = FastAPI(title="Repo2View")

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# Configure templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class CodeAnalyzer:
    """Analyze Python code and extract its structure."""

    @staticmethod
    def analyze_python_file(file_path: str) -> dict:
        """Analyze a Python file and extract its structure."""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read(), filename=file_path)
            except Exception as e:
                return {"error": f"Error parsing {file_path}: {str(e)}"}

        structure = {
            "file_path": file_path,
            "classes": [],
            "functions": [],
            "imports": [],
            "docstring": ast.get_docstring(tree) or ""
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "methods": [],
                    "line_start": node.lineno,
                    "line_end": node.end_lineno
                }
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info["methods"].append({
                            "name": item.name,
                            "docstring": ast.get_docstring(item) or "",
                            "line_start": item.lineno,
                            "line_end": item.end_lineno
                        })
                structure["classes"].append(class_info)
            elif (isinstance(node, ast.FunctionDef) and 
                  not any(isinstance(parent, (ast.ClassDef, ast.AsyncFunctionDef)) 
                         for parent in ast.walk(node))):
                structure["functions"].append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                structure["imports"].append(ast.unparse(node))

        return structure

    @staticmethod
    async def get_file_content(file_path: str) -> str:
        """Read file content asynchronously."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/repo2text", response_class=HTMLResponse)
async def repo2text(request: Request):
    """Serve the Repo2Text page."""
    return templates.TemplateResponse("repo2text.html", {"request": request})


@app.post("/analyze")
async def analyze_directories(
    directories: str = Form(""),
    extensions: str = Form(DEFAULT_EXTENSIONS)
) -> JSONResponse:
    """
    Analyze multiple directories and return their structure.
    
    Args:
        directories: Comma-separated list of directory paths
        extensions: Comma-separated list of file extensions to include
        
    Returns:
        JSONResponse: Analysis results
    """
    if not directories:
        raise HTTPException(
            status_code=400,
            detail="No directories provided"
        )

    dir_list = [d.strip() for d in directories.split(",") if d.strip()]
    
    # Verify all directories exist
    for directory in dir_list:
        if not os.path.exists(directory):
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {directory}"
            )
    
    analyzer = CodeAnalyzer()
    results = {
        "files": [],
        "total_files": 0,
        "total_classes": 0,
        "total_functions": 0,
        "directories_analyzed": []
    }

    extensions_list = [ext.strip() for ext in extensions.split(",") if ext.strip()]
    
    for directory in dir_list:
        dir_files = []
        dir_classes = 0
        dir_functions = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions_list):
                    file_path = os.path.join(root, file)
                    try:
                        analysis = analyzer.analyze_python_file(file_path)
                        if "error" not in analysis:
                            results["total_files"] += 1
                            results["total_classes"] += len(analysis["classes"])
                            results["total_functions"] += len(analysis["functions"])
                            results["files"].append(analysis)
                            
                            # Update directory stats
                            dir_files.append(file_path)
                            dir_classes += len(analysis["classes"])
                            dir_functions += len(analysis["functions"])
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {str(e)}")
        
        # Add directory statistics
        results["directories_analyzed"].append({
            "path": directory,
            "file_count": len(dir_files),
            "class_count": dir_classes,
            "function_count": dir_functions
        })
    
    return JSONResponse(content=results)


@app.get("/file")
async def get_file_content(path: str) -> dict:
    """Get the content of a specific file.
    
    Args:
        path: Path to the file
        
    Returns:
        dict: File content or error message
    """
    # Security check to prevent directory traversal
    if not os.path.isabs(path) or not path.startswith('/'):
        path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}"
        )
    
    try:
        content = await CodeAnalyzer.get_file_content(path)
        return {"content": content, "file_path": path}
    except Exception as e:
        print(f"Error reading file {path}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Ensure required directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Localhost only for security
        port=8000,
        reload=True
    )
