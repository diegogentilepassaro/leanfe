"""
Setup script for leanfe documentation build.

This script ensures the leanfe package and dependencies are installed 
before building the docs. Run this before `quarto render`.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install Jupyter and other dependencies needed for Quarto."""
    print("Installing documentation build dependencies...")
    deps = ["jupyter", "pyyaml"]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + deps + ["-q"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Warning: Error installing dependencies: {result.stderr}")


def install_leanfe():
    """Install leanfe from local package directory."""
    docs_dir = Path(__file__).parent
    python_pkg_dir = docs_dir.parent / "python"
    
    if not python_pkg_dir.exists():
        print(f"Error: Python package directory not found at {python_pkg_dir}")
        sys.exit(1)
    
    print(f"Installing leanfe from {python_pkg_dir}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(python_pkg_dir), "-q"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error installing leanfe: {result.stderr}")
        sys.exit(1)
    
    print("leanfe installed successfully!")
    
    # Verify import works
    try:
        import leanfe
        print(f"Verified: leanfe {leanfe.__version__ if hasattr(leanfe, '__version__') else '(version unknown)'} is importable")
    except ImportError as e:
        print(f"Warning: Could not import leanfe after installation: {e}")


if __name__ == "__main__":
    install_dependencies()
    install_leanfe()
