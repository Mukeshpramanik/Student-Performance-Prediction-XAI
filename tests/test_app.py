"""
Smoke tests for Streamlit application components and script imports.
"""

import pytest
import py_compile
from pathlib import Path


def test_app_syntax_and_compilation():
    """Verify app.py has zero syntax errors and compiles cleanly."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists()
    py_compile.compile(str(app_path), doraise=True)


def test_app_dependencies_available():
    """Verify all key dependencies for app.py can be imported."""
    import streamlit
    import plotly
    import shap
    import sklearn
    import pandas
    import numpy
