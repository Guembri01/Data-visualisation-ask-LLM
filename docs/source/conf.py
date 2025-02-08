import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Important: Path to your project root


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Data Visualization App'
copyright = '2025, Bilel Guembri et Wissal Ben Othmen'
author = 'Bilel Guembri et Wissal Ben Othmen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',     # Automatically document your Python code
    'sphinx.ext.napoleon',   # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',   # Add links to source code from documentation
    'sphinx.ext.intersphinx', # Link to other Sphinx documentation projects
    'sphinx_rtd_theme',      # Use the Read the Docs theme
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']