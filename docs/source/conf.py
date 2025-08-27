# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Hybridlane"
copyright = "%Y, Battelle Memorial Institute"
author = "PNNL"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",  # For automatic API documentation from docstrings
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",  # For generating summary tables of API elements
    "sphinx.ext.mathjax",  # For rendering math in HTML using MathJax/KaTeX
    # "sphinx_math_dollar",  # For inline math using $...$
    "sphinx.ext.viewcode",  # Link to the source code on GitHub/local
    "sphinx.ext.intersphinx",  # For linking to documentation of other projects
    "sphinx.ext.todo",  # For todo notes in documentation
    "sphinx.ext.extlinks",  # For defining short aliases for external links
    "sphinx.ext.githubpages",  # If deploying to GitHub Pages
    "sphinx_copybutton",  # Copy button for code blocks
]

# Configure autosummary to generate stubs
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit", None),
    "pennylane": ("https://docs.pennylane.ai/en/stable/", None),
}

# TODO extension settings
todo_include_todos = True

# Autoapi configuration
autoapi_dirs = [
    "../../src",  # build docs for the python package
]
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "special-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autodoc_typehints = "description"
autoapi_generate_api_docs = True
autoapi_python_class_content = "both"
autoapi_root = (
    "_autoapi"  # keep out of source directory so autobuild doesn't go haywire
)
autodoc_typehints_format = "short"
autoapi_own_page_level = "class"
autoapi_member_document_option = ["class"]
python_use_unqualified_type_names = True

# Exclude specific patterns from source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# Sphinx Math Dollar configuration for $...$ inline math
# math_dollar_inline = True
# math_dollar_display = True
