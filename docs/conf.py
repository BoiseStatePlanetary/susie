# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Susie'
copyright = '2023, Boise State University'
author = 'Holly VanLooy, Malia Barker'
root_doc = 'index'
release = 'Sept. 2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

mathjax_path="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'sphinx-mathjax-offline', 'myst_nb']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The order in which the classes and methods are shown
autodoc_member_order = 'bysource'

autodoc_default_flags = ['members', 'undoc-members', 'private-members']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
# html_theme = "furo"
html_permalinks_icon = '<span>#</span>'
# html_theme = 'sphinxawesome_theme'
html_theme = 'piccolo_theme'
