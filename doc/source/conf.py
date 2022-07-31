# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration scores, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'agas'
copyright = '2022, Eitan Hemed'
author = 'Eitan Hemed'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_automodapi.automodapi',
    'sphinx.ext.imgmath',
    'sphinx.ext.mathjax'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_sidebars = {
'**': ['localtoc.html'],
'index': []
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Display todos by setting to True
todo_include_todos = False

autodoc_inherit_docstrings = False
automodsumm_writereprocessed = True

# Generate the API documentation when building
autosummary_generate = True

numpydoc_show_class_members = False
automodapi_inheritance_diagram = False
autodoc_member_order = 'bysource'
