# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import shutil
import sys

from audiblelight import utils

sys.path.insert(0, os.path.abspath('..'))

project = 'AudibleLight'
copyright = '2025, Centre for Digital Music, Queen Mary University of London'
author = 'Centre for Digital Music, Queen Mary University of London'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'special-members': '__str__,__repr__,__eq__,__len__,__getitem__,__iter__',
    'undoc-members': True,
    'show-inheritance': True,
    'private-members': False,
}
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True


# Configure autodoc and typehints
autodoc_typehints = "description"
always_document_param_types = True
typehints_use_signature_return = True
typehints_fully_qualified = False


print("Copy example notebooks into docs/_examples")


def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if ".ipynb_checkpoints" in c:
            result.append(c)
        if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".ipynb")):
            result.append(c)
    return result


shutil.rmtree(utils.get_project_root() / "docs/_examples", ignore_errors=True)
shutil.copytree(utils.get_project_root() / "notebooks", utils.get_project_root() / "docs/_examples", ignore=all_but_ipynb)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_favicon = '_static/favicon.ico'
