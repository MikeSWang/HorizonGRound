# -*- coding: utf-8 -*-

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Configuration for the Sphinx Documentation Builder
#
# Read information here: http://www.sphinx-doc.org/en/master/config
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# -- Project information -----------------------------------------------------

project = 'HorizonGRound'
author = 'MS Wang'
copyright = '2019, MS Wang'
version = '0.0'
release = '0.0.dev0'


# -- Path setup --------------------------------------------------------------

import os, sys

sys.path.insert(0, os.path.abspath('../..'))


# -- General configuration ---------------------------------------------------

source_suffix = ['.rst', '.txt', '.md',]

master_doc = 'index'

exclude_patterns = []

templates_path = ['_templates']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

pygments_style = 'sphinx'

language = None


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

html_theme_options = {
    'page_width': '1125px',
    'sidebar_width': '225px',
    'fixed_sidebar' : True,
    'github_user': 'MikeSWang',
    'github_repo': 'HorizonGRound',
}

html_static_path = ['_static']

html_logo = '_static/HorizonGRound.png'

html_sidebars = {
    '**': ['navigation.html',
           'localtoc.html',
           'searchbox.html',
           ],
   'using/windows': ['windowssidebar.html', 'searchbox.html'],
}

htmlhelp_basename = 'HorizonGRound_doc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'\usepackage{upgreek}',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).

latex_documents = [
    (master_doc, 'HorizonGRound.tex',
     'HorizonGRound Documentation', author, 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).

man_pages = [
    (master_doc, 'horizonground', 'HorizonGRound Documentation', [author], 1),
]


# -- Extension configuration -------------------------------------------------

autodoc_member_order = 'bysource'

autosummary_generate = True

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None),
    'mpmath': ('http://mpmath.org/doc/1.1.0/', None),
    'nbodykit': ('https://nbodykit.readthedocs.io/en/latest', None),
}

todo_include_todos = True
