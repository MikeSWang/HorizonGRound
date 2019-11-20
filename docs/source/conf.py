import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'HorizonGRound'
copyright = '2019, Mike Shengbo Wang'
author = 'Mike Shengbo Wang'
release = '0.0'


# -- General configuration ---------------------------------------------------

source_suffix = ['.rst', '.txt', '.md']

master_doc = 'horizonground'

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

templates_path = ['_templates']

pygments_style = 'sphinx'

language = None

exclude_patterns = ['tests', 'scripts']


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

html_theme_options = {
    'page_width': '1150px',
    'sidebar_width': '250px',
    'fixed_sidebar' : True,
    'github_user': 'MikeSWang',
    'github_repo': 'HorizonGRound',
}

html_static_path = ['_static']

html_logo = '_static/HorizonGRound.png'

html_sidebars = {
    '**': ['navigation.html', 'searchbox.html'],
    'using/windows': ['windowssidebar.html', 'searchbox.html'],
}

htmlhelp_basename = 'HorizonGRound_doc'


# -- Extension configuration -------------------------------------------------

autodoc_member_order = 'bysource'
autosummary_generate = True

napoleon_include_special_with_doc = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None),
    'nbodykit': ('https://nbodykit.readthedocs.io/en/latest', None),
}

todo_include_todos = True