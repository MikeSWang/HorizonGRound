import os
import sys

sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

author = 'Mike Shengbo Wang'
copyright = '2020, Mike Shengbo Wang'
project = 'HorizonGRound'
release = '0.0'


# -- General configuration ---------------------------------------------------

exclude_patterns = ['config', 'application', 'scripts', 'tests']

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

language = None

master_doc = 'horizonground'

pygments_style = 'sphinx'

source_suffix = ['.rst', '.txt', '.md']

templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

htmlhelp_basename = 'HorizonGRound_doc'

html_logo = '_static/HorizonGRound.png'

html_sidebars = {
    '**': ['navigation.html', 'searchbox.html'],
    'using/windows': ['windowssidebar.html', 'searchbox.html'],
}

html_static_path = ['_static']

html_theme = 'alabaster'

html_theme_options = {
    'fixed_sidebar' : True,
    'github_repo': 'HorizonGRound',
    'github_user': 'MikeSWang',
    'page_width': '1150px',
    'sidebar_width': '250px',
}


# -- Extension configuration -------------------------------------------------

autodoc_member_order = 'bysource'
autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None),
    'nbodykit': ('https://nbodykit.readthedocs.io/en/latest', None),
}

napoleon_include_special_with_doc = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

todo_include_todos = True
