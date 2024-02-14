# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'PyJuice'
copyright = '2021, StarAI'
author = 'StarAI'

release = '2.0.0'
version = '2.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_sidebars = {
    '**': [
        '_templates/versions.html',
    ],
}

# autosummary_generate = True

def skip(app, what, name, obj, would_skip, options):
    if '__' in name or name == "clone":
        return True
    return would_skip

def setup(app):
    app.connect('autodoc-skip-member', skip)
