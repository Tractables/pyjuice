# Configuration file for the Sphinx documentation builder.

from sphinx_gallery.sorting import FileNameSortKey

# -- Project information

project = 'PyJuice'
copyright = '2021, StarAI'
author = 'StarAI'

release = '2.2.0'
version = '2.2.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery'
]

sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': '../../examples',
    # other configuration options
    'gallery_dirs': 'getting-started/tutorials',
    # sort key
    'within_subsection_order': FileNameSortKey
}

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
    flag = True

    if name == "need_meta_parameters":
        flag = False

    elif name == "num_parameters":
        flag = False

    elif name == "num_param_flows":
        flag = False

    elif "Nodes" in str(obj) and name == "duplicate":
        flag = False

    elif "Nodes" in str(obj) and name == "get_params":
        flag = False

    elif "Nodes" in str(obj) and name == "set_params":
        flag = False

    elif "InputNodes" in str(obj) and name == "set_meta_params":
        flag = False

    elif "Nodes" in str(obj) and name == "init_parameters":
        flag = False

    elif "Nodes" in str(obj) and name == "num_nodes":
        flag = False

    elif "Nodes" in str(obj) and name == "num_edges":
        flag = False

    elif "ProdNodes" in str(obj) and name == "edge_type":
        flag = False

    elif "ProdNodes" in str(obj) and name == "is_block_sparse":
        flag = False

    elif "ProdNodes" in str(obj) and name == "is_sparse":
        flag = False

    elif "SumNodes" in str(obj) and name == "update_parameters":
        flag = False

    elif "SumNodes" in str(obj) and name == "update_param_flows":
        flag = False

    elif "SumNodes" in str(obj) and name == "gather_parameters":
        flag = False
    
    return flag or would_skip

def setup(app):
    app.connect('autodoc-skip-member', skip)
