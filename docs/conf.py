from recommonmark.transform import AutoStructify

project = "RII_Pandas"
version = "0.5.0"
copyright = "2021, Munehiro Nishida"
author = "Munehiro Nishida"

master_doc = "index"
html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

html_static_path = ["static"]
htmlhelp_basename = project

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_markdown_tables",
    "sphinx.ext.doctest",
    "recommonmark",
]

# Exclude build directory and Jupyter backup files:
exclude_patterns = ["build", "**.ipynb_checkpoints"]

# Default language for syntax highlighting in reST and Markdown cells:
highlight_language = "none"

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# Work-around until https://github.com/sphinx-doc/sphinx/issues/4229 is solved:
html_scaled_image_link = False

nbsphinx_execute = "never"

nbsphinx_kernel_name = "python3"

autodoc_member_order = "bysource"


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {"auto_toc_tree_section": "Contents", "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
