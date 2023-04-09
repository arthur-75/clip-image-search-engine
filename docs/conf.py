"""Configuration file of the ipoly project."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ipoly"
copyright = "2022, Thomas Danguilhen"
author = "Thomas Danguilhen"
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys, os

sys.path.insert(0, os.path.abspath("../"))

extensions = [
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_search.extension",
]


autosummary_generate = True
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

import pathlib

# The readme that already exists
readme_path = pathlib.Path(__file__).parent.resolve().parent / "README.md"
# We copy a modified version here
readme_target = pathlib.Path(__file__).parent / "readme.md"

# The readme that already exists
contributing_path = pathlib.Path(__file__).parent.resolve().parent / "CONTRIBUTING.md"
# We copy a modified version here
contributing_target = pathlib.Path(__file__).parent / "contributing.md"

with readme_target.open("w") as outf:
    # Change the title to "Readme"
    outf.write(
        "\n".join(
            [
                "# Readme\n",
            ],
        ),
    )
    lines = []
    for line in readme_path.read_text().split("\n"):
        if line.startswith("# "):
            # Skip title, because we now use "Readme"
            # Could also simply exclude first line for the same effect
            continue
        lines.append(line)
    outf.write("\n".join(lines))

with contributing_target.open("w") as outf:
    # Change the title to "Contributing"
    outf.write(
        "\n".join(
            [
                "# Contributing\n",
            ],
        ),
    )
    lines = []
    for line in contributing_path.read_text().split("\n"):
        if line.startswith("# "):
            # Skip title, because we now use "Contributing"
            # Could also simply exclude first line for the same effect
            continue
        lines.append(line)
    outf.write("\n".join(lines))
