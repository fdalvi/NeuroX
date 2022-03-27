#!/bin/bash

# exit when any command fails
set -e

if [[ ! -f setup.cfg ]] || [[ ! -d neurox ]] || [[ ! -d scripts ]] || [[ ! -d tests ]]
then
  echo "generate_docs.sh must be run from the root of the repository"
  exit 1
fi


rm -rf docs-generator
sphinx-apidoc -o docs-generator neurox sphinx-apidoc --full -H 'NeuroX toolkit' -A 'Qatar Computing Research Institute, HBKU' --templatedir docs/sphinx_templates

cd docs-generator
mkdir -p _static/css
mkdir -p _static/img

cp -r ../docs/sphinx_templates/*.rst .
cp ../docs/sphinx_templates/custom.css _static/css/custom.css
cp ../logo.svg _static/img/logo.svg

echo "
import os
import sys
sys.path.insert(0,os.path.abspath('..'))
def skip(app, what, name, obj,would_skip, options):
    if name in ( '__init__',):
        return False
    return would_skip
def setup(app):
    app.connect('autodoc-skip-member', skip)
extensions.append('sphinx.ext.napoleon')
html_theme = 'sphinx_book_theme'
html_logo = '_static/img/logo.svg'

html_theme_options = {
    'repository_url': 'https://github.com/fdalvi/NeuroX',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_fullscreen_button': False,
    'home_page_in_toc': True,
    # 'extra_navbar': '<div class=\"version-footer\">Select version: </div>',
}

autodoc_member_order = 'bysource'

html_css_files = ['css/custom.css']
" >> conf.py

make html

open _build/html/index.html