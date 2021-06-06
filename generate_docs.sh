#!/bin/bash

rm -rf docs-generator
sphinx-apidoc -o docs-generator neurox sphinx-apidoc --full -H 'NeuroX toolkit' -A 'Fahim Dalvi' --templatedir docs/sphinx_templates

cd docs-generator
mkdir -p _static/css
mkdir -p _static/img

original_css_file_location=`python -c "import insegel as _; print(_.__path__[0])"`/static/css/insegel.css
cat ${original_css_file_location} > _static/css/insegel.css
echo "
.admonition {
    padding: 0.5em;
    display: flex;
    margin-top: 0.5em;
}

.admonition > p {
    margin-bottom: 0px;
}

.note {
    background-color: #B3B3B3;
    color: #2D2D2D;
}

.seealso {
    background-color: rgb(59, 104, 144);
}

.admonition-title {
    font-weight: bold !important;
    margin-right: 20px;
}" >> _static/css/insegel.css

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
html_theme = 'insegel'

autodoc_member_order = 'bysource'

html_theme_options = {

}
" >> conf.py

make html

open _build/html/index.html