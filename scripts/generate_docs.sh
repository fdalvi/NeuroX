#!/bin/bash

# exit when any command fails
set -e

if [[ ! -f setup.cfg ]] || [[ ! -d neurox ]] || [[ ! -d scripts ]] || [[ ! -d tests ]]
then
  echo "generate_docs.sh must be run from the root of the repository"
  exit 1
fi


sphinx-apidoc -o docs neurox sphinx-apidoc --templatedir docs/_templates
cp docs/_templates/*.rst docs

sphinx-build docs build

open build/index.html
