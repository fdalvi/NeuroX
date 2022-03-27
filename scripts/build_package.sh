#!/bin/bash

# exit when any command fails
set -e

if [[ ! -f setup.cfg ]] || [[ ! -d neurox ]] || [[ ! -d scripts ]] || [[ ! -d tests ]]
then
  echo "build_package.sh must be run from the root of the repository"
  exit 1
fi

rm -rf build dist neurox.egg-info
python -m build