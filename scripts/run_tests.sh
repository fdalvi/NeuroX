#!/bin/bash

# exit when any command fails
set -e

if [[ ! -f setup.cfg ]] || [[ ! -d neurox ]] || [[ ! -d scripts ]] || [[ ! -d tests ]]
then
  echo "run_tests.sh must be run from the root of the repository"
  exit 1
fi

python -m pytest -vv --cov