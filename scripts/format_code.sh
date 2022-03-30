#!/bin/bash

# exit when any command fails
set -e

if [[ ! -f setup.cfg ]] || [[ ! -d neurox ]] || [[ ! -d scripts ]] || [[ ! -d tests ]]
then
  echo "format_code.sh must be run from the root of the repository"
  exit 1
fi

ufmt format neurox
ufmt format tests
