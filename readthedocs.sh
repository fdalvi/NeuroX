#!/bin/bash -x

set +e

# surpress 'open' command in script 
function open() { echo "$@"; }
export -f open

if [[ -e ./generate_docs.sh ]]; then
  ./generate_docs.sh;
elif [[ -e ./scripts/generate_docs.sh ]]; then
  ./scripts/generate_docs.sh;
fi

true
