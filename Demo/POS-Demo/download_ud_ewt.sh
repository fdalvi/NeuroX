#! /usr/bin/env bash

echo "Downloading Universal Dependencies English Web Treebank Data (v1.0)"
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/r1.0/en-ud-train.conllu -O en_ewt-ud-train.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/r1.0/en-ud-dev.conllu -O en_ewt-ud-dev.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/r1.0/en-ud-test.conllu -O en_ewt-ud-test.conllu

echo "Done!"
