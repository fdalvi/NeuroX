

sh download_ud_ewt.sh

python changeToSent.py --file en_ewt-ud-train.conllu --sIndex 1 --lIndex 3 > tempFile
cut -f1 tempFile > en_ewt-ud-train.conllu.word
cut -f2 tempFile > en_ewt-ud-train.conllu.label
python removeDuplicates.py --s en_ewt-ud-train.conllu.word --l en_ewt-ud-train.conllu.label > tempFile
cut -f1 tempFile > en_ewt-ud-train.conllu.word
cut -f2 tempFile > en_ewt-ud-train.conllu.label

python changeToSent.py --file en_ewt-ud-dev.conllu --sIndex 1 --lIndex 3 > tempFile

cut -f1 tempFile >  en_ewt-ud-dev.conllu.word
cut -f2 tempFile >  en_ewt-ud-dev.conllu.label
python removeDuplicates.py --s en_ewt-ud-dev.conllu.word --l en_ewt-ud-dev.conllu.label > tempFile
cut -f1 tempFile >  en_ewt-ud-dev.conllu.word
cut -f2 tempFile >  en_ewt-ud-dev.conllu.label

python changeToSent.py --file en_ewt-ud-test.conllu --sIndex 1 --lIndex 3 > tempFile

cut -f1 tempFile >  en_ewt-ud-test.conllu.word
cut -f2 tempFile >  en_ewt-ud-test.conllu.label
python removeDuplicates.py --s en_ewt-ud-test.conllu.word --l en_ewt-ud-test.conllu.label > tempFile
cut -f1 tempFile >  en_ewt-ud-test.conllu.word
cut -f2 tempFile >  en_ewt-ud-test.conllu.label


