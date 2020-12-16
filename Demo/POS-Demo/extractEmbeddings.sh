#!/bin/bash -l
#SBATCH -J Extract Embeddings  #name of the job
#SBATCH -o output.txt
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 1
#SBATCH --mem 5000MB
#output file
#queue used
#number of gpus needed
#number of CPUs needed
#amount of RAM needed
module load cuda10.1/toolkit
conda activate neuron-analysis-bert

#LM=roberta-base
LM=bert-base-cased
#LM=xlnet-base-cased
dir=./
expDir=$dir/Representations/$LM

mkdir -p $expDir

cp $dir/en_ewt-ud-train.conllu.word $expDir/en_ewt-ud-train.conllu.word
cp $dir/en_ewt-ud-dev.conllu.word $expDir/en_ewt-ud-dev.conllu.word
cp $dir/en_ewt-ud-test.conllu.word $expDir/en_ewt-ud-test.conllu.word

cp $dir/en_ewt-ud-train.conllu.label $expDir/en_ewt-ud-train.conllu.label
cp $dir/en_ewt-ud-dev.conllu.label $expDir/en_ewt-ud-dev.conllu.label
cp $dir/en_ewt-ud-test.conllu.label $expDir/en_ewt-ud-test.conllu.label

sh changeQuotes-XLNet.sh $expDir/en_ewt-ud-train.conllu.word
sh changeQuotes-XLNet.sh $expDir/en_ewt-ud-dev.conllu.word
sh changeQuotes-XLNet.sh $expDir/en_ewt-ud-test.conllu.word

python extract_representations-hdf5.py --output-type hdf5 --aggregation last $LM $expDir/en_ewt-ud-train.conllu.word $expDir/en_ewt-ud-train.conllu.hdf5

python extract_representations-hdf5.py --output-type hdf5 --aggregation last $LM $expDir/en_ewt-ud-test.conllu.word $expDir/en_ewt-ud-test.conllu.hdf5

python extract_representations-hdf5.py --output-type hdf5 --aggregation last $LM $expDir/en_ewt-ud-dev.conllu.word $expDir/en_ewt-ud-dev.conllu.hdf5


