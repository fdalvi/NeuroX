# aux_classifier

Code to train classifier from activations extracted from https://github.com/fdalvi/nmt-shared-information.

## Options:
 - `--exp_type`: Possible choices: `word`, `charcnn`, `bpe_avg`, `bpe_last`, `char_avg`, `char_last`. bpe/char choices are for aggregating activations for subwords into a word activation.
 - `--train-source`: Path to train source word tokens file
 - `--train-labels`: Path to train labels file (one label per token)
 - `--train-activations`: Path to train activations file
 - `--test-source`: Path to test source word tokens file
 - `--test-labels`: Path to test labels file (one label per token)
 - `--test-activations`:  Path to test activations file
 - `--train-aux-source`: Path to bpe/char train files (if `exp_type` = `bpe*|char*`)
 - `--test-aux-source`: Path to bpe/char test files (if `exp_type` = `bpe*|char*`)
 - `--task-specific-tag`: Tag to assign for unknown words in the test set. For example, for POS, Noun tag is a good candidate for unknown words.
 - `--max-sent-l`: Max sentence length, should match what was used for preprocessing while training the seq2seq model
 - `--output-dir`: Output directory to store models, vocabs, predictions & results
 - `--filter-layers`: Train classifier with only a few layers. Specify layers as comma separated list. e.g. `f1,b2` (Forward layer 1, Backward Layer 2)
 - `--ignore-start-token`: Ignore the first token of every sentence in the source/source_aux/labels files
 - `--is-unidirectional`: Default: `false`. Specifies if the original MT model is unidirectional (or the analysis is decoder-side)
