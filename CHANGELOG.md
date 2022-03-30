# CHANGELOG

## 1.0.9 (14th March, 2022)
### New Features
- _Probeless_ method implementation has been added in the `interpretation` module
- Data balancing and binary annotation methods are now part of the `data` module
- New options for transformers extraction: Decompose layer activations in separate files, Save activations only for certain layers
- Dedicated Activations Writer class that abstracts away representation saving from representation extraction

### Bug fixes
- Fixed Packaging error that included debug code
- Linear regression probe did not supply the correct arguments (Issue #18)

## 1.0.8 (18th January, 2022)
### New Features
- Transformers extractor can now optionally take a tokenizer name instead of just a model name

