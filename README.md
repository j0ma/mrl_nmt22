# NAACL 2022 Paper on Multi-Task Learning for NMT

## Initialization
- Create folders `checkpoints` and `experiments`
- Run `git submodule update --init --recursive`
    - This will clone the Moses SMT toolkit (needed for scripts)

## Notes on requirements
- uralicNLP
- pycountry
- sacremoses
- toml
- attrs
- sentencepiece
- fairseq
- click
- lxml
- mtdata
- tqdm
- psutil

## Notes on languages
- Czech
- German
- Estonian
- Finnish
- Inuktitut
- Russian
- Turkish
- Uzbek

## Notes on installation

### HFST
On both Python 3.8 and Python 3.9, it seems like `pip install hfst` still fails despite the manylinux version purportedly on PyPi.

Got things to work following this: https://github.com/hfst/hfst/issues/557#issuecomment-881652221
