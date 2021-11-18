# NAACL 2022 Paper on Multi-Task Learning for NMT

## Notes on requirements
- uralicNLP
- pycountry
- sacremoses

## Notes on languages

```
#################################################

# Constraint grammar
[UralicNLP] Finnish, cg: True
[UralicNLP] Russian, cg: True
[UralicNLP] Czech, cg: True
[UralicNLP] German, cg: True
[UralicNLP] Amharic, cg: True
[UralicNLP] Swahili, cg: False
[UralicNLP] Turkish, cg: False
[UralicNLP] Inuktitut, cg: True

#################################################

# Dictionary

[UralicNLP] Finnish, dictionary: True
[UralicNLP] Russian, dictionary: True
[UralicNLP] Czech, dictionary: False
[UralicNLP] German, dictionary: True
[UralicNLP] Amharic, dictionary: False
[UralicNLP] Swahili, dictionary: False
[UralicNLP] Turkish, dictionary: False
[UralicNLP] Inuktitut, dictionary: False

#################################################

# Morphology

[UralicNLP] Finnish, morph: True
[UralicNLP] Russian, morph: True
[UralicNLP] Czech, morph: True
[UralicNLP] German, morph: True
[UralicNLP] Amharic, morph: True
[UralicNLP] Swahili, morph: False
[UralicNLP] Turkish, morph: False
[UralicNLP] Inuktitut, morph: True

#################################################
```

## Notes on installation

### HFST
On both Python 3.8 and Python 3.9, it seems like `pip install hfst` still fails despite the manylinux version purportedly on PyPi.

Got things to work following this: https://github.com/hfst/hfst/issues/557#issuecomment-881652221
